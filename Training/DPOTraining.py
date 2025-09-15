import os
import sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataModule.DataModule import CombinedDataset
from Model.Models import (ImageEmbedding, MotorEmbedding,
                          EncoderOnlyCandidateGenerator,
                          SimilarityModelImage, SimilarityModelDriver)
from tqdm import tqdm


# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------------------- 核心配置参数 ----------------------
CONFIG = {
    # 训练控制（逐帧模式）
    "batch_size": 1,  # 单样本输入（逐帧）
    "epochs": 5,  # DPO微调epoch数（无需过多）
    "lr": 5e-7,  # DPO学习率（需小，避免参数震荡）
    "num_candidates": 5,  # 生成动作候选数（固定为5）
    "sampling_workers": 2,  # 数据加载worker（单样本场景无需过多）
    "max_train_samples_per_epoch": 500,  # 每epoch训练样本上限（控制时间）

    # 验证控制（全量模式）
    "val_batch_size": 1,  # 验证仍用单样本（匹配训练输入格式）

    # DPO核心参数
    "dpo_beta": 0.1,  # 标准DPO温度系数（控制偏好强度）
    "repeat_threshold": 0.95,  # 动作重复判断阈值（余弦相似度>0.95视为重复）
    "history_cache_size": 10,  # 历史动作缓存大小（避免内存占用）
    "use_candidates": "candidates1",  # 使用模型输出的候选组（candidates1/candidates2）

    # 模型结构参数（与预训练一致）
    "embed_dim_gen": 128,  # 生成器嵌入维度
    "nhead_gen": 8,  # 生成器Transformer头数
    "num_layers_gen": 16,  # 生成器Transformer层数
    "motor_dim": 2,  # 原始动作维度（x/y两维）
    "gen_seq_len": 30,  # 观测序列长度（输入帧数量）
    "sim_seq_len": 30,  # 预测序列长度（未来帧数量）
    "embed_dim_sim": 128,  # 相似度模型嵌入维度
    "num_layers_sim": 3,  # 相似度模型Transformer层数
    "nhead_sim": 4,  # 相似度模型Transformer头数
    "similarity_dim": 32,  # 相似度投射后维度

    # 路径配置（用户指定）
    "data_root_dirs": '/data/cyzhao/collector_cydpo/dpo_data',  # 新数据路径
    "pretrained_model_path": "./saved_models/best_model",  # 预训练模型路径
    "dpo_save_path": "./saved_models/dpo_final_best_model",  # DPO最佳模型保存路径
    "dpo_loss_path": "./loss_records/dpo_final_loss.npy"  # DPO损失记录路径
}

# 创建必要的保存目录（避免路径不存在错误）
os.makedirs(os.path.dirname(CONFIG["dpo_save_path"]), exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["dpo_loss_path"]), exist_ok=True)


# ---------------------- 1. 模型加载（含标准DPO参考模型） ----------------------
def load_pretrained_models(pretrained_path):
    """
    加载预训练模型，冻结非策略模型（仅优化EncoderOnlyCandidateGenerator）
    返回：所有模型实例（策略模型+参考模型+冻结模型）
    """
    # 初始化图像嵌入模型（冻结）
    image_embed = ImageEmbedding(
        embed_dim=CONFIG["embed_dim_gen"],
        num_layers=3,
        is_resnet=False
    ).to(device)

    # 初始化动作嵌入模型（冻结，用于动作维度转换）
    motor_embed = MotorEmbedding(
        motor_dim=CONFIG["motor_dim"],
        embed_dim=CONFIG["embed_dim_gen"]
    ).to(device)

    # 初始化策略模型（待优化：EncoderOnlyCandidateGenerator）
    policy_generator = EncoderOnlyCandidateGenerator(
        embed_dim=CONFIG["embed_dim_gen"],
        nhead=CONFIG["nhead_gen"],
        num_layers=CONFIG["num_layers_gen"],
        motor_dim=CONFIG["motor_dim"],
        max_seq_length=CONFIG["gen_seq_len"]
    ).to(device)

    # 初始化参考模型（冻结，与策略模型权重一致）
    ref_generator = EncoderOnlyCandidateGenerator(
        embed_dim=CONFIG["embed_dim_gen"],
        nhead=CONFIG["nhead_gen"],
        num_layers=CONFIG["num_layers_gen"],
        motor_dim=CONFIG["motor_dim"],
        max_seq_length=CONFIG["gen_seq_len"]
    ).to(device)

    # 初始化图像相似度模型（冻结）
    img_sim_model = SimilarityModelImage(
        embed_dim=CONFIG["embed_dim_sim"],
        num_frames=CONFIG["sim_seq_len"],
        num_layers=CONFIG["num_layers_sim"],
        nhead=CONFIG["nhead_sim"],
        similarity_dim=CONFIG["similarity_dim"]
    ).to(device)

    # 初始化动作相似度模型（冻结）
    driver_sim_model = SimilarityModelDriver(
        embed_dim=CONFIG["embed_dim_sim"],
        similarity_dim=CONFIG["similarity_dim"]
    ).to(device)

    # 加载预训练权重（确保与原训练模型兼容）
    try:
        checkpoint = torch.load(pretrained_path, map_location=device)
        image_embed.load_state_dict(checkpoint["model_states"]["image_embed"])
        motor_embed.load_state_dict(checkpoint["model_states"]["motor_embed"])
        policy_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])
        ref_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])  # 参考模型用相同权重
        img_sim_model.load_state_dict(checkpoint["model_states"]["img_sim_model"])
        driver_sim_model.load_state_dict(checkpoint["model_states"]["driver_sim_model"])
        print(f"[模型加载] 成功加载预训练模型：{pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"[模型加载失败] {str(e)}") from e

    # 冻结非策略模型（仅策略模型可训练）
    freeze_models = [image_embed, motor_embed, ref_generator, img_sim_model, driver_sim_model]
    for model in freeze_models:
        for param in model.parameters():
            param.requires_grad = False
    print("[模型配置] 已冻结非策略模型，仅优化EncoderOnlyCandidateGenerator")

    return image_embed, motor_embed, policy_generator, ref_generator, img_sim_model, driver_sim_model


# ---------------------- 2. 数据加载（新路径+全量验证） ----------------------
def load_dataset():
    """
    加载数据集（使用用户指定的新路径）
    返回：训练集DataLoader（逐帧）、验证集DataLoader（全量）
    """
    data_root = CONFIG["data_root_dirs"]
    # 检查数据路径是否存在
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"[数据路径错误] 数据根目录不存在：{data_root}")

    # 筛选含"2025"的子目录（与原训练数据筛选逻辑一致）
    data_dir_list = []
    for subdir in os.listdir(data_root):
        subdir_path = os.path.join(data_root, subdir)
        if os.path.isdir(subdir_path) and "2025" in subdir:
            data_dir_list.append(subdir_path)
    if not data_dir_list:
        raise ValueError(f"[数据筛选错误] 新路径下无含'2025'的子目录：{data_root}")

    # 加载CombinedDataset
    try:
        all_dataset = CombinedDataset(
            dir_list=data_dir_list,
            frame_len=CONFIG["gen_seq_len"],
            predict_len=CONFIG["sim_seq_len"],
            show=True
        )
        train_dataset = all_dataset.training_dataset
        val_dataset = all_dataset.val_dataset
        print(f"[数据加载] 成功加载数据：")
        print(f"  - 训练集样本数：{len(train_dataset)} | 验证集样本数：{len(val_dataset)}")
        print(f"  - 观测序列长度：{CONFIG['gen_seq_len']} | 预测序列长度：{CONFIG['sim_seq_len']}")
    except Exception as e:
        raise RuntimeError(f"[数据集加载失败] {str(e)}") from e

    # 构建DataLoader（训练逐帧，验证全量）
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["sampling_workers"],
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CONFIG["val_batch_size"],
        shuffle=False,
        num_workers=CONFIG["sampling_workers"],
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader


# ---------------------- 3. DPO核心工具函数 ----------------------
def gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    计算动作在高斯分布下的对数概率（适配EncoderOnlyCandidateGenerator的输出）
    参数：
        mean: 高斯分布均值 (batch, motor_dim)
        std: 高斯分布标准差 (batch, motor_dim)
        action: 动作张量 (batch, motor_dim)
    返回：
        log_prob: 对数概率 (batch,)
    """
    eps = 1e-6  # 避免log(0)
    std = std + eps
    log_prob = -0.5 * torch.log(2 * torch.tensor(np.pi, device=device)) - torch.log(std) - (action - mean) ** 2 / (
                2 * std ** 2)
    return log_prob.sum(dim=-1)  # 对动作维度求和


def get_generator_distribution(generator: EncoderOnlyCandidateGenerator,
                               image_embedded: torch.Tensor,
                               motor_embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    获取生成器输出的高斯分布参数（均值+标准差），适配候选组选择
    返回：mean (batch, motor_dim), std (batch, motor_dim)
    """
    # 融合动作+图像嵌入特征
    combined = torch.cat([motor_embedded, image_embedded], dim=-1)  # (batch, seq_len, 3*embed_dim_gen)
    combined = generator.positional_encoding(combined)  # 加位置编码
    # Encoder提取特征
    encoder_out = generator.encoder(combined)  # (batch, seq_len, 3*embed_dim_gen)
    encoder_out = encoder_out[-1]  # 取最后一层输出
    global_feat = encoder_out.mean(dim=1)  # 时序维度平均（全局特征）

    # 根据候选组选择对应的输出层
    if CONFIG["use_candidates"] == "candidates1":
        mean = generator.fc_mean1(global_feat)
        logvar = generator.fc_logvar1(global_feat)
    else:
        mean = generator.fc_mean2(global_feat)
        logvar = generator.fc_logvar2(global_feat)

    # 限制logvar范围，避免标准差过大/过小
    logvar = torch.clamp(logvar, min=-5, max=5)
    std = torch.exp(0.5 * logvar)
    return mean, std


def select_preferred_rejected(candidates: list[torch.Tensor],
                              img_proj_future: torch.Tensor,
                              future_driver_last: torch.Tensor,
                              motor_embed: MotorEmbedding) -> tuple[torch.Tensor, torch.Tensor]:
    """
    选择偏好动作（Preferred）和非偏好动作（Rejected）
    选择逻辑：综合「图像相似度+动作相似度」，最大为Preferred，最小为Rejected
    参数：
        candidates: 动作候选列表（长度=num_candidates，每个元素=(batch, motor_dim)）
        img_proj_future: 未来图像投射向量 (batch, similarity_dim)
        future_driver_last: 未来动作最后一帧 (batch, motor_dim)
        motor_embed: 动作嵌入模型（用于维度转换）
    返回：
        preferred: 偏好动作 (batch, motor_dim)
        rejected: 非偏好动作 (batch, motor_dim)
    """
    # 1. 候选动作维度转换（2维→128维，适配相似度模型）
    candidate_embeddings = []
    for cand in candidates:
        # 增加时间维度（匹配motor_embed输入格式：(batch, seq_len, motor_dim)）
        cand_with_seq = cand.unsqueeze(1)  # (batch, 1, motor_dim)
        emb = motor_embed(cand_with_seq)  # (batch, 1, embed_dim_gen)
        candidate_embeddings.append(emb)

    # 2. 计算「候选动作-未来图像」相似度（图像相似度）
    sim_img = []
    for emb in candidate_embeddings:
        # 动作相似度模型输入：(batch, seq_len, embed_dim_gen) → 输出：(batch, similarity_dim)
        cand_proj = driver_sim_model(emb)
        # 余弦相似度（batch内计算）
        sim = F.cosine_similarity(cand_proj, img_proj_future, dim=1)  # (batch,)
        sim_img.append(sim.item())  # 提取标量值，避免张量维度问题

    # 3. 计算「候选动作-未来动作」相似度（动作相似度）
    # 未来动作维度转换
    future_driver_with_seq = future_driver_last.unsqueeze(1)  # (batch, 1, motor_dim)
    future_driver_emb = motor_embed(future_driver_with_seq)  # (batch, 1, embed_dim_gen)
    future_norm = F.normalize(future_driver_emb.squeeze(1), dim=1)  # (batch, embed_dim_gen)
    # 候选动作相似度计算
    sim_driver = []
    for emb in candidate_embeddings:
        emb_norm = F.normalize(emb.squeeze(1), dim=1)  # (batch, embed_dim_gen)
        sim = F.cosine_similarity(emb_norm, future_norm, dim=1)  # (batch,)
        sim_driver.append(sim.item())  # 提取标量值

    # 4. 综合相似度排序
    sim_total = torch.tensor(sim_img, device=device) + torch.tensor(sim_driver, device=device)  # (num_candidates,)
    preferred_idx = sim_total.argmax().item()  # 偏好动作索引（综合最大）
    rejected_idx = sim_total.argmin().item()  # 非偏好动作索引（综合最小）

    # 5. 提取最终动作（移除batch维度，返回单样本动作）
    preferred = candidates[preferred_idx].squeeze(0)  # (motor_dim,)
    rejected = candidates[rejected_idx].squeeze(0)  # (motor_dim,)
    return preferred, rejected


def is_action_repeated(current_action: torch.Tensor, history_actions: list[torch.Tensor]) -> bool:
    """
    判断当前动作是否与历史动作重复（避免重复优化）
    参数：
        current_action: 当前动作 (motor_dim,)
        history_actions: 历史动作列表（每个元素=(motor_dim,)）
    返回：
        True/False: 是否重复
    """
    if not history_actions:  # 历史缓存为空，无重复
        return False

    # 标准化动作（消除尺度影响）
    current_norm = F.normalize(current_action.unsqueeze(0), dim=1)  # (1, motor_dim)
    for hist_action in history_actions:
        hist_norm = F.normalize(hist_action.unsqueeze(0), dim=1)  # (1, motor_dim)
        sim = F.cosine_similarity(current_norm, hist_norm, dim=1).item()
        if sim > CONFIG["repeat_threshold"]:
            print(f"[动作重复] 相似度={sim:.4f} > 阈值={CONFIG['repeat_threshold']}，跳过优化")
            return True
    return False


def standard_dpo_loss(policy_gen: EncoderOnlyCandidateGenerator,
                      ref_gen: EncoderOnlyCandidateGenerator,
                      image_embedded: torch.Tensor,
                      motor_embedded: torch.Tensor,
                      preferred: torch.Tensor,
                      rejected: torch.Tensor) -> torch.Tensor:
    """
    标准DPO损失（基于策略模型与参考模型的相对概率）
    公式：-log(sigmoid(β * [(log P_θ(pref) - log P_ref(pref)) - (log P_θ(rej) - log P_ref(rej))]))
    返回：平均损失（标量）
    """
    # 1. 策略模型的对数概率（可训练，有梯度）
    policy_mean, policy_std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)
    log_p_theta_pref = gaussian_log_prob(policy_mean, policy_std, preferred.unsqueeze(0))  # (batch,)
    log_p_theta_rej = gaussian_log_prob(policy_mean, policy_std, rejected.unsqueeze(0))  # (batch,)

    # 2. 参考模型的对数概率（冻结，无梯度）
    with torch.no_grad():
        ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)
        log_p_ref_pref = gaussian_log_prob(ref_mean, ref_std, preferred.unsqueeze(0))  # (batch,)
        log_p_ref_rej = gaussian_log_prob(ref_mean, ref_std, rejected.unsqueeze(0))  # (batch,)

    # 3. 计算相对优势（Advantage）
    advantage = (log_p_theta_pref - log_p_ref_pref) - (log_p_theta_rej - log_p_ref_rej)  # (batch,)

    # 4. 计算标准DPO损失（负对数似然）
    loss = -F.logsigmoid(CONFIG["dpo_beta"] * advantage).mean()  # 对batch取平均
    return loss


# ---------------------- 4. 训练/验证函数 ----------------------
def train_one_epoch(epoch: int,
                    train_loader: DataLoader,
                    policy_gen: EncoderOnlyCandidateGenerator,
                    ref_gen: EncoderOnlyCandidateGenerator,
                    optimizer: torch.optim.Optimizer,
                    motor_embed: MotorEmbedding) -> tuple[float, int]:
    """
    逐帧训练一个epoch（单样本输入，重复动作跳过）
    返回：平均训练损失、实际优化的样本数
    """
    policy_gen.train()  # 策略模型设为训练模式
    total_loss = 0.0
    optimized_count = 0  # 实际优化的样本数（排除重复）
    history_actions = []  # 历史动作缓存

    # 进度条（按限制样本数显示）
    pbar = tqdm(enumerate(train_loader),
                desc=f"[训练] Epoch {epoch + 1}/{CONFIG['epochs']}",
                total=min(CONFIG["max_train_samples_per_epoch"], len(train_loader)))

    for sample_idx, batch in pbar:
        # 1. 控制每epoch训练样本数
        if sample_idx >= CONFIG["max_train_samples_per_epoch"]:
            print(f"\n[训练] 已达每epoch样本上限（{CONFIG['max_train_samples_per_epoch']}），终止训练")
            break

        # 2. 解包数据（保留driver原始数据，不置零）
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        # 图像数据：(batch, seq_len, 3, H, W) → (batch, seq_len, 2, 3, H, W)（双相机）
        images = torch.stack([imgs1, imgs2], dim=2).to(device)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        # 动作数据：保留原始值（不置零）
        driver = driver.to(device)  # (batch, gen_seq_len, motor_dim)
        future_driver = future_driver.to(device)  # (batch, sim_seq_len, motor_dim)
        future_driver_last = future_driver[:, -1, :]  # 未来动作最后一帧（用于相似度计算）

        # 3. 特征嵌入（冻结模型，无梯度）
        with torch.no_grad():
            image_embedded = image_embed(images)  # (batch, gen_seq_len, 2*embed_dim_gen)
            motor_embedded = motor_embed(driver)  # (batch, gen_seq_len, embed_dim_gen)
            # 未来图像投射（用于相似度计算）
            future_image_embedded = image_embed(future_images)  # (batch, sim_seq_len, 2*embed_dim_gen)
            img_proj_future = img_sim_model(future_image_embedded)  # (batch, similarity_dim)

        # 4. 生成动作候选（从策略模型采样5个）
        generator_output = policy_gen(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=CONFIG["num_candidates"],
            temperature=1.0
        )
        # 提取指定候选组（candidates1/candidates2）
        candidates = generator_output[CONFIG["use_candidates"]]  # 列表：(batch,1,motor_dim) × 5
        candidates = [cand.squeeze(1) for cand in candidates]  # 移除时间维度：(batch,motor_dim) ×5

        # 5. 选择偏好/非偏好动作
        preferred, rejected = select_preferred_rejected(
            candidates=candidates,
            img_proj_future=img_proj_future,
            future_driver_last=future_driver_last,
            motor_embed=motor_embed
        )

        # 6. 重复动作跳过优化
        if is_action_repeated(preferred, history_actions):
            pbar.set_postfix({"状态": "跳过重复动作", "优化样本数": optimized_count})
            continue

        # 7. 计算DPO损失并反向传播
        optimizer.zero_grad()  # 清零梯度
        loss = standard_dpo_loss(
            policy_gen=policy_gen,
            ref_gen=ref_gen,
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            preferred=preferred,
            rejected=rejected
        )
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 8. 更新历史动作缓存（保持固定大小）
        history_actions.append(preferred.detach())  # detach避免梯度残留
        if len(history_actions) > CONFIG["history_cache_size"]:
            history_actions.pop(0)  # 移除最早的动作

        # 9. 累计损失和计数
        total_loss += loss.item()
        optimized_count += 1
        # 更新进度条
        pbar.set_postfix({
            "DPO损失": f"{loss.item():.4f}",
            "优化样本数": optimized_count
        })

    # 计算平均损失（仅统计实际优化的样本）
    avg_loss = total_loss / optimized_count if optimized_count > 0 else 0.0
    print(f"[训练] Epoch {epoch + 1} 完成 | 平均损失：{avg_loss:.4f} | 优化样本数：{optimized_count}")
    return avg_loss, optimized_count


def validate_full(epoch: int,
                  val_loader: DataLoader,
                  policy_gen: EncoderOnlyCandidateGenerator,
                  ref_gen: EncoderOnlyCandidateGenerator,
                  motor_embed: MotorEmbedding) -> tuple[float, int]:
    """
    全量验证（遍历整个验证集，无样本数限制）
    返回：平均验证损失、验证总样本数
    """
    policy_gen.eval()  # 策略模型设为评估模式
    total_loss = 0.0
    sample_count = 0  # 验证总样本数

    # 关闭梯度计算（加速+节省内存）
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader),
                    desc=f"[验证] Epoch {epoch + 1}",
                    total=len(val_loader))  # 全量验证，进度条按总样本数显示

        for sample_idx, batch in pbar:
            # 1. 解包数据（与训练一致，保留原始driver）
            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
            images = torch.stack([imgs1, imgs2], dim=2).to(device)
            future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
            driver = driver.to(device)
            future_driver = future_driver.to(device)
            future_driver_last = future_driver[:, -1, :]

            # 2. 特征嵌入（与训练一致）
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)
            future_image_embedded = image_embed(future_images)
            img_proj_future = img_sim_model(future_image_embedded)

            # 3. 生成动作候选
            generator_output = policy_gen(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=CONFIG["num_candidates"],
                temperature=1.0
            )
            candidates = generator_output[CONFIG["use_candidates"]]
            candidates = [cand.squeeze(1) for cand in candidates]

            # 4. 选择偏好/非偏好动作
            preferred, rejected = select_preferred_rejected(
                candidates=candidates,
                img_proj_future=img_proj_future,
                future_driver_last=future_driver_last,
                motor_embed=motor_embed
            )

            # 5. 计算验证损失
            loss = standard_dpo_loss(
                policy_gen=policy_gen,
                ref_gen=ref_gen,
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                preferred=preferred,
                rejected=rejected
            )
            total_loss += loss.item()
            sample_count += 1

            # 更新进度条
            pbar.set_postfix({
                "验证损失": f"{loss.item():.4f}",
                "已验证样本": sample_count
            })

    # 计算平均验证损失
    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    print(f"[验证] Epoch {epoch + 1} 完成 | 平均损失：{avg_loss:.4f} | 总样本数：{sample_count}")
    return avg_loss, sample_count


# ---------------------- 5. 主函数（程序入口） ----------------------
def main():
    # 1. 初始化日志
    start_total_time = time.time()
    print("\n" + "=" * 60)
    print("                      EncoderOnlyCandidateGenerator 标准DPO优化")
    print("=" * 60)

    # 2. 加载模型和数据
    try:
        # 全局变量：方便训练/验证函数调用（避免过多参数传递）
        global image_embed, motor_embed, img_sim_model, driver_sim_model
        image_embed, motor_embed, policy_generator, ref_generator, img_sim_model, driver_sim_model = load_pretrained_models(
            CONFIG["pretrained_model_path"]
        )
        train_loader, val_loader = load_dataset()
    except Exception as e:
        print(f"[初始化失败] {str(e)}")
        return

    # 3. 配置优化器和学习率调度器
    optimizer = torch.optim.Adam(
        params=policy_generator.parameters(),  # 仅优化策略模型
        lr=CONFIG["lr"],
        betas=(0.9, 0.999),
        weight_decay=1e-6  # 轻微权重衰减，防止过拟合
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=2,  # 每2个epoch衰减一次
        gamma=0.5  # 衰减系数
    )

    # 4. 初始化损失记录
    loss_records = {
        "train_loss": [],  # 每个epoch的训练平均损失
        "val_loss": [],  # 每个epoch的验证平均损失
        "train_optimized_samples": [],  # 每个epoch的优化样本数
        "val_total_samples": [],  # 每个epoch的验证总样本数
        "lr": []  # 每个epoch的学习率（便于分析）
    }
    best_val_loss = float("inf")  # 最佳验证损失（初始为无穷大）

    # 5. 训练循环
    for epoch in range(CONFIG["epochs"]):
        print("\n" + "-" * 50)
        epoch_start_time = time.time()

        # 5.1 训练一个epoch
        train_loss, optimized_samples = train_one_epoch(
            epoch=epoch,
            train_loader=train_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            optimizer=optimizer,
            motor_embed=motor_embed
        )

        # 5.2 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 5.3 全量验证
        val_loss, val_samples = validate_full(
            epoch=epoch,
            val_loader=val_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            motor_embed=motor_embed
        )

        # 5.4 记录损失和学习率
        loss_records["train_loss"].append(train_loss)
        loss_records["val_loss"].append(val_loss)
        loss_records["train_optimized_samples"].append(optimized_samples)
        loss_records["val_total_samples"].append(val_samples)
        loss_records["lr"].append(current_lr)

        # 5.5 打印epoch总结
        epoch_time = time.time() - epoch_start_time
        print(f"\n[Epoch 总结] Epoch {epoch + 1}/{CONFIG['epochs']}")
        print(f"  - 耗时：{epoch_time:.2f} 秒")
        print(f"  - 学习率：{current_lr:.7f}")
        print(f"  - 训练：平均损失={train_loss:.4f} | 优化样本数={optimized_samples}")
        print(f"  - 验证：平均损失={val_loss:.4f} | 总样本数={val_samples}")

        # 5.6 保存最佳模型（基于验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存模型权重和训练状态
            torch.save({
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "policy_generator_state_dict": policy_generator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": CONFIG,  # 保存配置，便于复现
                "loss_records": loss_records  # 保存已有的损失记录
            }, CONFIG["dpo_save_path"])
            print(f"  - ✅ 保存最佳模型至：{CONFIG['dpo_save_path']}")

    # 6. 训练完成后处理
    total_time = time.time() - start_total_time
    # 保存最终损失记录
    np.save(CONFIG["dpo_loss_path"], loss_records)

    # 打印最终总结
    print("\n" + "=" * 60)
    print("                          DPO优化训练完成")
    print("=" * 60)
    print(f"总耗时：{total_time:.2f} 秒（约 {total_time / 60:.1f} 分钟）")
    print(f"最佳验证损失：{best_val_loss:.4f}")
    print(f"最佳模型路径：{CONFIG['dpo_save_path']}")
    print(f"损失记录路径：{CONFIG['dpo_loss_path']}")
    print(f"训练配置：{CONFIG}")
    print("=" * 60)


if __name__ == "__main__":
    main()