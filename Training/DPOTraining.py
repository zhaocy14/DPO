import os
import sys
import json
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

# 适配新模型架构的导入（Models.py）
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
from DataModule.DataModule import CombinedDataset
from Model.Models import (
    ImageEmbedding,
    MotorEmbedding,
    EncoderOnlyCandidateGenerator,
    SimilarityModelImage,
    SimilarityModelDriver
)

# ---------------------- 核心配置（更新模型路径+保留原所有配置项） ----------------------
CONFIG = {
    # 设备配置
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",

    # 训练基础配置（原逻辑全保留）
    "batch_size": 1,
    "epochs": 10,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "grad_clip_norm": 1.0,

    # 候选动作生成（原逻辑）
    "num_candidates": 5,
    "sampling_workers": 2,
    "max_train_samples_per_epoch": 500,
    "max_val_batches": 5,
    "val_batch_size": 6,

    # DPO核心参数（原逻辑）
    "dpo_beta": 0.3,
    "alpha": 0.9,  # 相似度加权系数
    "action_match_tolerance": 1e-4,

    # 重复动作检测（原逻辑）
    "repeat_threshold": 0.999,
    "history_cache_size": 1,

    # 模型架构参数（适配Models.py）
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,
    "gen_seq_len": 30,
    "sim_seq_len": 30,
    "embed_dim_sim": 128,
    "num_layers_sim": 3,
    "nhead_sim": 4,
    "similarity_dim": 32,

    # 路径配置（更新预训练模型名为best_model_cos_sim）
    "data_root_dirs": '/data/cyzhao/collector_cydpo/dpo_data',
    "pretrained_model_path": "./saved_models/best_model_cos_sim",  # 改为best_model_cos_sim
    "dpo_model_save_path": "./saved_models/dpo_final_best_model",  # DPO模型保存路径
    "trajectory_save_root": "./trajectory_records",  # 全量轨迹记录根目录
    "loss_records_path": "./trajectory_records/loss_records.npy",  # 损失轨迹
    "action_records_path": "./trajectory_records/action_records.json",  # 动作轨迹
    "similarity_records_path": "./trajectory_records/similarity_records.npy",  # 相似度轨迹
    "logp_records_path": "./trajectory_records/logp_records.npy",  # 对数概率轨迹
    "meta_records_path": "./trajectory_records/meta_records.json"  # 训练元信息
}

# 创建所有记录目录（保留原逻辑的目录结构）
os.makedirs(CONFIG["trajectory_save_root"], exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["dpo_model_save_path"]), exist_ok=True)

# ---------------------- 简化的全局记录容器（仅保留指定字段） ----------------------
loss_records = {
    "train_loss": [],
    "val_loss": [],
    "train_optimized_samples": [],
    "val_total_samples": [],
    "val_prefer_matches_model": [],  # prefer与最高概率动作匹配次数
    "val_match_rate": [],  # 匹配比例
    "lr": []
}

# 历史动作缓存（保留原逻辑的重复动作检测）
HISTORY_CACHE = {
    "actions": [],
    "max_size": CONFIG["history_cache_size"]
}


# ---------------------- 1. 模型加载（仅适配PreTraining路径，保留原加载逻辑） ----------------------
def load_pretrained_models():
    """
    保留原加载逻辑，仅适配PreTraining训练好的新模型（Models.py）和权重路径
    """
    # 初始化Models.py中的新模型（结构完全适配）
    image_embed = ImageEmbedding(
        embed_dim=CONFIG["embed_dim_gen"],
        num_layers=3,
        is_resnet=False
    ).to(CONFIG["device"])

    motor_embed = MotorEmbedding(
        motor_dim=CONFIG["motor_dim"],
        embed_dim=CONFIG["embed_dim_gen"]
    ).to(CONFIG["device"])

    # 策略模型（待训练）
    policy_generator = EncoderOnlyCandidateGenerator(
        embed_dim=CONFIG["embed_dim_gen"],
        nhead=CONFIG["nhead_gen"],
        num_layers=CONFIG["num_layers_gen"],
        motor_dim=CONFIG["motor_dim"],
        max_seq_length=CONFIG["gen_seq_len"]
    ).to(CONFIG["device"])

    # 参考模型（冻结，复用预训练权重）
    ref_generator = EncoderOnlyCandidateGenerator(
        embed_dim=CONFIG["embed_dim_gen"],
        nhead=CONFIG["nhead_gen"],
        num_layers=CONFIG["num_layers_gen"],
        motor_dim=CONFIG["motor_dim"],
        max_seq_length=CONFIG["gen_seq_len"]
    ).to(CONFIG["device"])

    # 相似度模型（冻结）
    img_sim_model = SimilarityModelImage(
        embed_dim=CONFIG["embed_dim_sim"],
        num_frames=CONFIG["sim_seq_len"],
        num_layers=CONFIG["num_layers_sim"],
        nhead=CONFIG["nhead_sim"],
        similarity_dim=CONFIG["similarity_dim"],
        motor_dim=CONFIG["motor_dim"]
    ).to(CONFIG["device"])

    driver_sim_model = SimilarityModelDriver(
        embed_dim=CONFIG["embed_dim_sim"],
        similarity_dim=CONFIG["similarity_dim"]
    ).to(CONFIG["device"])

    # 加载PreTraining的权重（适配其保存格式，保留原加载逻辑）
    try:
        checkpoint = torch.load(CONFIG["pretrained_model_path"], map_location=CONFIG["device"])
        # 兼容PreTraining的两种保存格式（原逻辑）
        if "model_states" in checkpoint:
            image_embed.load_state_dict(checkpoint["model_states"]["image_embed"])
            motor_embed.load_state_dict(checkpoint["model_states"]["motor_embed"])
            policy_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])
            ref_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])
            img_sim_model.load_state_dict(checkpoint["model_states"]["img_sim_model"])
            driver_sim_model.load_state_dict(checkpoint["model_states"]["driver_sim_model"])
        else:
            image_embed.load_state_dict(checkpoint["image_embed"])
            motor_embed.load_state_dict(checkpoint["motor_embed"])
            policy_generator.load_state_dict(checkpoint["candidate_generator"])
            ref_generator.load_state_dict(checkpoint["candidate_generator"])
            img_sim_model.load_state_dict(checkpoint["img_sim_model"])
            driver_sim_model.load_state_dict(checkpoint["driver_sim_model"])
        print(f"✅ 成功加载PreTraining模型：{CONFIG['pretrained_model_path']}")
    except Exception as e:
        raise RuntimeError(f"❌ 预训练模型加载失败：{str(e)}")

    # 冻结非策略模型（原逻辑）
    for model in [image_embed, motor_embed, ref_generator, img_sim_model, driver_sim_model]:
        for param in model.parameters():
            param.requires_grad = False
    print("✅ 非策略模型已冻结，仅训练policy_generator")

    return image_embed, motor_embed, policy_generator, ref_generator, img_sim_model, driver_sim_model


# ---------------------- 2. 数据加载（保留原DPOTraining逻辑） ----------------------
def load_dataset():
    """保留原数据加载逻辑，仅适配路径"""
    data_root = CONFIG["data_root_dirs"]
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"❌ 数据路径不存在：{data_root}")

    # 筛选2025开头的目录（原逻辑）
    data_dir_list = [
        os.path.join(data_root, f) for f in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, f)) and "2025" in f
    ]
    if not data_dir_list:
        raise ValueError(f"❌ {data_root} 下无2025开头的子目录")

    # 加载CombinedDataset（原逻辑）
    all_dataset = CombinedDataset(
        dir_list=data_dir_list,
        frame_len=CONFIG["gen_seq_len"],
        predict_len=CONFIG["sim_seq_len"],
        show=True
    )
    train_dataset = all_dataset.training_dataset
    val_dataset = all_dataset.val_dataset
    print(f"✅ 数据集加载完成 | 训练集：{len(train_dataset)} | 验证集：{len(val_dataset)}")

    # 数据加载器（原逻辑）
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

    # 验证batch_size（原逻辑）
    def check_batch(dataloader, name):
        target_bs = CONFIG["batch_size"] if name == "train" else CONFIG["val_batch_size"]
        for batch in dataloader:
            if batch[0].shape[0] != target_bs:
                raise RuntimeError(f"❌ {name}集batch_size不匹配：实际{batch[0].shape[0]}，期望{target_bs}")
            break

    check_batch(train_loader, "train")
    check_batch(val_loader, "val")

    return train_loader, val_loader


# ---------------------- 3. 核心工具函数（修复所有bug，重点解决维度不匹配） ----------------------
def gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """保留原高斯对数概率计算逻辑"""
    eps = 1e-6
    std = std + eps
    log_prob = -0.5 * torch.log(2 * torch.tensor(np.pi, device=CONFIG["device"])) - torch.log(std) - (
            action - mean) ** 2 / (2 * std ** 2)
    return log_prob.sum(dim=-1)


def get_generator_distribution(generator, image_embedded, motor_embedded):
    """保留原分布参数获取逻辑，适配新模型输出"""
    outputs = generator(
        image_embedded=image_embedded,
        motor_embedded=motor_embedded,
        num_candidates=CONFIG["num_candidates"],
        temperature=1.0
    )
    return outputs['mean'], outputs['std']


def is_repeated_action(action: torch.Tensor) -> bool:
    """修复RuntimeError：带梯度的张量转numpy前先detach()"""
    if not HISTORY_CACHE["actions"]:
        # 修复1：detach()后再转numpy（关键）
        HISTORY_CACHE["actions"].append(action.detach().cpu().numpy())
        return False

    # 修复2：detach()后再转numpy（关键）
    action_np = action.detach().cpu().numpy()
    for hist_action in HISTORY_CACHE["actions"]:
        similarity = np.corrcoef(action_np.flatten(), hist_action.flatten())[0, 1]
        # if similarity > CONFIG["repeat_threshold"]:
        if similarity == 1:
            return True

    # 更新缓存（原逻辑）
    if len(HISTORY_CACHE["actions"]) >= HISTORY_CACHE["max_size"]:
        HISTORY_CACHE["actions"].pop(0)
    HISTORY_CACHE["actions"].append(action_np)
    return False


def select_preferred_rejected(candidates: list[torch.Tensor],
                              img_proj_future: torch.Tensor | tuple,
                              future_driver_last: torch.Tensor,
                              motor_embed: MotorEmbedding,
                              driver_sim_model: SimilarityModelDriver,
                              batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """选择偏好/非偏好动作（适配原始维度 (B,1,2)，不过度squeeze）"""

    # 处理img_proj_future为元组的情况
    if isinstance(img_proj_future, tuple):
        img_proj_future = img_proj_future[0]

    # 维度检查：候选动作保持原始维度 (B, 1, 2)
    assert len(candidates) == CONFIG["num_candidates"], f"候选数={len(candidates)}，需为{CONFIG['num_candidates']}"
    for i, cand in enumerate(candidates):
        # 期望维度: (batch_size, 1, motor_dim)
        assert cand.shape == (batch_size, 1, CONFIG["motor_dim"]), \
            f"候选{i}维度错误：{cand.shape}，需为({batch_size},1,{CONFIG['motor_dim']})"

    assert future_driver_last.shape == (batch_size, CONFIG["motor_dim"]), \
        f"future_driver_last维度错误：{future_driver_last.shape}"

    # 1. 候选动作嵌入 (B, 1, 2) -> (B, 1, embed_dim)
    candidate_embeddings = []
    for cand in candidates:
        emb = motor_embed(cand)  # (B, 1, embed_dim_gen)
        candidate_embeddings.append(emb)

    # 2. 计算图像相似度
    sim_img = []
    for emb in candidate_embeddings:
        # 候选动作投影到similarity_dim空间
        cand_proj = driver_sim_model(emb).squeeze(1)  # (B, similarity_dim)
        # 未来图像特征
        img_proj_squeezed = img_proj_future.squeeze(1)  # (B, similarity_dim)
        # 余弦相似度
        sim = F.cosine_similarity(cand_proj, img_proj_squeezed, dim=1)  # (B,)
        sim_img.append(sim)

    # 3. 计算动作相似度
    # 真实未来动作嵌入
    future_driver_seq = future_driver_last.unsqueeze(1)  # (B, 1, 2)
    future_driver_emb = motor_embed(future_driver_seq)  # (B, 1, embed_dim_gen)
    future_emb_squeezed = future_driver_emb.squeeze(1)  # (B, embed_dim_gen)
    future_norm = F.normalize(future_emb_squeezed, dim=1)  # (B, embed_dim_gen)

    sim_driver = []
    for emb in candidate_embeddings:
        emb_squeezed = emb.squeeze(1)  # (B, embed_dim_gen)
        emb_norm = F.normalize(emb_squeezed, dim=1)  # (B, embed_dim_gen)
        sim = F.cosine_similarity(emb_norm, future_norm, dim=1)  # (B,)
        sim_driver.append(sim)

    # 4. 计算总相似度（加权融合）
    alpha = CONFIG["alpha"]
    sim_img_tensor = torch.stack(sim_img).T  # (B, num_candidates)
    sim_driver_tensor = torch.stack(sim_driver).T  # (B, num_candidates)
    sim_total = alpha * sim_img_tensor + (1 - alpha) * sim_driver_tensor  # (B, num_candidates)

    # 5. 选择偏好/非偏好动作
    preferred_idx = sim_total.argmax(dim=1)  # (B,)
    rejected_idx = sim_total.argmin(dim=1)  # (B,)

    # 6. 提取动作 - 保持维度 (B, 1, 2) 然后squeeze到 (B, 2)
    candidates_tensor = torch.stack(candidates).permute(1, 0, 2, 3)  # (B, num_candidates, 1, 2)

    # Gather preferred: (B, 1, 1, 2) -> squeeze -> (B, 2)
    preferred = torch.gather(
        candidates_tensor, 1,
        preferred_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, CONFIG["motor_dim"])
    ).squeeze(1).squeeze(1)  # (B, 2)

    # Gather rejected: (B, 1, 1, 2) -> squeeze -> (B, 2)
    rejected = torch.gather(
        candidates_tensor, 1,
        rejected_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, CONFIG["motor_dim"])
    ).squeeze(1).squeeze(1)  # (B, 2)

    return preferred, rejected, sim_total


def get_highest_prob_action(candidates: list[torch.Tensor],
                            image_embedded: torch.Tensor,
                            motor_embedded: torch.Tensor,
                            policy_gen: EncoderOnlyCandidateGenerator) -> tuple[torch.Tensor, torch.Tensor]:
    """修复最高概率动作计算中的维度问题"""
    # 候选动作已经是 (B, 1, 2)，需要squeeze到 (B, 2) 用于计算log_prob
    processed_candidates = []
    for cand in candidates:
        # cand: (B, 1, 2) -> squeeze -> (B, 2)
        cand_processed = cand.squeeze(1)  # 只squeeze第1维
        processed_candidates.append(cand_processed)

    candidates_for_logp = processed_candidates

    batch_size = image_embedded.shape[0]
    mean, std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)  # (B,2), (B,2)

    # 计算每个候选动作的概率
    candidate_logps = []
    for cand in candidates_for_logp:  # cand: (B, 2)
        log_prob = gaussian_log_prob(mean, std, cand)  # (B,)
        candidate_logps.append(log_prob)

    # 转换为概率张量并选最高概率动作
    candidate_logps_tensor = torch.stack(candidate_logps).T  # (B, num_candidates)
    highest_prob_idx = candidate_logps_tensor.argmax(dim=1)  # (B,)

    # 从原始candidates中选择（保持维度一致性）
    candidates_tensor = torch.stack(candidates).permute(1, 0, 2, 3)  # (B, 5, 1, 2)
    highest_prob_action = torch.gather(
        candidates_tensor, 1,
        highest_prob_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, CONFIG["motor_dim"])
    ).squeeze(1).squeeze(1)  # (B, 2)

    return highest_prob_action, candidate_logps_tensor


def dpo_loss(policy_chosen_logps: torch.Tensor,
             policy_rejected_logps: torch.Tensor,
             ref_chosen_logps: torch.Tensor,
             ref_rejected_logps: torch.Tensor) -> torch.Tensor:
    """
    计算DPO损失（适配单组动作的对数概率）
    公式参考：https://arxiv.org/abs/2305.18290
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(CONFIG["dpo_beta"] * logits)
    return loss.mean()


# ---------------------- 4. 训练逻辑（简化记录，仅保留指定字段） ----------------------
def train_one_epoch(epoch, train_loader, models, optimizer):
    """训练一个epoch，仅记录指定的训练字段"""
    image_embed, motor_embed, policy_gen, ref_gen, img_sim_model, driver_sim_model = models
    policy_gen.train()
    # 冻结模型设为eval模式
    for model in [image_embed, motor_embed, ref_gen, img_sim_model, driver_sim_model]:
        model.eval()

    total_loss = 0.0
    batch_count = 0
    optimized_samples = 0  # 累计优化样本数
    pbar = tqdm(
        enumerate(train_loader),
        desc=f"Train Epoch {epoch + 1}/{CONFIG['epochs']}",
        total=min(CONFIG["max_train_samples_per_epoch"], len(train_loader))
    )

    for batch_idx, batch in pbar:
        if batch_count >= CONFIG["max_train_samples_per_epoch"]:
            break

        # 解包数据（原逻辑）
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        images = torch.stack([imgs1, imgs2], dim=2).to(CONFIG["device"])
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(CONFIG["device"])
        driver = driver.to(CONFIG["device"])
        future_driver_last = future_driver[:, 0, :].to(CONFIG["device"])
        batch_size = images.shape[0]

        optimizer.zero_grad()

        # 1. 特征嵌入（冻结模型的前向传播）
        with torch.no_grad():
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)
            future_image_embedded = image_embed(future_images)
            img_proj_future = img_sim_model(future_image_embedded)
            if isinstance(img_proj_future, tuple):
                img_proj_future = img_proj_future[0]

        # 2. 生成候选动作（policy模型）
        policy_outputs = policy_gen(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=CONFIG["num_candidates"]
        )
        policy_candidates = policy_outputs['candidates']  # list of (B, 1, 2)

        # 3. 选择偏好/非偏好动作（保持原始维度）
        preferred, rejected, _ = select_preferred_rejected(
            candidates=policy_candidates,
            img_proj_future=img_proj_future,
            future_driver_last=future_driver_last,
            motor_embed=motor_embed,
            driver_sim_model=driver_sim_model,
            batch_size=batch_size
        )

        # 4. 重复动作检测（对preferred动作）
        is_repeat = False
        if batch_size == 1:
            is_repeat = is_repeated_action(preferred)
        else:
            # 批量处理时逐个检测
            repeat_flags = [is_repeated_action(preferred[i:i + 1]) for i in range(batch_size)]
            is_repeat = any(repeat_flags)

        # 重复则跳过当前batch的反向传播（原逻辑）
        if is_repeat:
            print(f"\n⚠️ Batch {batch_idx} 检测到重复动作，跳过优化")
            batch_count += 1
            continue

        # 5. 计算对数概率（policy需要梯度，ref不需要梯度）
        # Policy模型：需要梯度，必须在no_grad外面
        policy_mean, policy_std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)

        # Ref模型：冻结，不需要梯度
        with torch.no_grad():
            ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)

        # 计算policy的logp（需要梯度）
        policy_chosen_logps = gaussian_log_prob(policy_mean, policy_std, preferred)
        policy_rejected_logps = gaussian_log_prob(policy_mean, policy_std, rejected)

        # 计算ref的logp（不需要梯度）
        with torch.no_grad():
            ref_chosen_logps = gaussian_log_prob(ref_mean, ref_std, preferred)
            ref_rejected_logps = gaussian_log_prob(ref_mean, ref_std, rejected)

        # 6. 计算DPO损失并反向传播
        loss = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_gen.parameters(), CONFIG["grad_clip_norm"])
        optimizer.step()

        # 累计损失和样本数
        total_loss += loss.item() * batch_size
        optimized_samples += batch_size
        batch_count += 1

        # 更新进度条
        pbar.set_postfix({"train_loss": loss.item(), "optimized_samples": optimized_samples})

    # ---------------------- 记录当前epoch的训练数据 ----------------------
    epoch_avg_loss = total_loss / max(optimized_samples, 1)  # 防止除0
    loss_records["train_loss"].append(epoch_avg_loss)
    loss_records["train_optimized_samples"].append(optimized_samples)
    loss_records["lr"].append(optimizer.param_groups[0]['lr'])  # 记录当前学习率

    print(f"\nEpoch {epoch + 1} Train | Avg Loss: {epoch_avg_loss:.4f} | Optimized Samples: {optimized_samples}")
    return epoch_avg_loss


# ---------------------- 5. 验证逻辑（简化记录，仅保留指定字段） ----------------------
def val_one_epoch(epoch, val_loader, models):
    """验证一个epoch，记录指定的验证字段"""
    image_embed, motor_embed, policy_gen, ref_gen, img_sim_model, driver_sim_model = models
    # 所有模型设为eval
    for model in [image_embed, motor_embed, policy_gen, ref_gen, img_sim_model, driver_sim_model]:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    prefer_matches_model = 0  # prefer与最高概率动作匹配次数
    pbar = tqdm(
        enumerate(val_loader),
        desc=f"Val Epoch {epoch + 1}/{CONFIG['epochs']}",
        total=min(CONFIG["max_val_batches"], len(val_loader))
    )

    with torch.no_grad():
        for batch_idx, batch in pbar:
            if batch_idx >= CONFIG["max_val_batches"]:
                break

            # 解包数据（同训练逻辑）
            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
            images = torch.stack([imgs1, imgs2], dim=2).to(CONFIG["device"])
            future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(CONFIG["device"])
            driver = driver.to(CONFIG["device"])
            future_driver_last = future_driver[:, 0, :].to(CONFIG["device"])
            batch_size = images.shape[0]

            # 1. 特征嵌入
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)
            future_image_embedded = image_embed(future_images)
            img_proj_future = img_sim_model(future_image_embedded)
            if isinstance(img_proj_future, tuple):
                img_proj_future = img_proj_future[0]

            # 2. 生成候选动作
            policy_outputs = policy_gen(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=CONFIG["num_candidates"]
            )
            policy_candidates = policy_outputs['candidates']

            # 3. 选择prefer/rejected动作 + 计算最高概率动作
            preferred, rejected, _ = select_preferred_rejected(
                candidates=policy_candidates,
                img_proj_future=img_proj_future,
                future_driver_last=future_driver_last,
                motor_embed=motor_embed,
                driver_sim_model=driver_sim_model,
                batch_size=batch_size
            )

            # 获取模型最高概率动作
            highest_prob_action, _ = get_highest_prob_action(
                candidates=policy_candidates,
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                policy_gen=policy_gen
            )

            # 4. 统计prefer与最高概率动作的匹配次数
            # 按batch逐个样本比较（容忍微小误差）
            match = torch.isclose(preferred, highest_prob_action, atol=CONFIG["action_match_tolerance"]).all(dim=1)
            prefer_matches_model += match.sum().item()

            # 5. 计算DPO损失
            policy_mean, policy_std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)
            ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)

            policy_chosen_logps = gaussian_log_prob(policy_mean, policy_std, preferred)
            policy_rejected_logps = gaussian_log_prob(policy_mean, policy_std, rejected)
            ref_chosen_logps = gaussian_log_prob(ref_mean, ref_std, preferred)
            ref_rejected_logps = gaussian_log_prob(ref_mean, ref_std, rejected)

            loss = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps
            )

            # 累计损失和样本数
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # 更新进度条
            pbar.set_postfix({"val_loss": loss.item(), "match_count": prefer_matches_model})

    # ---------------------- 记录当前epoch的验证数据 ----------------------
    epoch_avg_loss = total_loss / max(total_samples, 1)
    match_rate = prefer_matches_model / max(total_samples, 1)  # 匹配比例
    loss_records["val_loss"].append(epoch_avg_loss)
    loss_records["val_total_samples"].append(total_samples)
    loss_records["val_prefer_matches_model"].append(prefer_matches_model)
    loss_records["val_match_rate"].append(match_rate)

    print(f"Epoch {epoch + 1} Val | Avg Loss: {epoch_avg_loss:.4f} | Match Rate: {match_rate:.4f}")
    return epoch_avg_loss


# ---------------------- 6. 主训练函数 ----------------------
def main():
    # 1. 加载模型和数据
    models = load_pretrained_models()
    train_loader, val_loader = load_dataset()

    # 2. 初始化优化器
    optimizer = torch.optim.AdamW(
        models[2].parameters(),  # 仅优化policy_generator
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    # 3. 训练主循环
    best_val_loss = float('inf')
    for epoch in range(CONFIG["epochs"]):
        # 训练一个epoch
        train_one_epoch(epoch, train_loader, models, optimizer)
        # 验证一个epoch
        val_loss = val_one_epoch(epoch, val_loader, models)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "policy_gen_state_dict": models[2].state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "loss_records": loss_records
            }, CONFIG["dpo_model_save_path"])
            print(f"✅ 保存最优模型（Val Loss: {best_val_loss:.4f}）")

    # 4. 保存最终的loss_records到文件
    # 保存为JSON（方便查看）+ NPY（方便后续数值分析）
    with open(os.path.join(CONFIG["trajectory_save_root"], "loss_records.json"), "w") as f:
        json.dump({
            k: [float(vv) for vv in v] if isinstance(v, list) else v
            for k, v in loss_records.items()
        }, f, indent=4)
    np.save(os.path.join(CONFIG["trajectory_save_root"], "loss_records.npy"), loss_records)

    print(f"\n📊 训练完成！记录已保存到 {CONFIG['trajectory_save_root']}")
    print("最终记录：", loss_records)


if __name__ == "__main__":
    main()