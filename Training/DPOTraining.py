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
    # 验证控制
    "max_val_batches": 50,
    "val_batch_size": 12,

    # 相似度加权系数
    "alpha": 0.9,
    "action_match_tolerance": 1e-4,  # 动作匹配的容差（处理浮点数精度）

    "batch_size": 1,
    "epochs": 50,
    "lr": 5e-4,
    "num_candidates": 5,
    "sampling_workers": 2,
    "max_train_samples_per_epoch": 100,
    "dpo_beta": 0.1,
    "repeat_threshold": 0.95,
    "history_cache_size": 10,
    "use_candidates": "candidates1",
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
    "data_root_dirs": '/data/cyzhao/collector_cydpo/dpo_data',
    "pretrained_model_path": "./saved_models/best_model",
    "dpo_save_path": "./saved_models/dpo_final_best_model",
    "dpo_loss_path": "./loss_records/dpo_final_loss.npy"
}

# 创建保存目录
os.makedirs(os.path.dirname(CONFIG["dpo_save_path"]), exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["dpo_loss_path"]), exist_ok=True)


# ---------------------- 1. 模型加载 ----------------------
def load_pretrained_models(pretrained_path):
    image_embed = ImageEmbedding(
        embed_dim=CONFIG["embed_dim_gen"],
        num_layers=3,
        is_resnet=False
    ).to(device)

    motor_embed = MotorEmbedding(
        motor_dim=CONFIG["motor_dim"],
        embed_dim=CONFIG["embed_dim_gen"]
    ).to(device)

    policy_generator = EncoderOnlyCandidateGenerator(
        embed_dim=CONFIG["embed_dim_gen"],
        nhead=CONFIG["nhead_gen"],
        num_layers=CONFIG["num_layers_gen"],
        motor_dim=CONFIG["motor_dim"],
        max_seq_length=CONFIG["gen_seq_len"]
    ).to(device)

    ref_generator = EncoderOnlyCandidateGenerator(
        embed_dim=CONFIG["embed_dim_gen"],
        nhead=CONFIG["nhead_gen"],
        num_layers=CONFIG["num_layers_gen"],
        motor_dim=CONFIG["motor_dim"],
        max_seq_length=CONFIG["gen_seq_len"]
    ).to(device)

    img_sim_model = SimilarityModelImage(
        embed_dim=CONFIG["embed_dim_sim"],
        num_frames=CONFIG["sim_seq_len"],
        num_layers=CONFIG["num_layers_sim"],
        nhead=CONFIG["nhead_sim"],
        similarity_dim=CONFIG["similarity_dim"]
    ).to(device)

    driver_sim_model = SimilarityModelDriver(
        embed_dim=CONFIG["embed_dim_sim"],
        similarity_dim=CONFIG["similarity_dim"]
    ).to(device)

    # 加载权重
    try:
        checkpoint = torch.load(pretrained_path, map_location=device)
        image_embed.load_state_dict(checkpoint["model_states"]["image_embed"])
        motor_embed.load_state_dict(checkpoint["model_states"]["motor_embed"])
        policy_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])
        ref_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])
        img_sim_model.load_state_dict(checkpoint["model_states"]["img_sim_model"])
        driver_sim_model.load_state_dict(checkpoint["model_states"]["driver_sim_model"])
        print(f"[模型加载] 成功加载预训练模型：{pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"[模型加载失败] {str(e)}") from e

    # 冻结非策略模型
    for model in [image_embed, motor_embed, ref_generator, img_sim_model, driver_sim_model]:
        for param in model.parameters():
            param.requires_grad = False
    print("[模型配置] 仅EncoderOnlyCandidateGenerator可训练")

    return image_embed, motor_embed, policy_generator, ref_generator, img_sim_model, driver_sim_model


# ---------------------- 2. 数据加载 ----------------------
def load_dataset():
    data_root = CONFIG["data_root_dirs"]
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"[数据路径错误] {data_root} 不存在")

    data_dir_list = [os.path.join(data_root, f) for f in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, f)) and "2025" in f]
    if not data_dir_list:
        raise ValueError(f"[数据筛选错误] {data_root} 下无含'2025'的子目录")

    try:
        all_dataset = CombinedDataset(
            dir_list=data_dir_list,
            frame_len=CONFIG["gen_seq_len"],
            predict_len=CONFIG["sim_seq_len"],
            show=True
        )
        train_dataset = all_dataset.training_dataset
        val_dataset = all_dataset.val_dataset
        print(f"[数据加载] 训练集：{len(train_dataset)}样本 | 验证集：{len(val_dataset)}样本")
    except Exception as e:
        raise RuntimeError(f"[数据集加载失败] {str(e)}") from e

    # 检查batch_size
    def check_batch_size(dataloader, name, expected_size):
        for batch in dataloader:
            imgs1 = batch[0]
            if imgs1.shape[0] != expected_size:
                raise RuntimeError(f"[BatchSize错误] {name}的batch_size={imgs1.shape[0]}，需为{expected_size}")
            break

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

    check_batch_size(train_loader, "训练集", CONFIG["batch_size"])
    check_batch_size(val_loader, "验证集", CONFIG["val_batch_size"])
    print(f"[数据配置] 训练集batch_size={CONFIG['batch_size']} | 验证集batch_size={CONFIG['val_batch_size']}")

    return train_loader, val_loader


# ---------------------- 3. 核心工具函数（新增动作匹配逻辑） ----------------------
def gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    std = std + eps
    log_prob = -0.5 * torch.log(2 * torch.tensor(np.pi, device=device)) - torch.log(std) - (action - mean) ** 2 / (
                2 * std ** 2)
    return log_prob.sum(dim=-1)


def get_generator_distribution(generator: EncoderOnlyCandidateGenerator,
                               image_embedded: torch.Tensor,
                               motor_embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat([motor_embedded, image_embedded], dim=-1)
    combined = generator.positional_encoding(combined)
    encoder_out = generator.encoder(combined)
    encoder_out = encoder_out[-1]
    global_feat = encoder_out.mean(dim=1)

    if CONFIG["use_candidates"] == "candidates1":
        mean = generator.fc_mean1(global_feat)
        logvar = generator.fc_logvar1(global_feat)
    else:
        mean = generator.fc_mean2(global_feat)
        logvar = generator.fc_logvar2(global_feat)

    logvar = torch.clamp(logvar, min=-5, max=5)
    std = torch.exp(0.5 * logvar)
    return mean, std


def select_preferred_rejected(candidates: list[torch.Tensor],
                              img_proj_future: torch.Tensor,
                              future_driver_last: torch.Tensor,
                              motor_embed: MotorEmbedding,
                              batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    新增返回值：sim_total（各候选的总相似度）
    用于后续确定模型生成的最高概率动作
    """
    # 1. 维度检查
    assert len(candidates) == CONFIG["num_candidates"], f"候选数={len(candidates)}，需为{CONFIG['num_candidates']}"
    for i, cand in enumerate(candidates):
        assert cand.shape == (batch_size, CONFIG["motor_dim"]), \
            f"候选{i}维度错误：{cand.shape}，需为({batch_size},{CONFIG['motor_dim']})"
    assert future_driver_last.shape == (batch_size, CONFIG["motor_dim"]), \
        f"future_driver_last维度错误：{future_driver_last.shape}"

    # 2. 候选动作嵌入
    candidate_embeddings = []
    for cand in candidates:
        cand_with_seq = cand.unsqueeze(1)  # (B,1,2)
        emb = motor_embed(cand_with_seq)  # (B,1,128)
        candidate_embeddings.append(emb)

    # 3. 计算图像相似度
    sim_img = []
    for emb in candidate_embeddings:
        cand_proj = driver_sim_model(emb).squeeze(1)  # (B,32)
        img_proj_future_squeezed = img_proj_future.squeeze(1)  # (B,32)
        sim = F.cosine_similarity(cand_proj, img_proj_future_squeezed, dim=1)  # (B,)
        sim_img.append(sim)

    # 4. 计算动作相似度
    future_driver_with_seq = future_driver_last.unsqueeze(1)  # (B,1,2)
    future_driver_emb = motor_embed(future_driver_with_seq)  # (B,1,128)
    future_emb_squeezed = future_driver_emb.squeeze(1)  # (B,128)
    future_norm = F.normalize(future_emb_squeezed, dim=1)  # (B,128)

    sim_driver = []
    for emb in candidate_embeddings:
        emb_squeezed = emb.squeeze(1)  # (B,128)
        emb_norm = F.normalize(emb_squeezed, dim=1)  # (B,128)
        sim = F.cosine_similarity(emb_norm, future_norm, dim=1)  # (B,)
        sim_driver.append(sim)

    # 5. 计算总相似度（用于确定preferred和返回）
    alpha = CONFIG["alpha"]
    sim_img_tensor = torch.stack(sim_img).T  # (B, 5)
    sim_driver_tensor = torch.stack(sim_driver).T  # (B, 5)
    sim_total = alpha * sim_img_tensor + (1 - alpha) * sim_driver_tensor  # (B, 5)

    # 6. 选择偏好/非偏好动作
    preferred_idx = sim_total.argmax(dim=1)  # (B,)
    rejected_idx = sim_total.argmin(dim=1)  # (B,)

    # 7. 提取动作
    candidates_tensor = torch.stack(candidates).permute(1, 0, 2)  # (B,5,2)
    preferred = torch.gather(
        candidates_tensor, 1,
        preferred_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, CONFIG["motor_dim"])
    ).squeeze(1)  # (B,2)
    rejected = torch.gather(
        candidates_tensor, 1,
        rejected_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, CONFIG["motor_dim"])
    ).squeeze(1)  # (B,2)

    return preferred, rejected, sim_total  # 新增返回sim_total


def get_model_highest_prob_action(candidates: list[torch.Tensor],
                                  candidates_tensor: torch.Tensor,
                                  image_embedded: torch.Tensor,
                                  motor_embedded: torch.Tensor,
                                  policy_gen: EncoderOnlyCandidateGenerator) -> torch.Tensor:
    """
    计算模型对每个候选动作的概率，返回最高概率的动作
    candidates: 候选动作列表 [5, (B,2)]
    candidates_tensor: 候选动作张量 (B,5,2)
    """
    batch_size = image_embedded.shape[0]
    num_candidates = len(candidates)

    # 1. 获取模型的分布参数（均值和标准差）
    mean, std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)  # (B,2), (B,2)

    # 2. 计算每个候选动作的概率（对数概率）
    candidate_probs = []
    for cand in candidates:
        # 计算当前候选的对数概率 (B,)
        log_prob = gaussian_log_prob(mean, std, cand)
        candidate_probs.append(log_prob)

    # 3. 转换为概率分布（B,5）
    probs_tensor = torch.stack(candidate_probs).T  # (B,5)
    probs_tensor = F.softmax(probs_tensor, dim=1)  # 归一化到概率分布

    # 4. 找到最高概率的候选索引
    highest_prob_idx = probs_tensor.argmax(dim=1)  # (B,)

    # 5. 提取最高概率动作
    highest_prob_action = torch.gather(
        candidates_tensor, 1,
        highest_prob_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, CONFIG["motor_dim"])
    ).squeeze(1)  # (B,2)

    return highest_prob_action, probs_tensor


def is_action_repeated(current_action: torch.Tensor, history_actions: list[torch.Tensor]) -> bool:
    assert current_action.shape == (CONFIG["motor_dim"],), f"current_action维度错误：{current_action.shape}"
    if not history_actions:
        return False
    current_norm = F.normalize(current_action.unsqueeze(0), dim=1)  # (1,2)
    for hist_action in history_actions:
        assert hist_action.shape == (CONFIG["motor_dim"],), f"历史动作维度错误：{hist_action.shape}"
        hist_norm = F.normalize(hist_action.unsqueeze(0), dim=1)  # (1,2)
        sim = F.cosine_similarity(current_norm, hist_norm, dim=1).item()
        if sim > CONFIG["repeat_threshold"]:
            return True
    return False


def standard_dpo_loss(policy_gen: EncoderOnlyCandidateGenerator,
                      ref_gen: EncoderOnlyCandidateGenerator,
                      image_embedded: torch.Tensor,
                      motor_embedded: torch.Tensor,
                      preferred: torch.Tensor,
                      rejected: torch.Tensor) -> torch.Tensor:
    batch_size = preferred.shape[0]
    assert preferred.shape == (batch_size, CONFIG["motor_dim"]), f"preferred维度错误：{preferred.shape}"
    assert rejected.shape == (batch_size, CONFIG["motor_dim"]), f"rejected维度错误：{rejected.shape}"

    policy_mean, policy_std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)
    log_p_theta_pref = gaussian_log_prob(policy_mean, policy_std, preferred)  # (B,)
    log_p_theta_rej = gaussian_log_prob(policy_mean, policy_std, rejected)  # (B,)

    with torch.no_grad():
        ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)
        log_p_ref_pref = gaussian_log_prob(ref_mean, ref_std, preferred)
        log_p_ref_rej = gaussian_log_prob(ref_mean, ref_std, rejected)

    advantage = (log_p_theta_pref - log_p_ref_pref) - (log_p_theta_rej - log_p_ref_rej)  # (B,)
    return -F.logsigmoid(CONFIG["dpo_beta"] * advantage).mean()


# ---------------------- 4. 训练/验证函数（核心：新增匹配统计） ----------------------
def train_one_epoch(epoch: int,
                    train_loader: DataLoader,
                    policy_gen: EncoderOnlyCandidateGenerator,
                    ref_gen: EncoderOnlyCandidateGenerator,
                    optimizer: torch.optim.Optimizer,
                    motor_embed: MotorEmbedding) -> tuple[float, int]:
    policy_gen.train()
    total_loss = 0.0
    optimized_count = 0
    history_actions = []
    batch_size = CONFIG["batch_size"]

    pbar = tqdm(enumerate(train_loader),
                desc=f"[训练] Epoch {epoch + 1}/{CONFIG['epochs']}",
                total=min(CONFIG["max_train_samples_per_epoch"], len(train_loader)))

    for sample_idx, batch in pbar:
        if sample_idx >= CONFIG["max_train_samples_per_epoch"]:
            print(f"\n[训练] 已达样本上限（{CONFIG['max_train_samples_per_epoch']}），终止")
            break

        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        assert imgs1.shape[0] == batch_size, f"训练集batch_size错误：{imgs1.shape[0]}"

        images = torch.stack([imgs1, imgs2], dim=2).to(device)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        driver = driver.to(device)
        future_driver = future_driver.to(device)
        future_driver_last = future_driver[:, -1, :]

        with torch.no_grad():
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)
            future_image_embedded = image_embed(future_images)
            img_proj_future = img_sim_model(future_image_embedded)

        generator_output = policy_gen(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=CONFIG["num_candidates"],
            temperature=1.0
        )
        candidates = generator_output[CONFIG["use_candidates"]]
        candidates = [cand.squeeze(1) for cand in candidates]

        preferred, rejected, _ = select_preferred_rejected(
            candidates=candidates,
            img_proj_future=img_proj_future,
            future_driver_last=future_driver_last,
            motor_embed=motor_embed,
            batch_size=batch_size
        )

        current_action = preferred.squeeze(0)
        if is_action_repeated(current_action, history_actions):
            pbar.set_postfix({"状态": "跳过重复动作", "优化样本数": optimized_count})
            continue

        optimizer.zero_grad()
        loss = standard_dpo_loss(
            policy_gen=policy_gen,
            ref_gen=ref_gen,
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            preferred=preferred,
            rejected=rejected
        )
        loss.backward()
        optimizer.step()

        history_actions.append(current_action.detach())
        if len(history_actions) > CONFIG["history_cache_size"]:
            history_actions.pop(0)

        total_loss += loss.item()
        optimized_count += 1
        pbar.set_postfix({
            "DPO损失": f"{loss.item():.4f}",
            "优化样本数": optimized_count
        })

    avg_loss = total_loss / optimized_count if optimized_count > 0 else 0.0
    print(f"[训练] Epoch {epoch + 1} | 平均损失：{avg_loss:.4f} | 优化样本数：{optimized_count}")
    return avg_loss, optimized_count


def validate_full(epoch: int,
                  val_loader: DataLoader,
                  policy_gen: EncoderOnlyCandidateGenerator,
                  ref_gen: EncoderOnlyCandidateGenerator,
                  motor_embed: MotorEmbedding) -> tuple[float, int, int]:
    """
    新增返回值：match_count（prefer动作与模型最高概率动作匹配的次数）
    """
    policy_gen.eval()
    total_loss = 0.0
    sample_count = 0
    match_count = 0  # 记录匹配次数
    max_batches = CONFIG["max_val_batches"]
    batch_size = CONFIG["val_batch_size"]
    tolerance = CONFIG["action_match_tolerance"]  # 浮点数比较容差

    with torch.no_grad():
        pbar_total = min(max_batches, len(val_loader))
        pbar = tqdm(enumerate(val_loader),
                    desc=f"[验证] Epoch {epoch + 1}",
                    total=pbar_total)

        for sample_idx, batch in pbar:
            if sample_idx >= max_batches:
                print(f"\n[验证] 已达最大batch数（{max_batches}），终止验证")
                break

            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
            assert imgs1.shape[0] == batch_size, f"验证集batch_size错误：{imgs1.shape[0]}"

            # 数据预处理
            images = torch.stack([imgs1, imgs2], dim=2).to(device)
            future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
            driver = driver.to(device)
            future_driver = future_driver.to(device)
            future_driver_last = future_driver[:, -1, :]

            # 特征嵌入
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)
            future_image_embedded = image_embed(future_images)
            img_proj_future = img_sim_model(future_image_embedded)

            # 生成候选动作
            generator_output = policy_gen(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=CONFIG["num_candidates"],
                temperature=1.0
            )
            candidates = generator_output[CONFIG["use_candidates"]]
            candidates = [cand.squeeze(1) for cand in candidates]  # [5, (B,2)]
            candidates_tensor = torch.stack(candidates).permute(1, 0, 2)  # (B,5,2)

            # 1. 选择preferred action
            preferred, rejected, _ = select_preferred_rejected(
                candidates=candidates,
                img_proj_future=img_proj_future,
                future_driver_last=future_driver_last,
                motor_embed=motor_embed,
                batch_size=batch_size
            )  # (B,2)

            # 2. 获取模型最高概率action
            highest_prob_action, _ = get_model_highest_prob_action(
                candidates=candidates,
                candidates_tensor=candidates_tensor,
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                policy_gen=policy_gen
            )  # (B,2)

            # 3. 比较两个动作是否匹配（逐样本检查）
            # 对每个样本，检查所有维度是否在容差范围内
            batch_matches = torch.allclose(
                preferred,
                highest_prob_action,
                atol=tolerance,  # 绝对容差
                rtol=0  # 相对容差
            )  # 若批次所有样本都匹配则为True，这里我们需要逐样本统计

            # 逐样本统计匹配次数
            for i in range(batch_size):
                if torch.allclose(
                        preferred[i],
                        highest_prob_action[i],
                        atol=tolerance,
                        rtol=0
                ):
                    match_count += 1

            # 计算损失
            loss = standard_dpo_loss(
                policy_gen=policy_gen,
                ref_gen=ref_gen,
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                preferred=preferred,
                rejected=rejected
            )
            total_loss += loss.item()
            sample_count += batch_size

            # 计算当前批次的匹配率
            batch_match_rate = (match_count - (sample_count - batch_size)) / batch_size
            pbar.set_postfix({
                "验证损失": f"{loss.item():.4f}",
                "已验证样本": sample_count,
                "批次匹配率": f"{batch_match_rate:.2%}"
            })

    avg_loss = total_loss / (sample_count / batch_size) if sample_count > 0 else 0.0
    match_rate = match_count / sample_count if sample_count > 0 else 0.0
    print(f"[验证] Epoch {epoch + 1} | 平均损失：{avg_loss:.4f} | 总样本数：{sample_count}")
    print(f"[验证] 匹配统计：prefer与模型最高概率动作匹配 {match_count}/{sample_count}（{match_rate:.2%}）")
    return avg_loss, sample_count, match_count  # 新增返回match_count


# ---------------------- 5. 主函数（记录匹配统计） ----------------------
def main():
    start_total_time = time.time()
    print("\n" + "=" * 60)
    print("                EncoderOnlyCandidateGenerator DPO优化（带匹配统计）")
    print("=" * 60)
    print(f"[配置信息] 验证集batch_size={CONFIG['val_batch_size']} | 动作匹配容差={CONFIG['action_match_tolerance']}")

    # 加载模型和数据
    try:
        global image_embed, motor_embed, img_sim_model, driver_sim_model
        image_embed, motor_embed, policy_generator, ref_generator, img_sim_model, driver_sim_model = load_pretrained_models(
            CONFIG["pretrained_model_path"]
        )
        train_loader, val_loader = load_dataset()
    except Exception as e:
        print(f"[初始化失败] {str(e)}")
        return

    # 优化器配置
    optimizer = torch.optim.Adam(
        params=policy_generator.parameters(),
        lr=CONFIG["lr"],
        weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=2,
        gamma=0.5
    )

    # 损失记录（新增匹配统计字段）
    loss_records = {
        "train_loss": [],
        "val_loss": [],
        "train_optimized_samples": [],
        "val_total_samples": [],
        "val_prefer_matches_model": [],  # 新增：prefer与模型最高概率动作匹配的次数
        "val_match_rate": [],  # 新增：匹配比例
        "lr": []
    }
    best_val_loss = float("inf")

    # 训练循环
    for epoch in range(CONFIG["epochs"]):
        print("\n" + "-" * 50)
        epoch_start_time = time.time()

        # 训练
        train_loss, optimized_samples = train_one_epoch(
            epoch=epoch,
            train_loader=train_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            optimizer=optimizer,
            motor_embed=motor_embed
        )

        # 学习率调度
        # scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 验证（获取匹配统计）
        val_loss, val_samples, val_matches = validate_full(
            epoch=epoch,
            val_loader=val_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            motor_embed=motor_embed
        )
        val_match_rate = val_matches / val_samples if val_samples > 0 else 0.0

        # 记录（新增匹配相关字段）
        loss_records["train_loss"].append(train_loss)
        loss_records["val_loss"].append(val_loss)
        loss_records["train_optimized_samples"].append(optimized_samples)
        loss_records["val_total_samples"].append(val_samples)
        loss_records["val_prefer_matches_model"].append(val_matches)  # 保存匹配次数
        loss_records["val_match_rate"].append(val_match_rate)  # 保存匹配比例
        loss_records["lr"].append(current_lr)

        # 打印总结
        epoch_time = time.time() - epoch_start_time
        print(f"\n[Epoch 总结] Epoch {epoch + 1}/{CONFIG['epochs']}")
        print(f"  - 耗时：{epoch_time:.2f}秒 | 学习率：{current_lr:.7f}")
        print(f"  - 训练：{train_loss:.4f}（{optimized_samples}样本）")
        print(f"  - 验证：{val_loss:.4f}（{val_samples}样本）")
        print(f"  - 匹配统计：{val_matches}/{val_samples}（{val_match_rate:.2%}）")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "policy_generator_state_dict": policy_generator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": CONFIG,
                "loss_records": loss_records
            }, CONFIG["dpo_save_path"])
            print(f"  - ✅ 保存最佳模型至：{CONFIG['dpo_save_path']}")

    # 训练完成
    total_time = time.time() - start_total_time
    np.save(CONFIG["dpo_loss_path"], loss_records)

    print("\n" + "=" * 60)
    print("                          DPO优化训练完成")
    print("=" * 60)
    print(f"总耗时：{total_time:.2f}秒 | 最佳验证损失：{best_val_loss:.4f}")
    print(f"总匹配统计：{sum(loss_records['val_prefer_matches_model'])}/{sum(loss_records['val_total_samples'])}")
    print(f"模型路径：{CONFIG['dpo_save_path']} | 损失记录：{CONFIG['dpo_loss_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
