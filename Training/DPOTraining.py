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

# ---------------------- 核心配置（保留原DPOTraining所有配置项） ----------------------
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
    "max_train_samples_per_epoch": 100,
    "max_val_batches": 20,
    "val_batch_size": 6,

    # DPO核心参数（原逻辑）
    "dpo_beta": 0.1,
    "alpha": 0.0,  # 相似度加权系数
    "action_match_tolerance": 1e-4,

    # 重复动作检测（原逻辑）
    "repeat_threshold": 0.97,
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

    # 路径配置（仅适配PreTraining路径，保留原存储结构）
    "data_root_dirs": '/data/cyzhao/collector_cydpo/dpo_data',
    "pretrained_model_path": "./saved_models/best_model",  # PreTraining训练好的模型路径
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

# ---------------------- 全局轨迹记录容器（保留原逻辑，记录所有训练数据） ----------------------
# 初始化全量轨迹记录（batch级粒度，原DPOTraining完整记录）
TRAIN_RECORDS = {
    "train": {
        "loss": {"batch": [], "epoch": []},  # batch级+epoch级损失
        "action": {"candidates": [], "preferred": [], "rejected": [], "highest_prob": []},  # 动作轨迹
        "similarity": {"img": [], "driver": [], "total": []},  # 相似度轨迹
        "logp": {"policy_chosen": [], "policy_rejected": [], "ref_chosen": [], "ref_rejected": []},  # logp轨迹
        "meta": {"epoch": [], "batch_idx": [], "timestamp": []}  # 元信息
    },
    "val": {
        "loss": {"batch": [], "epoch": []},
        "action": {"candidates": [], "preferred": [], "rejected": [], "highest_prob": []},
        "similarity": {"img": [], "driver": [], "total": []},
        "logp": {"policy_chosen": [], "policy_rejected": [], "ref_chosen": [], "ref_rejected": []},
        "meta": {"epoch": [], "batch_idx": [], "timestamp": []}
    }
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


# ---------------------- 3. 核心工具函数（修复select_preferred_rejected函数bug） ----------------------
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
    """保留原重复动作检测逻辑"""
    if not HISTORY_CACHE["actions"]:
        HISTORY_CACHE["actions"].append(action.cpu().numpy())
        return False

    # 计算与历史动作的相似度（原逻辑）
    action_np = action.cpu().numpy()
    for hist_action in HISTORY_CACHE["actions"]:
        similarity = np.corrcoef(action_np.flatten(), hist_action.flatten())[0, 1]
        if similarity > CONFIG["repeat_threshold"]:
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
    """选择偏好/非偏好动作（修复img_proj_future为元组的bug）"""
    # 1. 维度检查（适配单组候选）
    assert len(candidates) == CONFIG["num_candidates"], f"候选数={len(candidates)}，需为{CONFIG['num_candidates']}"
    for i, cand in enumerate(candidates):
        assert cand.shape == (batch_size, CONFIG["motor_dim"]), \
            f"候选{i}维度错误：{cand.shape}，需为({batch_size},{CONFIG['motor_dim']})"
    assert future_driver_last.shape == (batch_size, CONFIG["motor_dim"]), \
        f"future_driver_last维度错误：{future_driver_last.shape}"

    # ========== 核心修复：处理img_proj_future为元组的情况 ==========
    # 如果是元组，提取第一个元素（核心投影张量）
    if isinstance(img_proj_future, tuple):
        img_proj_future = img_proj_future[0]
    # ==============================================================

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
        img_proj_future_squeezed = img_proj_future.squeeze(1)  # (B,32) 现在可正常调用squeeze
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

    # 5. 计算总相似度
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

    return preferred, rejected, sim_total


def get_highest_prob_action(candidates: list[torch.Tensor],
                            image_embedded: torch.Tensor,
                            motor_embedded: torch.Tensor,
                            policy_gen: EncoderOnlyCandidateGenerator) -> tuple[torch.Tensor, torch.Tensor]:
    """保留原最高概率动作计算逻辑"""
    batch_size = image_embedded.shape[0]
    mean, std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)  # (B,2), (B,2)

    # 计算每个候选动作的概率
    candidate_logps = []
    for cand in candidates:
        log_prob = gaussian_log_prob(mean, std, cand)  # (B,)
        candidate_logps.append(log_prob)

    # 转换为概率张量并选最高概率动作
    candidate_logps_tensor = torch.stack(candidate_logps).T  # (B, num_candidates)
    highest_prob_idx = candidate_logps_tensor.argmax(dim=1)  # (B,)

    candidates_tensor = torch.stack(candidates).permute(1, 0, 2)  # (B,5,2)
    highest_prob_action = torch.gather(
        candidates_tensor, 1,
        highest_prob_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, CONFIG["motor_dim"])
    ).squeeze(1)  # (B,2)

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


# ---------------------- 4. 训练/验证逻辑（保留原DPOTraining所有逻辑，完整记录） ----------------------
def train_one_epoch(epoch, train_loader, models, optimizer):
    """保留原训练逻辑，完整记录所有batch级数据"""
    image_embed, motor_embed, policy_gen, ref_gen, img_sim_model, driver_sim_model = models
    policy_gen.train()
    image_embed.eval()
    motor_embed.eval()
    ref_gen.eval()
    img_sim_model.eval()
    driver_sim_model.eval()

    total_loss = 0.0
    batch_count = 0
    optimized_samples = 0
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
        images = torch.stack([imgs1, imgs2], dim=2).to(CONFIG["device"])  # (B, seq, 2, 3, H, W)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(CONFIG["device"])
        driver = driver.to(CONFIG["device"])
        future_driver_last = future_driver[:, 0, :].to(CONFIG["device"])  # (B, 2)
        batch_size = images.shape[0]

        optimizer.zero_grad()

        # 1. 特征嵌入（冻结模型的前向传播）
        with torch.no_grad():
            image_embedded = image_embed(images)  # (B, seq, embed_dim)
            motor_embedded = motor_embed(driver)  # (B, seq, embed_dim)
            future_image_embedded = image_embed(future_images)  # (B, sim_seq, embed_dim)
            img_proj_future = img_sim_model(future_image_embedded)  # 可能返回元组

        # 2. 生成候选动作（policy模型）
        policy_outputs = policy_gen(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=CONFIG["num_candidates"]
        )
        policy_candidates = policy_outputs['candidates']  # list[(B,2), ...]

        # 3. 重复动作检测（原逻辑）
        candidates_mean = torch.stack(policy_candidates).mean(dim=0)  # (B,2)
        if is_repeated_action(candidates_mean):
            print(f"⚠️ 跳过重复动作 | Epoch {epoch + 1} Batch {batch_idx}")
            continue
        optimized_samples += 1

        # 4. 选择偏好/非偏好动作
        preferred_action, rejected_action, sim_total = select_preferred_rejected(
            candidates=policy_candidates,
            img_proj_future=img_proj_future,
            future_driver_last=future_driver_last,
            motor_embed=motor_embed,
            driver_sim_model=driver_sim_model,
            batch_size=batch_size
        )

        # 5. 计算最高概率动作
        highest_prob_action, candidate_logps = get_highest_prob_action(
            candidates=policy_candidates,
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            policy_gen=policy_gen
        )

        # 6. 计算Policy模型的对数概率
        policy_mean, policy_std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)
        policy_chosen_logp = gaussian_log_prob(policy_mean, policy_std, preferred_action)  # (B,)
        policy_rejected_logp = gaussian_log_prob(policy_mean, policy_std, rejected_action)  # (B,)

        # 7. 计算Reference模型的对数概率（冻结）
        with torch.no_grad():
            ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)
            ref_chosen_logp = gaussian_log_prob(ref_mean, ref_std, preferred_action)  # (B,)
            ref_rejected_logp = gaussian_log_prob(ref_mean, ref_std, rejected_action)  # (B,)

        # 8. 计算DPO损失
        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logp,
            policy_rejected_logps=policy_rejected_logp,
            ref_chosen_logps=ref_chosen_logp,
            ref_rejected_logps=ref_rejected_logp
        )

        # 9. 反向传播与优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_gen.parameters(), max_norm=CONFIG["grad_clip_norm"])  # 梯度裁剪
        optimizer.step()

        # ---------------------- 核心：记录所有训练数据（原DPOTraining完整记录） ----------------------
        # 记录损失
        TRAIN_RECORDS["train"]["loss"]["batch"].append(loss.item())
        # 记录动作（转numpy方便保存）
        candidates_tensor = torch.stack(policy_candidates).permute(1, 0, 2).cpu().numpy()  # (B,5,2)
        TRAIN_RECORDS["train"]["action"]["candidates"].append(candidates_tensor)
        TRAIN_RECORDS["train"]["action"]["preferred"].append(preferred_action.cpu().numpy())
        TRAIN_RECORDS["train"]["action"]["rejected"].append(rejected_action.cpu().numpy())
        TRAIN_RECORDS["train"]["action"]["highest_prob"].append(highest_prob_action.cpu().numpy())
        # 记录相似度
        TRAIN_RECORDS["train"]["similarity"]["img"].append(sim_total[:, :].cpu().numpy())  # 兼容原记录逻辑
        TRAIN_RECORDS["train"]["similarity"]["driver"].append(sim_total[:, :].cpu().numpy())
        TRAIN_RECORDS["train"]["similarity"]["total"].append(sim_total.cpu().numpy())
        # 记录对数概率
        TRAIN_RECORDS["train"]["logp"]["policy_chosen"].append(policy_chosen_logp.cpu().numpy())
        TRAIN_RECORDS["train"]["logp"]["policy_rejected"].append(policy_rejected_logp.cpu().numpy())
        TRAIN_RECORDS["train"]["logp"]["ref_chosen"].append(ref_chosen_logp.cpu().numpy())
        TRAIN_RECORDS["train"]["logp"]["ref_rejected"].append(ref_rejected_logp.cpu().numpy())
        # 记录元信息
        TRAIN_RECORDS["train"]["meta"]["epoch"].append(epoch)
        TRAIN_RECORDS["train"]["meta"]["batch_idx"].append(batch_idx)
        TRAIN_RECORDS["train"]["meta"]["timestamp"].append(time.time())

        # 更新累计损失
        total_loss += loss.item()
        batch_count += 1
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Avg Loss": f"{total_loss / batch_count:.4f}",
            "Optimized Samples": optimized_samples
        })

    # 记录epoch级损失
    avg_epoch_loss = total_loss / max(batch_count, 1)
    TRAIN_RECORDS["train"]["loss"]["epoch"].append(avg_epoch_loss)
    print(f"✅ Train Epoch {epoch + 1} | Avg Loss: {avg_epoch_loss:.4f} | Optimized Samples: {optimized_samples}")
    return avg_epoch_loss, optimized_samples


def validate_full(epoch, val_loader, models):
    """保留原验证逻辑，完整记录所有验证数据"""
    image_embed, motor_embed, policy_gen, ref_gen, img_sim_model, driver_sim_model = models
    policy_gen.eval()

    total_val_loss = 0.0
    batch_count = 0
    pbar = tqdm(
        enumerate(val_loader),
        desc=f"Val Epoch {epoch + 1}/{CONFIG['epochs']}",
        total=min(CONFIG["max_val_batches"], len(val_loader))
    )

    with torch.no_grad():
        for batch_idx, batch in pbar:
            if batch_count >= CONFIG["max_val_batches"]:
                break

            # 解包数据（原逻辑）
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
            img_proj_future = img_sim_model(future_image_embedded)  # 可能返回元组

            # 2. 生成候选动作
            policy_outputs = policy_gen(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=CONFIG["num_candidates"]
            )
            policy_candidates = policy_outputs['candidates']

            # 3. 选择偏好/非偏好动作
            preferred_action, rejected_action, sim_total = select_preferred_rejected(
                candidates=policy_candidates,
                img_proj_future=img_proj_future,
                future_driver_last=future_driver_last,
                motor_embed=motor_embed,
                driver_sim_model=driver_sim_model,
                batch_size=batch_size
            )

            # 4. 计算对数概率
            policy_mean, policy_std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)
            policy_chosen_logp = gaussian_log_prob(policy_mean, policy_std, preferred_action)
            policy_rejected_logp = gaussian_log_prob(policy_mean, policy_std, rejected_action)

            ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)
            ref_chosen_logp = gaussian_log_prob(ref_mean, ref_std, preferred_action)
            ref_rejected_logp = gaussian_log_prob(ref_mean, ref_std, rejected_action)

            # 5. 计算损失
            loss = dpo_loss(
                policy_chosen_logps=policy_chosen_logp,
                policy_rejected_logps=policy_rejected_logp,
                ref_chosen_logps=ref_chosen_logp,
                ref_rejected_logps=ref_rejected_logp
            )

            # ---------------------- 记录所有验证数据（原DPOTraining完整记录） ----------------------
            TRAIN_RECORDS["val"]["loss"]["batch"].append(loss.item())
            candidates_tensor = torch.stack(policy_candidates).permute(1, 0, 2).cpu().numpy()
            TRAIN_RECORDS["val"]["action"]["candidates"].append(candidates_tensor)
            TRAIN_RECORDS["val"]["action"]["preferred"].append(preferred_action.cpu().numpy())
            TRAIN_RECORDS["val"]["action"]["rejected"].append(rejected_action.cpu().numpy())
            # 计算最高概率动作用于记录
            highest_prob_action, _ = get_highest_prob_action(
                candidates=policy_candidates,
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                policy_gen=policy_gen
            )
            TRAIN_RECORDS["val"]["action"]["highest_prob"].append(highest_prob_action.cpu().numpy())
            # 记录相似度
            TRAIN_RECORDS["val"]["similarity"]["img"].append(sim_total[:, :].cpu().numpy())
            TRAIN_RECORDS["val"]["similarity"]["driver"].append(sim_total[:, :].cpu().numpy())
            TRAIN_RECORDS["val"]["similarity"]["total"].append(sim_total.cpu().numpy())
            # 记录对数概率
            TRAIN_RECORDS["val"]["logp"]["policy_chosen"].append(policy_chosen_logp.cpu().numpy())
            TRAIN_RECORDS["val"]["logp"]["policy_rejected"].append(policy_rejected_logp.cpu().numpy())
            TRAIN_RECORDS["val"]["logp"]["ref_chosen"].append(ref_chosen_logp.cpu().numpy())
            TRAIN_RECORDS["val"]["logp"]["ref_rejected"].append(ref_rejected_logp.cpu().numpy())
            # 记录元信息
            TRAIN_RECORDS["val"]["meta"]["epoch"].append(epoch)
            TRAIN_RECORDS["val"]["meta"]["batch_idx"].append(batch_idx)
            TRAIN_RECORDS["val"]["meta"]["timestamp"].append(time.time())

            # 更新累计损失
            total_val_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({
                "Val Loss": f"{loss.item():.4f}",
                "Avg Val Loss": f"{total_val_loss / batch_count:.4f}"
            })

    # 记录epoch级验证损失
    avg_epoch_val_loss = total_val_loss / max(batch_count, 1)
    TRAIN_RECORDS["val"]["loss"]["epoch"].append(avg_epoch_val_loss)
    print(f"✅ Val Epoch {epoch + 1} | Avg Loss: {avg_epoch_val_loss:.4f}")
    return avg_epoch_val_loss


# ---------------------- 5. 保存逻辑（保留原DPOTraining所有保存逻辑） ----------------------
def save_all_records():
    """保存所有训练轨迹记录（原逻辑完整保留）"""
    # 保存损失记录
    np.save(CONFIG["loss_records_path"], {
        "train_batch": TRAIN_RECORDS["train"]["loss"]["batch"],
        "train_epoch": TRAIN_RECORDS["train"]["loss"]["epoch"],
        "val_batch": TRAIN_RECORDS["val"]["loss"]["batch"],
        "val_epoch": TRAIN_RECORDS["val"]["loss"]["epoch"]
    })
    print(f"✅ 损失记录已保存：{CONFIG['loss_records_path']}")

    # 保存相似度记录
    np.save(CONFIG["similarity_records_path"], {
        "train_img": TRAIN_RECORDS["train"]["similarity"]["img"],
        "train_driver": TRAIN_RECORDS["train"]["similarity"]["driver"],
        "train_total": TRAIN_RECORDS["train"]["similarity"]["total"],
        "val_img": TRAIN_RECORDS["val"]["similarity"]["img"],
        "val_driver": TRAIN_RECORDS["val"]["similarity"]["driver"],
        "val_total": TRAIN_RECORDS["val"]["similarity"]["total"]
    })
    print(f"✅ 相似度记录已保存：{CONFIG['similarity_records_path']}")

    # 保存对数概率记录
    np.save(CONFIG["logp_records_path"], {
        "train_policy_chosen": TRAIN_RECORDS["train"]["logp"]["policy_chosen"],
        "train_policy_rejected": TRAIN_RECORDS["train"]["logp"]["policy_rejected"],
        "train_ref_chosen": TRAIN_RECORDS["train"]["logp"]["ref_chosen"],
        "train_ref_rejected": TRAIN_RECORDS["train"]["logp"]["ref_rejected"],
        "val_policy_chosen": TRAIN_RECORDS["val"]["logp"]["policy_chosen"],
        "val_policy_rejected": TRAIN_RECORDS["val"]["logp"]["policy_rejected"],
        "val_ref_chosen": TRAIN_RECORDS["val"]["logp"]["ref_chosen"],
        "val_ref_rejected": TRAIN_RECORDS["val"]["logp"]["ref_rejected"]
    })
    print(f"✅ 对数概率记录已保存：{CONFIG['logp_records_path']}")

    # 保存动作记录（转列表方便JSON保存）
    action_records = {
        "train": {
            "candidates": [arr.tolist() for arr in TRAIN_RECORDS["train"]["action"]["candidates"]],
            "preferred": [arr.tolist() for arr in TRAIN_RECORDS["train"]["action"]["preferred"]],
            "rejected": [arr.tolist() for arr in TRAIN_RECORDS["train"]["action"]["rejected"]],
            "highest_prob": [arr.tolist() for arr in TRAIN_RECORDS["train"]["action"]["highest_prob"]]
        },
        "val": {
            "candidates": [arr.tolist() for arr in TRAIN_RECORDS["val"]["action"]["candidates"]],
            "preferred": [arr.tolist() for arr in TRAIN_RECORDS["val"]["action"]["preferred"]],
            "rejected": [arr.tolist() for arr in TRAIN_RECORDS["val"]["action"]["rejected"]],
            "highest_prob": [arr.tolist() for arr in TRAIN_RECORDS["val"]["action"]["highest_prob"]]
        }
    }
    with open(CONFIG["action_records_path"], "w") as f:
        json.dump(action_records, f, indent=4)
    print(f"✅ 动作记录已保存：{CONFIG['action_records_path']}")

    # 保存元信息
    meta_records = {
        "train": {
            "epoch": TRAIN_RECORDS["train"]["meta"]["epoch"],
            "batch_idx": TRAIN_RECORDS["train"]["meta"]["batch_idx"],
            "timestamp": TRAIN_RECORDS["train"]["meta"]["timestamp"]
        },
        "val": {
            "epoch": TRAIN_RECORDS["val"]["meta"]["epoch"],
            "batch_idx": TRAIN_RECORDS["val"]["meta"]["batch_idx"],
            "timestamp": TRAIN_RECORDS["val"]["meta"]["timestamp"]
        },
        "config": CONFIG  # 保存配置参数
    }
    with open(CONFIG["meta_records_path"], "w") as f:
        json.dump(meta_records, f, indent=4)
    print(f"✅ 元信息记录已保存：{CONFIG['meta_records_path']}")


def save_dpo_model(models, optimizer, epoch):
    """保存DPO模型（保留原格式，仅适配新模型）"""
    image_embed, motor_embed, policy_gen, ref_gen, img_sim_model, driver_sim_model = models
    save_dict = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": TRAIN_RECORDS["train"]["loss"]["epoch"],
        "val_losses": TRAIN_RECORDS["val"]["loss"]["epoch"],
        # 兼容PreTraining的保存格式（原逻辑）
        "image_embed": image_embed.state_dict(),
        "motor_embed": motor_embed.state_dict(),
        "candidate_generator": policy_gen.state_dict(),
        "img_sim_model": img_sim_model.state_dict(),
        "driver_sim_model": driver_sim_model.state_dict()
    }
    torch.save(save_dict, CONFIG["dpo_model_save_path"])
    print(f"✅ DPO模型已保存：{CONFIG['dpo_model_save_path']}")


# ---------------------- 6. 主流程（保留原DPOTraining所有逻辑） ----------------------
def main():
    print("=" * 50)
    print("🚀 启动DPOTraining（保留原逻辑+适配新模型+修复元组bug）")
    print("=" * 50)

    # 1. 加载预训练模型（适配新Models.py）
    models = load_pretrained_models()
    image_embed, motor_embed, policy_gen, ref_gen, img_sim_model, driver_sim_model = models

    # 2. 加载数据集（保留原逻辑）
    train_loader, val_loader = load_dataset()

    # 3. 初始化优化器（原逻辑）
    optimizer = torch.optim.AdamW(
        params=policy_gen.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    print("✅ 优化器初始化完成")

    # 4. 训练循环（保留原逻辑）
    best_val_loss = float('inf')
    for epoch in range(CONFIG["epochs"]):
        print(f"\n📌 Epoch {epoch + 1}/{CONFIG['epochs']}")
        # 训练
        train_loss, optimized_samples = train_one_epoch(epoch, train_loader, models, optimizer)
        # 验证
        val_loss = validate_full(epoch, val_loader, models)
        # 保存最优模型（原逻辑）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dpo_model(models, optimizer, epoch)
            print(f"🏆 保存最优模型 | 最佳验证损失：{best_val_loss:.4f}")

    # 5. 保存所有记录（原逻辑完整保留）
    save_all_records()
    # 保存最终模型
    save_dpo_model(models, optimizer, CONFIG["epochs"])

    print("\n" + "=" * 50)
    print("🎉 DPOTraining完成 | 所有记录已保存 | Bug修复完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
