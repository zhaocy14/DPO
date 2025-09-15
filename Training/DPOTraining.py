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

# ---------------------- 1. 配置参数 ----------------------
config = {
    # 训练参数
    "batch_size": 1,
    "epochs": 5,
    "lr": 5e-7,
    "num_candidates": 5,
    "sampling_workers": 2,
    "max_train_samples_per_epoch": 500,

    # 验证参数
    "val_batch_size": 1,

    # DPO参数
    "dpo_beta": 0.1,
    "repeat_threshold": 0.95,
    "history_cache_size": 10,
    "use_candidates": "candidates1",

    # 模型参数（关键：motor_dim=2，embed_dim_gen=128）
    "embed_dim_gen": 128,  # driver_sim_model期望的输入维度
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,  # 原始动作维度
    "gen_seq_len": 30,
    "sim_seq_len": 30,
    "embed_dim_sim": 128,
    "num_layers_sim": 3,
    "nhead_sim": 4,
    "similarity_dim": 32,

    # 路径
    "data_root_dirs": '/data/cyzhao/collector_cydpo/dpo_data',
    "pretrained_model_path": "./saved_models/best_model",
    "dpo_save_path": "./saved_models/dpo_fixed_dim_model",
    "dpo_loss_path": "./loss_records/dpo_fixed_dim_loss.npy"
}

os.makedirs(os.path.dirname(config["dpo_save_path"]), exist_ok=True)
os.makedirs(os.path.dirname(config["dpo_loss_path"]), exist_ok=True)


# ---------------------- 2. 模型加载 ----------------------
def load_models_with_reference(pretrained_path):
    image_embed = ImageEmbedding(embed_dim=config["embed_dim_gen"], num_layers=3, is_resnet=False).to(device)
    motor_embed = MotorEmbedding(motor_dim=config["motor_dim"], embed_dim=config["embed_dim_gen"]).to(
        device)  # 将2维动作转为128维嵌入
    policy_generator = EncoderOnlyCandidateGenerator(
        embed_dim=config["embed_dim_gen"],
        nhead=config["nhead_gen"],
        num_layers=config["num_layers_gen"],
        motor_dim=config["motor_dim"],
        max_seq_length=config["gen_seq_len"]
    ).to(device)
    ref_generator = EncoderOnlyCandidateGenerator(
        embed_dim=config["embed_dim_gen"],
        nhead=config["nhead_gen"],
        num_layers=config["num_layers_gen"],
        motor_dim=config["motor_dim"],
        max_seq_length=config["gen_seq_len"]
    ).to(device)
    img_sim_model = SimilarityModelImage(
        embed_dim=config['embed_dim_sim'],
        num_frames=config['sim_seq_len'],
        num_layers=config['num_layers_sim'],
        nhead=config['nhead_sim'],
        similarity_dim=config['similarity_dim']
    ).to(device)
    driver_sim_model = SimilarityModelDriver(
        embed_dim=config['embed_dim_sim'],
        similarity_dim=config['similarity_dim']
    ).to(device)

    # 加载预训练权重
    checkpoint = torch.load(pretrained_path, map_location=device)
    image_embed.load_state_dict(checkpoint["model_states"]["image_embed"])
    motor_embed.load_state_dict(checkpoint["model_states"]["motor_embed"])
    policy_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])
    ref_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])
    img_sim_model.load_state_dict(checkpoint["model_states"]["img_sim_model"])
    driver_sim_model.load_state_dict(checkpoint["model_states"]["driver_sim_model"])

    # 冻结非策略模型
    for model in [image_embed, motor_embed, ref_generator, img_sim_model, driver_sim_model]:
        for param in model.parameters():
            param.requires_grad = False

    return image_embed, motor_embed, policy_generator, ref_generator, img_sim_model, driver_sim_model


# ---------------------- 3. 数据加载 ----------------------
def load_data():
    data_root = config["data_root_dirs"]
    data_dir_list = [os.path.join(data_root, f) for f in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, f)) and "2025" in f]

    all_dataset = CombinedDataset(
        dir_list=data_dir_list,
        frame_len=config["gen_seq_len"],
        predict_len=config['sim_seq_len'],
        show=True
    )
    train_dataset = all_dataset.training_dataset
    val_dataset = all_dataset.val_dataset
    print(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["sampling_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=config["sampling_workers"],
        pin_memory=True
    )
    return train_loader, val_loader


# ---------------------- 4. 核心工具函数 ----------------------
def gaussian_log_prob(mean, std, action):
    eps = 1e-6
    std = std + eps
    log_prob = -0.5 * torch.log(2 * torch.tensor(np.pi, device=device)) - torch.log(std) - (action - mean) ** 2 / (
                2 * std ** 2)
    return log_prob.sum(dim=-1)


def get_generator_distribution(generator, image_embedded, motor_embedded, use_candidates):
    combined = torch.cat([motor_embedded, image_embedded], dim=-1)
    combined = generator.positional_encoding(combined)
    encoder_out = generator.encoder(combined)
    encoder_out = encoder_out[-1]
    global_feat = encoder_out.mean(dim=1)

    if use_candidates == "candidates1":
        mean = generator.fc_mean1(global_feat)
        logvar = generator.fc_logvar1(global_feat)
    else:
        mean = generator.fc_mean2(global_feat)
        logvar = generator.fc_logvar2(global_feat)

    logvar = torch.clamp(logvar, min=-5, max=5)
    std = torch.exp(0.5 * logvar)
    return mean, std


def select_preferred_rejected(candidates, img_proj_future, future_driver_last, motor_embed):
    """修复：通过motor_embed将动作转换为嵌入维度（2→128）"""
    # 1. 候选动作先经过motor_embed转换维度（关键修复）
    # 原始候选动作：(batch, motor_dim=2) → 嵌入后：(batch, seq_len=1, embed_dim=128)
    candidate_embeddings = [motor_embed(cand.unsqueeze(1)) for cand in candidates]
    # 2. 再输入到driver_sim_model（此时维度匹配）
    candidate_projs = [driver_sim_model(emb) for emb in candidate_embeddings]

    # 3. 未来动作也需要同样处理（保持维度一致）
    future_driver_emb = motor_embed(future_driver_last.unsqueeze(1))  # (batch, 1, 128)
    future_norm = F.normalize(future_driver_emb.squeeze(1), dim=1)  # 移除时间维度

    # 4. 计算动作相似度（使用嵌入后的数据）
    sim_driver = []
    for emb in candidate_embeddings:
        emb_norm = F.normalize(emb.squeeze(1), dim=1)  # (batch, 128)
        sim = F.cosine_similarity(emb_norm, future_norm, dim=1)  # (batch,)
        sim_driver.append(sim)

    # 5. 图像相似度计算（保持不变）
    sim_img = [F.cosine_similarity(cand_proj, img_proj_future, dim=1) for cand_proj in candidate_projs]

    # 6. 综合排序
    sim_total = torch.tensor(sim_img, device=device) + torch.tensor(sim_driver, device=device)
    preferred_idx = sim_total.argmax().item()
    rejected_idx = sim_total.argmin().item()
    return candidates[preferred_idx].squeeze(0), candidates[rejected_idx].squeeze(0)


def is_action_repeated(current_action, history_actions):
    if not history_actions:
        return False
    current_norm = F.normalize(current_action.unsqueeze(0), dim=1)
    for hist_action in history_actions:
        hist_norm = F.normalize(hist_action.unsqueeze(0), dim=1)
        sim = F.cosine_similarity(current_norm, hist_norm, dim=1).item()
        if sim > config["repeat_threshold"]:
            return True
    return False


def standard_dpo_loss(policy_gen, ref_gen, image_embedded, motor_embedded, preferred, rejected):
    policy_mean, policy_std = get_generator_distribution(
        policy_gen, image_embedded, motor_embedded, config["use_candidates"]
    )
    log_p_theta_pref = gaussian_log_prob(policy_mean, policy_std, preferred.unsqueeze(0))
    log_p_theta_rej = gaussian_log_prob(policy_mean, policy_std, rejected.unsqueeze(0))

    with torch.no_grad():
        ref_mean, ref_std = get_generator_distribution(
            ref_gen, image_embedded, motor_embedded, config["use_candidates"]
        )
        log_p_ref_pref = gaussian_log_prob(ref_mean, ref_std, preferred.unsqueeze(0))
        log_p_ref_rej = gaussian_log_prob(ref_mean, ref_std, rejected.unsqueeze(0))

    advantage = (log_p_theta_pref - log_p_ref_pref) - (log_p_theta_rej - log_p_ref_rej)
    return -F.logsigmoid(config["dpo_beta"] * advantage).mean()


# ---------------------- 5. 训练/验证函数 ----------------------
def train_online_dpo(epoch, train_loader, policy_gen, ref_gen, optimizer, motor_embed):
    policy_gen.train()
    total_loss = 0.0
    optimized_count = 0
    history_actions = []

    pbar = tqdm(enumerate(train_loader), desc=f"训练 Epoch {epoch + 1}",
                total=min(config["max_train_samples_per_epoch"], len(train_loader)))

    for sample_idx, batch in pbar:
        if sample_idx >= config["max_train_samples_per_epoch"]:
            break

        # 解包数据
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        images = torch.stack([imgs1, imgs2], dim=2).to(device)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        driver = driver.to(device)
        future_driver = future_driver.to(device)
        future_driver_last = future_driver[:, -1, :]  # 原始动作：(batch, motor_dim=2)

        # 特征嵌入
        with torch.no_grad():
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)
            future_image_embedded = image_embed(future_images)
            img_proj_future = img_sim_model(future_image_embedded)

        # 生成动作候选
        generator_output = policy_generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=config["num_candidates"],
            temperature=1.0
        )
        candidates = generator_output[config["use_candidates"]]
        candidates = [cand.squeeze(1) for cand in candidates]  # 每个候选：(batch, motor_dim=2)

        # 选择偏好/非偏好动作（传入motor_embed用于维度转换）
        preferred, rejected = select_preferred_rejected(
            candidates=candidates,
            img_proj_future=img_proj_future,
            future_driver_last=future_driver_last,
            motor_embed=motor_embed  # 关键：传入嵌入模型
        )

        # 重复动作跳过优化
        if is_action_repeated(preferred, history_actions):
            pbar.set_postfix({"状态": "跳过重复动作", "优化样本数": optimized_count})
            continue

        # 计算损失并优化
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

        # 更新历史缓存
        history_actions.append(preferred.detach())
        if len(history_actions) > config["history_cache_size"]:
            history_actions.pop(0)

        # 累计损失
        total_loss += loss.item()
        optimized_count += 1
        pbar.set_postfix({
            "DPO损失": f"{loss.item():.4f}",
            "优化样本数": optimized_count
        })

    avg_loss = total_loss / optimized_count if optimized_count > 0 else 0.0
    return avg_loss, optimized_count


def validate_full_dpo(val_loader, policy_gen, ref_gen, motor_embed):
    policy_gen.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), desc="全量验证", total=len(val_loader))

        for sample_idx, batch in pbar:
            # 解包数据
            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
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

            # 生成动作候选
            generator_output = policy_generator(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=config["num_candidates"],
                temperature=1.0
            )
            candidates = generator_output[config["use_candidates"]]
            candidates = [cand.squeeze(1) for cand in candidates]

            # 选择偏好/非偏好动作
            preferred, rejected = select_preferred_rejected(
                candidates=candidates,
                img_proj_future=img_proj_future,
                future_driver_last=future_driver_last,
                motor_embed=motor_embed
            )

            # 计算验证损失
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
            pbar.set_postfix({
                "验证损失": f"{loss.item():.4f}",
                "已验证样本": sample_count
            })

    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    print(f"全量验证完成 | 总样本数：{sample_count} | 平均损失：{avg_loss:.4f}")
    return avg_loss, sample_count


# ---------------------- 6. Main函数 ----------------------
def main():
    # 加载模型和数据
    global image_embed, motor_embed, img_sim_model, driver_sim_model, policy_generator
    image_embed, motor_embed, policy_generator, ref_generator, img_sim_model, driver_sim_model = load_models_with_reference(
        config["pretrained_model_path"]
    )
    train_loader, val_loader = load_data()

    # 优化器
    optimizer = torch.optim.Adam(
        params=policy_generator.parameters(),
        lr=config["lr"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # 损失记录
    loss_records = {
        "train_loss": [],
        "val_loss": [],
        "train_optimized_samples": [],
        "val_total_samples": []
    }
    best_val_loss = float("inf")

    print("\n" + "=" * 50)
    print(f"开始训练 | 动作维度：{config['motor_dim']} → 嵌入维度：{config['embed_dim_gen']}")
    print("=" * 50)

    # 训练循环
    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        # 逐帧训练（传入motor_embed用于维度转换）
        train_loss, optimized_samples = train_online_dpo(
            epoch=epoch,
            train_loader=train_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            optimizer=optimizer,
            motor_embed=motor_embed  # 关键：传入嵌入模型
        )
        scheduler.step()

        # 全量验证
        val_loss, val_total_samples = validate_full_dpo(
            val_loader=val_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            motor_embed=motor_embed
        )

        # 记录损失
        loss_records["train_loss"].append(train_loss)
        loss_records["val_loss"].append(val_loss)
        loss_records["train_optimized_samples"].append(optimized_samples)
        loss_records["val_total_samples"].append(val_total_samples)

        # 打印日志
        epoch_time = time.time() - epoch_start
        print("\n" + "=" * 30)
        print(f"Epoch {epoch + 1}/{config['epochs']} | 耗时：{epoch_time:.2f}秒")
        print(f"训练：平均损失={train_loss:.4f}（优化样本数={optimized_samples}）")
        print(f"验证：平均损失={val_loss:.4f}（总样本数={val_total_samples}）")
        print("=" * 30 + "\n")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "policy_generator": policy_generator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config
            }, config["dpo_save_path"])
            print(f"✅ 保存最佳模型至：{config['dpo_save_path']}\n")

    # 保存损失记录
    np.save(config["dpo_loss_path"], loss_records)
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳全量验证损失：{best_val_loss:.4f}")
    print(f"模型路径：{config['dpo_save_path']}")
    print(f"损失记录：{config['dpo_loss_path']}")
    print("=" * 50)


if __name__ == "__main__":
    main()

