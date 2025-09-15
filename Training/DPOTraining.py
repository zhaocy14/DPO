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

# ---------------------- 核心配置参数（新增alpha系数） ----------------------
CONFIG = {
    # 新增：相似度加权系数（0.9图像 + 0.1动作）
    "alpha": 0.9,

    "batch_size": 1,
    "epochs": 5,
    "lr": 5e-7,
    "num_candidates": 5,
    "sampling_workers": 2,
    "max_train_samples_per_epoch": 500,
    "val_batch_size": 8,
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
    def check_batch_size(dataloader, name):
        for batch in dataloader:
            imgs1 = batch[0]
            if imgs1.shape[0] != 1:
                raise RuntimeError(f"[BatchSize错误] {name}的batch_size={imgs1.shape[0]}，需为1")
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

    check_batch_size(train_loader, "训练集")
    check_batch_size(val_loader, "验证集")
    print("[数据配置] 已确认所有DataLoader的batch_size=1")

    return train_loader, val_loader


# ---------------------- 3. 核心工具函数（修改相似度计算） ----------------------
def gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    std = std + eps
    log_prob = -0.5 * torch.log(2 * torch.tensor(np.pi, device=device)) - torch.log(std) - (action - mean) ** 2 / (
                2 * std ** 2)
    return log_prob.sum(dim=-1)


def get_generator_distribution(generator: EncoderOnlyCandidateGenerator,
                               image_embedded: torch.Tensor,
                               motor_embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert image_embedded.shape[0] == 1, f"image_embedded batch_size={image_embedded.shape[0]}，需为1"
    assert motor_embedded.shape[0] == 1, f"motor_embedded batch_size={motor_embedded.shape[0]}，需为1"

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
                              motor_embed: MotorEmbedding) -> tuple[torch.Tensor, torch.Tensor]:
    """
    核心修改：总相似度 = alpha*sim_img + (1-alpha)*sim_driver
    """
    # 1. 维度检查
    assert len(candidates) == CONFIG["num_candidates"], f"候选数={len(candidates)}，需为{CONFIG['num_candidates']}"
    for i, cand in enumerate(candidates):
        assert cand.shape == (1, CONFIG["motor_dim"]), f"候选{i}维度错误：{cand.shape}，需为(1,{CONFIG['motor_dim']})"
    assert future_driver_last.shape == (1,
                                        CONFIG["motor_dim"]), f"future_driver_last维度错误：{future_driver_last.shape}"

    # 2. 候选动作嵌入
    candidate_embeddings = []
    for cand in candidates:
        cand_with_seq = cand.unsqueeze(1)  # (1,1,2)
        emb = motor_embed(cand_with_seq)  # (1,1,128)
        candidate_embeddings.append(emb)

    # 3. 计算图像相似度
    sim_img = []
    for emb in candidate_embeddings:
        cand_proj = driver_sim_model(emb).squeeze(1)  # (1,32)
        img_proj_future_squeezed = img_proj_future.squeeze(1)  # (1,32)
        sim = F.cosine_similarity(cand_proj, img_proj_future_squeezed, dim=1)  # (1,)
        sim_img.append(sim[0].item())  # 标量值

    # 4. 计算动作相似度
    future_driver_with_seq = future_driver_last.unsqueeze(1)  # (1,1,2)
    future_driver_emb = motor_embed(future_driver_with_seq)  # (1,1,128)
    future_emb_squeezed = future_driver_emb.squeeze(1)  # (1,128)
    future_norm = F.normalize(future_emb_squeezed, dim=1)  # (1,128)

    sim_driver = []
    for emb in candidate_embeddings:
        emb_squeezed = emb.squeeze(1)  # (1,128)
        emb_norm = F.normalize(emb_squeezed, dim=1)  # (1,128)
        sim = F.cosine_similarity(emb_norm, future_norm, dim=1)  # (1,)
        sim_driver.append(sim[0].item())  # 标量值

    # 5. 核心修改：带alpha系数的加权求和（0.9*图像 + 0.1*动作）
    alpha = CONFIG["alpha"]
    sim_total = alpha * torch.tensor(sim_img, device=device) + (1 - alpha) * torch.tensor(sim_driver, device=device)
    preferred_idx = sim_total.argmax().item()
    rejected_idx = sim_total.argmin().item()

    # 6. 提取动作
    preferred = candidates[preferred_idx].squeeze(0)  # (2,)
    rejected = candidates[rejected_idx].squeeze(0)  # (2,)
    return preferred, rejected


def is_action_repeated(current_action: torch.Tensor, history_actions: list[torch.Tensor]) -> bool:
    if not history_actions:
        return False
    assert current_action.shape == (CONFIG["motor_dim"],), f"current_action维度错误：{current_action.shape}"
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
    assert preferred.shape == (CONFIG["motor_dim"],), f"preferred维度错误：{preferred.shape}"
    assert rejected.shape == (CONFIG["motor_dim"],), f"rejected维度错误：{rejected.shape}"

    policy_mean, policy_std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)
    log_p_theta_pref = gaussian_log_prob(policy_mean, policy_std, preferred.unsqueeze(0))  # (1,)
    log_p_theta_rej = gaussian_log_prob(policy_mean, policy_std, rejected.unsqueeze(0))  # (1,)

    with torch.no_grad():
        ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)
        log_p_ref_pref = gaussian_log_prob(ref_mean, ref_std, preferred.unsqueeze(0))
        log_p_ref_rej = gaussian_log_prob(ref_mean, ref_std, rejected.unsqueeze(0))

    advantage = (log_p_theta_pref - log_p_ref_pref) - (log_p_theta_rej - log_p_ref_rej)  # (1,)
    return -F.logsigmoid(CONFIG["dpo_beta"] * advantage).mean()


# ---------------------- 4. 训练/验证函数 ----------------------
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

    pbar = tqdm(enumerate(train_loader),
                desc=f"[训练] Epoch {epoch + 1}/{CONFIG['epochs']}",
                total=min(CONFIG["max_train_samples_per_epoch"], len(train_loader)))

    for sample_idx, batch in pbar:
        if sample_idx >= CONFIG["max_train_samples_per_epoch"]:
            print(f"\n[训练] 已达样本上限（{CONFIG['max_train_samples_per_epoch']}），终止")
            break

        # 解包数据
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        assert imgs1.shape[0] == 1, f"imgs1 batch_size错误：{imgs1.shape[0]}"
        assert driver.shape[0] == 1, f"driver batch_size错误：{driver.shape[0]}"

        # 数据预处理
        images = torch.stack([imgs1, imgs2], dim=2).to(device)  # (1,30,2,3,H,W)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        driver = driver.to(device)  # 保留原始数据
        future_driver = future_driver.to(device)
        future_driver_last = future_driver[:, -1, :]  # (1,2)

        # 特征嵌入
        with torch.no_grad():
            image_embedded = image_embed(images)  # (1,30,256)
            motor_embedded = motor_embed(driver)  # (1,30,128)
            future_image_embedded = image_embed(future_images)  # (1,30,256)
            img_proj_future = img_sim_model(future_image_embedded)  # 可能为(1,1,32)

        # 生成动作候选
        generator_output = policy_gen(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=CONFIG["num_candidates"],
            temperature=1.0
        )
        candidates = generator_output[CONFIG["use_candidates"]]  # 列表：(1,1,2)×5
        candidates = [cand.squeeze(1) for cand in candidates]  # 每个：(1,2)

        # 选择偏好/非偏好动作
        preferred, rejected = select_preferred_rejected(
            candidates=candidates,
            img_proj_future=img_proj_future,
            future_driver_last=future_driver_last,
            motor_embed=motor_embed
        )

        # 重复动作跳过
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
        if len(history_actions) > CONFIG["history_cache_size"]:
            history_actions.pop(0)

        # 累计损失
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
                  motor_embed: MotorEmbedding) -> tuple[float, int]:
    policy_gen.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader),
                    desc=f"[验证] Epoch {epoch + 1}",
                    total=len(val_loader))

        for sample_idx, batch in pbar:
            # 解包数据
            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
            assert imgs1.shape[0] == 1, f"验证集imgs1 batch_size错误：{imgs1.shape[0]}"

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

            # 生成候选
            generator_output = policy_gen(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=CONFIG["num_candidates"],
                temperature=1.0
            )
            candidates = generator_output[CONFIG["use_candidates"]]
            candidates = [cand.squeeze(1) for cand in candidates]

            # 选择动作
            preferred, rejected = select_preferred_rejected(
                candidates=candidates,
                img_proj_future=img_proj_future,
                future_driver_last=future_driver_last,
                motor_embed=motor_embed
            )

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
            sample_count += 1

            pbar.set_postfix({
                "验证损失": f"{loss.item():.4f}",
                "已验证样本": sample_count
            })

    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    print(f"[验证] Epoch {epoch + 1} | 平均损失：{avg_loss:.4f} | 总样本数：{sample_count}")
    return avg_loss, sample_count


# ---------------------- 5. 主函数 ----------------------
def main():
    start_total_time = time.time()
    print("\n" + "=" * 60)
    print("                      EncoderOnlyCandidateGenerator DPO优化（带alpha系数）")
    print("=" * 60)
    print(f"[配置信息] 相似度加权系数：alpha={CONFIG['alpha']}（图像占比），1-alpha={1 - CONFIG['alpha']}（动作占比）")

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

    # 损失记录
    loss_records = {
        "train_loss": [],
        "val_loss": [],
        "train_optimized_samples": [],
        "val_total_samples": [],
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
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 验证
        val_loss, val_samples = validate_full(
            epoch=epoch,
            val_loader=val_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            motor_embed=motor_embed
        )

        # 记录
        loss_records["train_loss"].append(train_loss)
        loss_records["val_loss"].append(val_loss)
        loss_records["train_optimized_samples"].append(optimized_samples)
        loss_records["val_total_samples"].append(val_samples)
        loss_records["lr"].append(current_lr)

        # 打印总结
        epoch_time = time.time() - epoch_start_time
        print(f"\n[Epoch 总结] Epoch {epoch + 1}/{CONFIG['epochs']}")
        print(f"  - 耗时：{epoch_time:.2f}秒 | 学习率：{current_lr:.7f}")
        print(f"  - 训练：{train_loss:.4f}（{optimized_samples}样本）")
        print(f"  - 验证：{val_loss:.4f}（{val_samples}样本）")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "policy_generator_state_dict": policy_generator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": CONFIG,  # 保存配置（含alpha）
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
    print(f"模型路径：{CONFIG['dpo_save_path']} | 损失记录：{CONFIG['dpo_loss_path']}")
    print(f"使用的相似度系数：alpha={CONFIG['alpha']}")
    print("=" * 60)


if __name__ == "__main__":
    main()