import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 路径配置
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

# 导入模型和数据集类
from DataModule.DataModule import CombinedDataset
from Model.Models import (
    ImageEmbedding,
    MotorEmbedding,
    EncoderOnlyCandidateGenerator,
    SimilarityModelImage,
    SimilarityModelDriver,
    calculate_model_size
)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置参数
config = {
    # 训练参数
    "batch_size": 4,
    "epochs": 100,
    "lr": 5e-5,
    "sampling_workers": 20,
    "reverse_mse_weight": 0.1,
    "sim_loss_weight": 2.0,
    # ===================== 新增：相似度损失类型选择 =====================
    "sim_loss_type": "cos_sim",  # 可选："info_ce"（原有） / "cos_sim"（新）
    "info_ce_temperature": 0.07,  # InfoCE的温度参数（保留）
    # ===================================================================

    # 生成器模型参数
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,
    "gen_seq_len": 30,

    # 相似度模型参数
    "sim_seq_len": 30,
    "embed_dim_sim": 128,
    "num_layers_sim": 3,
    "nhead_sim": 4,
    "similarity_dim": 32,

    # 数据和模型路径
    "data_root_dirs": '/data/cyzhao/collector_cydpo',
    "save_path": "./saved_models",
    "loss_data_path": "./loss_records"
}

# 创建保存目录
os.makedirs(config["save_path"], exist_ok=True)
os.makedirs(config["loss_data_path"], exist_ok=True)

# 初始化模型（原有逻辑不变）
image_embed = ImageEmbedding(
    embed_dim=config["embed_dim_gen"],
    num_layers=3,
    is_resnet=False
).to(device)

motor_embed = MotorEmbedding(
    motor_dim=config["motor_dim"],
    embed_dim=config["embed_dim_gen"]
).to(device)

candidate_generator = EncoderOnlyCandidateGenerator(
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
    similarity_dim=config['similarity_dim'],
    motor_dim=config['motor_dim'],
    dropout_rate=0.2
).to(device)

driver_sim_model = SimilarityModelDriver(
    embed_dim=config['embed_dim_sim'],
    similarity_dim=config['similarity_dim'],
).to(device)

# 加载数据集（原有逻辑不变）
data_root = config["data_root_dirs"]
data_dir_list = [
    os.path.join(data_root, file)
    for file in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, file)) and "2025" in file
]

all_dataset = CombinedDataset(
    dir_list=data_dir_list,
    frame_len=config["gen_seq_len"],
    predict_len=config['sim_seq_len'],
    show=True
)

train_dataset = all_dataset.training_dataset
val_dataset = all_dataset.val_dataset
print(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")

# 数据加载器（原有逻辑不变）
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['sampling_workers']
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['sampling_workers']
)

# 优化器和学习率调度器（原有逻辑不变）
optimizer = torch.optim.Adam(
    params=[
        {'params': image_embed.parameters()},
        {'params': motor_embed.parameters()},
        {'params': candidate_generator.parameters()},
        {'params': img_sim_model.parameters()},
        {'params': driver_sim_model.parameters()}
    ],
    lr=config['lr']
)
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def nll_loss(mean, std, target):
    """原有：高斯分布的负对数似然损失"""
    eps = 1e-6
    std = std + eps
    nll = torch.log(std) + (target - mean) ** 2 / (2 * std ** 2)
    return nll.mean()


# ===================== 保留原有info_ce_loss完整定义 =====================
def info_ce_loss(img_proj, candidate_projections, temperature=0.1):
    """
    原有：Info Noise Contrastive Estimation Loss
    :param img_proj: 未来图像序列投影 (batch, similarity_dim)
    :param candidate_projections: 候选动作投影列表，第一个为正样本
    :param temperature: 温度参数
    :return: InfoCE损失
    """
    batch_size = img_proj.shape[0]
    candidates = torch.stack(candidate_projections, dim=1)  # (batch, num_candidates, similarity_dim)

    # 新增L2归一化（修复原有逻辑的缺陷）
    img_proj = F.normalize(img_proj, p=2, dim=-1)
    candidates = F.normalize(candidates, p=2, dim=-1)

    # 计算余弦相似度
    similarities = F.cosine_similarity(
        img_proj.unsqueeze(1),
        candidates,
        dim=2
    )

    # 温度缩放 + 交叉熵损失
    similarities = similarities / temperature
    loss = F.cross_entropy(
        similarities,
        torch.zeros(batch_size, dtype=torch.long, device=img_proj.device)
    )
    return loss


# =======================================================================


# ===================== 新增cos_sim_loss定义 =====================
def cos_sim_loss(img_proj, best_action_proj):
    """
    新：1-余弦相似度损失（仅对比最优动作）
    :param img_proj: 图像序列投影 (batch, similarity_dim)
    :param best_action_proj: 最优动作投影 (batch, similarity_dim)
    :return: 平均损失值
    """
    # L2归一化
    img_proj = F.normalize(img_proj, p=2, dim=-1)
    best_action_proj = F.normalize(best_action_proj, p=2, dim=-1)

    # 计算余弦相似度
    cos_similarity = F.cosine_similarity(img_proj, best_action_proj, dim=-1)

    # 1 - 余弦相似度作为损失
    loss = 1 - cos_similarity
    return loss.mean()


# =================================================================


def reverse_mse_loss(pred, target, weight=1.0):
    """原有：反向MSE损失"""
    mse = F.mse_loss(pred, target, reduction='mean')
    return weight * (-mse)


def train_one_epoch(epoch):
    """训练单个epoch（新增损失分支逻辑）"""
    image_embed.train()
    motor_embed.train()
    candidate_generator.train()
    img_sim_model.train()
    driver_sim_model.train()

    total_gen_loss = 0.0
    total_sim_loss = 0.0
    total_reverse_mse_loss = 0.0
    total_loss = 0.0

    max_train_batches = 10
    batch_count = 0

    pbar = tqdm(
        enumerate(train_loader),
        desc=f"训练 Epoch {epoch + 1}/{config['epochs']} (损失类型: {config['sim_loss_type']})",
        total=min(max_train_batches, len(train_loader))
    )

    for batch_idx, batch in pbar:
        if batch_count >= max_train_batches:
            break
        batch_count += 1

        # 解包数据（原有逻辑不变）
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        images = torch.stack([imgs1, imgs2], dim=2).to(device)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        driver = driver.to(device)
        next_driver = future_driver[:, 0, :].to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 1. 特征嵌入（原有逻辑不变）
        image_embedded = image_embed(images)
        motor_embedded = motor_embed(driver)

        # 2. 动作生成（原有逻辑不变，保留候选动作生成）
        num_candidates = 5
        outputs = candidate_generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=num_candidates,
            temperature=1.0
        )
        mean = outputs['mean']
        std = outputs['std']

        # 生成损失（原有逻辑不变）
        gen_loss = nll_loss(mean, std, next_driver)

        # 3. 相似度损失计算（核心分支逻辑）
        future_images_emb = image_embed(future_images)
        img_proj, img_motor_pred = img_sim_model(future_images_emb)

        # 反向MSE损失（原有逻辑不变）
        reverse_mse_loss_val = reverse_mse_loss(
            pred=img_motor_pred,
            target=next_driver,
            weight=config["reverse_mse_weight"]
        )

        # ===================== 损失类型分支 =====================
        if config["sim_loss_type"] == "info_ce":
            # 原有InfoCE损失逻辑（完整保留）
            # 均值动作作为正样本
            mean_action = mean.unsqueeze(1)
            mean_embedded = motor_embed(mean_action)
            mean_proj = driver_sim_model(mean_embedded[:, -1, :])
            candidate_projections = [mean_proj]

            # 候选动作作为负样本（原有遍历逻辑）
            for candidate in outputs['candidates']:
                candidate_embedded = motor_embed(candidate)
                candidate_proj = driver_sim_model(candidate_embedded[:, -1, :])
                candidate_projections.append(candidate_proj)

            # 计算InfoCE损失
            sim_loss = info_ce_loss(
                img_proj=img_proj,
                candidate_projections=candidate_projections,
                temperature=config["info_ce_temperature"]
            )

        elif config["sim_loss_type"] == "cos_sim":
            # 新cos_sim_loss逻辑（仅最优动作）
            # 均值动作作为最优动作
            best_action = mean.unsqueeze(1)
            best_action_emb = motor_embed(best_action)
            best_action_proj = driver_sim_model(best_action_emb[:, -1, :])

            # 计算1-余弦相似度损失
            sim_loss = cos_sim_loss(img_proj, best_action_proj)

        else:
            raise ValueError(f"无效的损失类型：{config['sim_loss_type']}，可选info_ce/cos_sim")
        # =======================================================

        # 应用相似度损失权重
        sim_loss = sim_loss * config["sim_loss_weight"]

        # 总损失（原有逻辑不变）
        loss = gen_loss + sim_loss + reverse_mse_loss_val
        loss.backward()
        optimizer.step()

        # 累计损失（原有逻辑不变）
        total_gen_loss += gen_loss.item()
        total_sim_loss += sim_loss.item()
        total_reverse_mse_loss += reverse_mse_loss_val.item()
        total_loss += loss.item()

        # 更新进度条（兼容两种损失）
        pbar.set_postfix({
            "总损失": f"{loss.item():.4f}",
            "生成损失": f"{gen_loss.item():.4f}",
            "相似度损失": f"{sim_loss.item():.4f}",
            "反向MSE损失": f"{reverse_mse_loss_val.item():.4f}"
        })

    # 计算平均损失（原有逻辑不变）
    avg_gen_loss = total_gen_loss / min(max_train_batches, len(train_loader))
    avg_sim_loss = total_sim_loss / min(max_train_batches, len(train_loader))
    avg_reverse_mse_loss = total_reverse_mse_loss / min(max_train_batches, len(train_loader))
    avg_total_loss = total_loss / min(max_train_batches, len(train_loader))

    return avg_total_loss, avg_gen_loss, avg_sim_loss, avg_reverse_mse_loss


def validate_one_epoch(epoch):
    """验证单个epoch（与训练分支逻辑完全一致）"""
    image_embed.eval()
    motor_embed.eval()
    candidate_generator.eval()
    img_sim_model.eval()
    driver_sim_model.eval()

    total_gen_loss = 0.0
    total_sim_loss = 0.0
    total_reverse_mse_loss = 0.0
    total_loss = 0.0

    max_val_batches = 100
    batch_count = 0

    pbar = tqdm(
        enumerate(val_loader),
        desc=f"验证 Epoch {epoch + 1}/{config['epochs']} (损失类型: {config['sim_loss_type']})",
        total=min(max_val_batches, len(val_loader))
    )

    with torch.no_grad():
        for batch_idx, batch in pbar:
            if batch_count >= max_val_batches:
                break
            batch_count += 1

            # 解包数据（原有逻辑不变）
            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
            images = torch.stack([imgs1, imgs2], dim=2).to(device)
            future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
            driver = driver.to(device)
            next_driver = future_driver[:, 0, :].to(device)

            # 特征嵌入（原有逻辑不变）
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)

            # 动作生成（原有逻辑不变）
            num_candidates = 5
            outputs = candidate_generator(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=num_candidates,
                temperature=1.0
            )
            mean = outputs['mean']
            std = outputs['std']
            gen_loss = nll_loss(mean, std, next_driver)

            # 相似度损失计算（与训练一致的分支逻辑）
            future_image_emb = image_embed(future_images)
            img_proj, img_motor_pred = img_sim_model(future_image_emb)

            reverse_mse_loss_val = reverse_mse_loss(
                pred=img_motor_pred,
                target=next_driver,
                weight=config["reverse_mse_weight"]
            )

            # ===================== 损失类型分支 =====================
            if config["sim_loss_type"] == "info_ce":
                # 原有InfoCE逻辑
                mean_action = mean.unsqueeze(1)
                mean_embedded = motor_embed(mean_action)
                mean_proj = driver_sim_model(mean_embedded[:, -1, :])
                candidate_projections = [mean_proj]

                for candidate in outputs['candidates']:
                    candidate_embedded = motor_embed(candidate)
                    candidate_proj = driver_sim_model(candidate_embedded[:, -1, :])
                    candidate_projections.append(candidate_proj)

                sim_loss = info_ce_loss(
                    img_proj=img_proj,
                    candidate_projections=candidate_projections,
                    temperature=config["info_ce_temperature"]
                )

            elif config["sim_loss_type"] == "cos_sim":
                # 新cos_sim逻辑
                best_action = mean.unsqueeze(1)
                best_action_emb = motor_embed(best_action)
                best_action_proj = driver_sim_model(best_action_emb[:, -1, :])
                sim_loss = cos_sim_loss(img_proj, best_action_proj)

            else:
                raise ValueError(f"无效的损失类型：{config['sim_loss_type']}，可选info_ce/cos_sim")
            # =======================================================

            sim_loss = sim_loss * config["sim_loss_weight"]
            total_batch_loss = gen_loss + sim_loss + reverse_mse_loss_val

            # 累计损失（原有逻辑不变）
            total_gen_loss += gen_loss.item()
            total_sim_loss += sim_loss.item()
            total_reverse_mse_loss += reverse_mse_loss_val.item()
            total_loss += total_batch_loss.item()

            # 进度条显示（兼容两种损失）
            pbar.set_postfix({
                "验证总损失": f"{total_batch_loss.item():.4f}",
                "验证生成损失": f"{gen_loss.item():.4f}",
                "验证相似度损失": f"{sim_loss.item():.4f}",
                "验证反向MSE损失": f"{reverse_mse_loss_val.item():.4f}"
            })

    # 计算平均损失（原有逻辑不变）
    avg_gen_loss = total_gen_loss / min(max_val_batches, len(val_loader))
    avg_sim_loss = total_sim_loss / min(max_val_batches, len(val_loader))
    avg_reverse_mse_loss = total_reverse_mse_loss / min(max_val_batches, len(val_loader))
    avg_total_loss = total_loss / min(max_val_batches, len(val_loader))

    return avg_total_loss, avg_gen_loss, avg_sim_loss, avg_reverse_mse_loss


def main():
    best_val_loss = float('inf')
    print("=" * 50)
    print(f"开始训练（相似度损失类型：{config['sim_loss_type']}）")
    print(f"总epoch数：{config['epochs']} | 批量大小：{config['batch_size']} | 设备：{device}")
    print(f"相似度损失权重：{config['sim_loss_weight']} | 反向MSE权重：{config['reverse_mse_weight']}")
    if config["sim_loss_type"] == "info_ce":
        print(f"InfoCE温度参数：{config['info_ce_temperature']}")
    print("=" * 50)

    # 损失记录（原有逻辑不变）
    loss_records = {
        "train_total": [], "train_gen": [], "train_sim": [], "train_reverse_mse": [],
        "val_total": [], "val_gen": [], "val_sim": [], "val_reverse_mse": []
    }

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()

        # 训练+验证（原有逻辑不变）
        train_total, train_gen, train_sim, train_reverse_mse = train_one_epoch(epoch)
        sch.step()
        val_total, val_gen, val_sim, val_reverse_mse = validate_one_epoch(epoch)
        epoch_time = time.time() - epoch_start_time

        # 记录损失（原有逻辑不变）
        loss_records["train_total"].append(train_total)
        loss_records["train_gen"].append(train_gen)
        loss_records["train_sim"].append(train_sim)
        loss_records["train_reverse_mse"].append(train_reverse_mse)
        loss_records["val_total"].append(val_total)
        loss_records["val_gen"].append(val_gen)
        loss_records["val_sim"].append(val_sim)
        loss_records["val_reverse_mse"].append(val_reverse_mse)

        # 打印epoch信息（兼容两种损失）
        print("\n" + "=" * 30)
        print(f"Epoch {epoch + 1}/{config['epochs']} | 耗时：{epoch_time:.2f}秒")
        print(f"【训练集】总损失：{train_total:.4f} | 生成损失：{train_gen:.4f} | 相似度损失：{train_sim:.4f}")
        print(f"【验证集】总损失：{val_total:.4f} | 生成损失：{val_gen:.4f} | 相似度损失：{val_sim:.4f}")
        print("=" * 30 + "\n")

        # 保存最佳模型（原有逻辑不变）
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_model_path = os.path.join(config["save_path"], f"best_model_{config['sim_loss_type']}")
            torch.save({
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "sim_loss_type": config["sim_loss_type"],  # 记录使用的损失类型
                "model_states": {
                    "image_embed": image_embed.state_dict(),
                    "motor_embed": motor_embed.state_dict(),
                    "candidate_generator": candidate_generator.state_dict(),
                    "img_sim_model": img_sim_model.state_dict(),
                    "driver_sim_model": driver_sim_model.state_dict()
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": sch.state_dict(),
                "config": config
            }, best_model_path)
            print(f"✅ 保存最佳模型（验证总损失：{best_val_loss:.4f}）至：{best_model_path}")

    # 保存损失记录（原有逻辑不变）
    loss_save_path = os.path.join(config["loss_data_path"], f"loss_records_{config['sim_loss_type']}.npy")
    np.save(loss_save_path, loss_records)

    # 训练结束总结
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"使用的相似度损失类型：{config['sim_loss_type']}")
    print(f"最佳验证总损失：{best_val_loss:.4f}")
    # print(f"最佳模型路径：{os.path.join(config['save_path'], f'best_model_{config['sim_loss_type']}')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
