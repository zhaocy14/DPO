import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 路径配置（与你的代码完全一致）
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

# 导入模型（替换为Judge系列模型，保留原有生成器模型）
from DataModule.DataModule import CombinedDataset
from Model.Models import (
    ImageEmbedding,
    MotorEmbedding,
    EncoderOnlyCandidateGenerator,
    JudgeModelImage,
    JudgeModelDriver,
    JudgeModel,
    ActionExtract,
    calculate_model_size
)

# ======================== 1. 配置参数（完全复用你的配置） ========================
config = {
    # 训练参数
    "batch_size": 4,
    "epochs": 100,
    "lr": 1e-5,
    "sampling_workers": 20,
    "max_train_batches": 20,  # 限制训练批次
    "max_val_batches": 20,  # 限制验证批次

    # 生成器模型参数
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,  # 两个电机，动作维度为2
    "gen_seq_len": 30,  # 观测序列长度

    # Judge模型参数（替代原相似度模型）
    "judge_seq_len": 30,  # 预测序列长度（对应sim_seq_len）
    "embed_dim_judge": 128,
    "num_layers_judge": 3,
    "nhead_judge": 4,
    "judge_dim": 32,  # 对应similarity_dim
    "num_candidates": 5,  # 生成的候选动作数量

    # 数据和模型路径
    "data_root_dirs": '/data/cyzhao/collector_cydpo',
    "save_path": "./saved_models",
    "loss_data_path": "./loss_records"
}

# 创建保存目录（与你的代码一致）
os.makedirs(config["save_path"], exist_ok=True)
os.makedirs(config["loss_data_path"], exist_ok=True)

# 设置设备（与你的代码一致）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ======================== 2. 加载真实数据集（完全复用你的代码） ========================
data_root = config["data_root_dirs"]
data_dir_list = [
    os.path.join(data_root, file)
    for file in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, file)) and "2025" in file
]

all_dataset = CombinedDataset(
    dir_list=data_dir_list,
    frame_len=config["gen_seq_len"],
    predict_len=config['judge_seq_len'],  # 对应原sim_seq_len
    show=True
)

train_dataset = all_dataset.training_dataset
val_dataset = all_dataset.val_dataset
print(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")

# 数据加载器（与你的代码一致）
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

# ======================== 3. 模型初始化（替换为Judge系列模型） ========================
# 基础嵌入模型（与你的代码一致）
image_embed = ImageEmbedding(
    embed_dim=config["embed_dim_gen"],
    num_layers=3,
    is_resnet=False
).to(device)

motor_embed = MotorEmbedding(
    motor_dim=config["motor_dim"],
    embed_dim=config["embed_dim_gen"]
).to(device)

# 候选生成模型（与你的代码一致）
candidate_generator = EncoderOnlyCandidateGenerator(
    embed_dim=config["embed_dim_gen"],
    nhead=config["nhead_gen"],
    num_layers=config["num_layers_gen"],
    motor_dim=config["motor_dim"],
    max_seq_length=config["gen_seq_len"]
).to(device)

# Judge模型（替代原相似度模型）
judge_image_model = JudgeModelImage(
    embed_dim=config['embed_dim_judge'],
    num_frames=config['judge_seq_len'],
    num_layers=config['num_layers_judge'],
    nhead=config['nhead_judge'],
    judge_dim=config['judge_dim']
).to(device)

judge_driver_model = JudgeModelDriver(
    embed_dim=config['embed_dim_judge'],
    judge_dim=config['judge_dim'],
).to(device)

judge_total_model = JudgeModel(
    embed_dim=config['embed_dim_judge'],
    num_frames=config['judge_seq_len'],
    num_layers=config['num_layers_judge'],
    nhead=config['nhead_judge'],
    judge_dim=config['judge_dim']
).to(device)

# 打印模型大小
print("=" * 60)
print("模型大小统计（MB）：")
print(f"图像嵌入模型：{calculate_model_size(image_embed):.2f}")
print(f"电机嵌入模型：{calculate_model_size(motor_embed):.2f}")
print(f"候选生成模型：{calculate_model_size(candidate_generator):.2f}")
print(f"Judge图像模型：{calculate_model_size(judge_image_model):.2f}")
print(f"Judge电机模型：{calculate_model_size(judge_driver_model):.2f}")
print(f"Judge总模型：{calculate_model_size(judge_total_model):.2f}")
print("=" * 60)


# ======================== 4. 核心工具函数：计算候选动作的生成概率 ========================
def gaussian_pdf(x, mean, std):
    """
    计算多维高斯分布的概率密度（PDF）
    :param x: 候选动作 (batch, motor_dim)
    :param mean: 生成模型输出的均值 (batch, motor_dim)
    :param std: 生成模型输出的标准差 (batch, motor_dim)
    :return: 每个样本的PDF值 (batch,)
    """
    eps = 1e-6
    std = std + eps  # 避免除0
    # 多维高斯PDF：乘积形式（独立假设）
    pdf_per_dim = (1 / (torch.sqrt(2 * torch.pi) * std)) * torch.exp(-((x - mean) ** 2) / (2 * std ** 2))
    pdf = torch.prod(pdf_per_dim, dim=1)  # 各维度乘积 → 整体PDF
    return pdf


# ======================== 5. 损失函数（保留你的NLL损失 + Judge损失） ========================
def nll_loss(mean, std, target):
    """复用你的负对数似然损失"""
    eps = 1e-6
    std = std + eps  # 避免log(0)
    nll = torch.log(std) + (target - mean) ** 2 / (2 * std ** 2)
    return nll.mean()


def judge_mse_loss(match_scores, judge_labels):
    """Judge模型的MSE损失（匹配分数 vs one-hot标签）"""
    return F.mse_loss(match_scores, judge_labels)


# ======================== 6. 通用计算函数（核心：基于生成模型概率生成标签） ========================
def compute_loss_and_metrics(batch):
    """
    计算损失和Judge准确率
    核心逻辑：
    1. 标签 = 生成模型对候选动作的概率（PDF）最大的候选为1，其余为0
    2. 准确率 = Judge预测的最优候选 VS 生成模型概率最大的候选
    """
    # 1. 解包数据（完全复用你的解包逻辑）
    imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
    images = torch.stack([imgs1, imgs2], dim=2).to(device)  # (batch, seq, 2, 3, H, W)
    future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
    driver = driver.to(device)  # (batch, seq, motor_dim=2)
    next_driver = future_driver[:, 0, :].to(device)  # 仅用于生成器NLL损失，不参与Judge标签
    batch_size = images.shape[0]

    # 2. 基础特征提取
    image_embedded = image_embed(images)  # (batch, seq, 2*embed_dim_gen)
    motor_embedded = motor_embed(driver)  # (batch, seq, embed_dim_gen)
    future_image_embedded = image_embed(future_images)  # (batch, judge_seq_len, 2*embed_dim_gen)

    # 3. 候选动作生成（复用你的生成逻辑）
    outputs = candidate_generator(
        image_embedded=image_embedded,
        motor_embedded=motor_embedded,
        num_candidates=config["num_candidates"],
        temperature=1.0
    )
    mean = outputs['mean']  # (batch, motor_dim=2) → 生成模型的高斯均值
    std = outputs['std']  # (batch, motor_dim=2) → 生成模型的高斯标准差

    # 4. 核心：计算每个候选动作在生成模型高斯分布下的概率（PDF）
    candidate_probs = []  # 存储每个候选的概率 (num_candidates, batch)
    for candidate in outputs['candidates']:
        # candidate形状: (batch, 1, 2) → 压缩为 (batch, 2)
        candidate_2d = candidate.squeeze(1)  # (batch, motor_dim)
        # 计算该候选在生成模型高斯分布下的概率密度
        prob = gaussian_pdf(candidate_2d, mean, std)  # (batch,)
        candidate_probs.append(prob)

    # 4.1 转为张量：(num_candidates, batch) → 转置为 (batch, num_candidates)
    candidate_probs = torch.stack(candidate_probs, dim=0).T  # (batch, num_candidates)

    # 4.2 生成标签：生成模型概率最大的候选为1，其余为0
    gen_best_indices = torch.argmax(candidate_probs, dim=1)  # (batch,) → 生成模型认为最优的候选索引
    judge_labels = torch.zeros_like(candidate_probs).to(device)
    judge_labels.scatter_(1, gen_best_indices.unsqueeze(1), 1.0)  # one-hot标签

    # 5. 构建候选动作特征列表（用于Judge模型）
    candidate_emb_list = []
    for candidate in outputs['candidates']:
        candidate_emb = motor_embed(candidate)  # (batch, 1, embed_dim_gen)
        candidate_emb = candidate_emb[:, -1, :]  # (batch, embed_dim_gen)
        candidate_emb_list.append(candidate_emb)

    # 6. Judge模型前向
    judge_img_feat = judge_image_model(future_image_embedded)  # (batch, judge_dim=32)
    judge_driver_feats = [judge_driver_model(emb) for emb in candidate_emb_list]
    match_scores = judge_total_model(judge_img_feat, judge_driver_feats)  # (batch, num_candidates)

    # 7. 计算Judge准确率（核心：Judge预测 VS 生成模型最优）
    pred_best_indices = torch.argmax(match_scores, dim=1)  # Judge预测的最优候选索引
    judge_correct = (pred_best_indices == gen_best_indices).sum().item()  # 对比生成模型的最优索引
    judge_acc = judge_correct / batch_size  # 真实准确率

    # 8. 损失计算
    gen_loss = nll_loss(mean, std, next_driver)  # 生成器损失（复用你的NLL）
    judge_loss = judge_mse_loss(match_scores, judge_labels)  # Judge损失（匹配生成模型的概率标签）
    total_loss = gen_loss + judge_loss  # 总损失

    return {
        "total_loss": total_loss,
        "gen_loss": gen_loss,
        "judge_loss": judge_loss,
        "judge_acc": judge_acc,
        "batch_size": batch_size,
        "gen_best_idx": gen_best_indices,  # 调试用：生成模型最优索引
        "pred_best_idx": pred_best_indices  # 调试用：Judge预测索引
    }


# ======================== 7. 训练函数（保留你的批次限制 + 进度条） ========================
def train_one_epoch(epoch, optimizer):
    """单轮训练（复用你的tqdm进度条 + 批次限制）"""
    # 模型训练模式
    image_embed.train()
    motor_embed.train()
    candidate_generator.train()
    judge_image_model.train()
    judge_driver_model.train()
    judge_total_model.train()

    total_loss = 0.0
    gen_loss_total = 0.0
    judge_loss_total = 0.0
    judge_acc_total = 0.0
    total_samples = 0
    batch_count = 0

    # 复用你的tqdm进度条
    pbar = tqdm(
        enumerate(train_loader),
        desc=f"训练 Epoch {epoch + 1}/{config['epochs']}",
        total=min(config["max_train_batches"], len(train_loader))
    )

    for batch_idx, batch in pbar:
        if batch_count >= config["max_train_batches"]:
            pbar.write(f"\n已训练{config['max_train_batches']}个batch，提前终止当前epoch")
            break
        batch_count += 1

        optimizer.zero_grad()
        # 计算损失和准确率
        metrics = compute_loss_and_metrics(batch)
        # 反向传播
        metrics["total_loss"].backward()
        optimizer.step()

        # 累计指标
        total_loss += metrics["total_loss"].item() * metrics["batch_size"]
        gen_loss_total += metrics["gen_loss"].item() * metrics["batch_size"]
        judge_loss_total += metrics["judge_loss"].item() * metrics["batch_size"]
        judge_acc_total += metrics["judge_acc"] * metrics["batch_size"]
        total_samples += metrics["batch_size"]

        # 更新进度条（保留你的格式 + 新增Judge准确率）
        pbar.set_postfix({
            "总损失": f"{metrics['total_loss'].item():.4f}",
            "生成损失": f"{metrics['gen_loss'].item():.4f}",
            "Judge损失": f"{metrics['judge_loss'].item():.4f}",
            "Judge准确率": f"{metrics['judge_acc']:.2%}"
        })

    # 计算平均指标
    avg_total_loss = total_loss / total_samples
    avg_gen_loss = gen_loss_total / total_samples
    avg_judge_loss = judge_loss_total / total_samples
    avg_judge_acc = judge_acc_total / total_samples

    return {
        "total_loss": avg_total_loss,
        "gen_loss": avg_gen_loss,
        "judge_loss": avg_judge_loss,
        "judge_acc": avg_judge_acc
    }


# ======================== 8. 验证函数（保留你的批次限制 + 进度条） ========================
def validate_one_epoch(epoch):
    """单轮验证（复用你的tqdm进度条 + 批次限制）"""
    # 模型评估模式
    image_embed.eval()
    motor_embed.eval()
    candidate_generator.eval()
    judge_image_model.eval()
    judge_driver_model.eval()
    judge_total_model.eval()

    total_loss = 0.0
    gen_loss_total = 0.0
    judge_loss_total = 0.0
    judge_acc_total = 0.0
    total_samples = 0
    batch_count = 0

    # 复用你的tqdm进度条
    pbar = tqdm(
        enumerate(val_loader),
        desc=f"验证 Epoch {epoch + 1}/{config['epochs']}",
        total=min(config["max_val_batches"], len(val_loader))
    )

    with torch.no_grad():  # 关闭梯度
        for batch_idx, batch in pbar:
            if batch_count >= config["max_val_batches"]:
                pbar.write(f"\n已验证{config['max_val_batches']}个batch，提前终止当前epoch")
                break
            batch_count += 1

            # 计算损失和准确率
            metrics = compute_loss_and_metrics(batch)

            # 累计指标
            total_loss += metrics["total_loss"].item() * metrics["batch_size"]
            gen_loss_total += metrics["gen_loss"].item() * metrics["batch_size"]
            judge_loss_total += metrics["judge_loss"].item() * metrics["batch_size"]
            judge_acc_total += metrics["judge_acc"] * metrics["batch_size"]
            total_samples += metrics["batch_size"]

            # 更新进度条（保留你的格式 + 新增Judge准确率）
            pbar.set_postfix({
                "验证总损失": f"{metrics['total_loss'].item():.4f}",
                "验证生成损失": f"{metrics['gen_loss'].item():.4f}",
                "验证Judge损失": f"{metrics['judge_loss'].item():.4f}",
                "验证Judge准确率": f"{metrics['judge_acc']:.2%}"
            })

    # 计算平均指标
    avg_total_loss = total_loss / total_samples
    avg_gen_loss = gen_loss_total / total_samples
    avg_judge_loss = judge_loss_total / total_samples
    avg_judge_acc = judge_acc_total / total_samples

    return {
        "total_loss": avg_total_loss,
        "gen_loss": avg_gen_loss,
        "judge_loss": avg_judge_loss,
        "judge_acc": avg_judge_acc
    }


# ======================== 9. 主训练流程（复用你的逻辑 + 新增Judge指标） ========================
def main():
    # 优化器（复用你的参数分组 + 学习率）
    optimizer = torch.optim.Adam(
        params=[
            {'params': image_embed.parameters()},
            {'params': motor_embed.parameters()},
            {'params': candidate_generator.parameters()},
            {'params': judge_image_model.parameters()},
            {'params': judge_driver_model.parameters()},
            {'params': judge_total_model.parameters()}
        ],
        lr=config['lr']
    )

    # 学习率调度器（复用你的StepLR）
    sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 初始化最佳指标
    best_val_loss = float('inf')
    best_val_judge_acc = 0.0

    # 记录损失和准确率
    records = {
        "train_total": [], "train_gen": [], "train_judge": [], "train_judge_acc": [],
        "val_total": [], "val_gen": [], "val_judge": [], "val_judge_acc": []
    }

    # 训练开始提示（复用你的格式）
    print("=" * 50)
    print("开始训练（含验证集评估 + Judge准确率）")
    print(f"总epoch数：{config['epochs']} | 批量大小：{config['batch_size']} | 设备：{device}")
    print(f"loss数据保存目录：{config['loss_data_path']}")
    print("=" * 50)

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()

        # 训练
        train_metrics = train_one_epoch(epoch, optimizer)
        # 学习率调度
        sch.step()
        # 验证
        val_metrics = validate_one_epoch(epoch)

        # 计算耗时
        epoch_time = time.time() - epoch_start_time

        # 记录指标
        records["train_total"].append(train_metrics["total_loss"])
        records["train_gen"].append(train_metrics["gen_loss"])
        records["train_judge"].append(train_metrics["judge_loss"])
        records["train_judge_acc"].append(train_metrics["judge_acc"])
        records["val_total"].append(val_metrics["total_loss"])
        records["val_gen"].append(val_metrics["gen_loss"])
        records["val_judge"].append(val_metrics["judge_loss"])
        records["val_judge_acc"].append(val_metrics["judge_acc"])

        # 打印epoch信息（复用你的格式 + 新增Judge准确率）
        print("\n" + "=" * 30)
        print(f"Epoch {epoch + 1}/{config['epochs']} | 耗时：{epoch_time:.2f}秒")
        print(
            f"【训练集】总损失：{train_metrics['total_loss']:.4f} | 生成损失：{train_metrics['gen_loss']:.4f} | Judge损失：{train_metrics['judge_loss']:.4f} | Judge准确率：{train_metrics['judge_acc']:.2%}")
        print(
            f"【验证集】总损失：{val_metrics['total_loss']:.4f} | 生成损失：{val_metrics['gen_loss']:.4f} | Judge损失：{val_metrics['judge_loss']:.4f} | Judge准确率：{val_metrics['judge_acc']:.2%}")
        print("=" * 30 + "\n")

        # 保存最佳模型（基于验证总损失）
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_val_judge_acc = val_metrics["judge_acc"]
            best_model_path = os.path.join(config["save_path"], "best_model")
            torch.save({
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_val_judge_acc": best_val_judge_acc,
                "model_states": {
                    "image_embed": image_embed.state_dict(),
                    "motor_embed": motor_embed.state_dict(),
                    "candidate_generator": candidate_generator.state_dict(),
                    "judge_image_model": judge_image_model.state_dict(),
                    "judge_driver_model": judge_driver_model.state_dict(),
                    "judge_total_model": judge_total_model.state_dict()
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": sch.state_dict(),
                "config": config
            }, best_model_path)

            print(
                f"✅ 保存最佳模型（验证总损失：{best_val_loss:.4f} | 验证Judge准确率：{best_val_judge_acc:.2%}）至：{best_model_path}")

    # 保存损失和准确率记录
    record_save_path = os.path.join(config["loss_data_path"], "train_val_records.npy")
    np.save(record_save_path, records)

    # 训练结束总结（复用你的格式 + 新增Judge准确率）
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳验证总损失：{best_val_loss:.4f}")
    print(f"最佳验证Judge准确率：{best_val_judge_acc:.2%}")
    print(f"最佳模型路径：{os.path.join(config['save_path'], 'best_model')}")
    print(f"记录保存路径：{record_save_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
