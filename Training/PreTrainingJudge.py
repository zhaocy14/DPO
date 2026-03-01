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

# 导入模型（补充ActionExtract）
from DataModule.DataModule import CombinedDataset
from Model.Models import (
    ImageEmbedding,
    MotorEmbedding,
    EncoderOnlyCandidateGenerator,
    JudgeModelImage,
    JudgeModelDriver,
    JudgeModel,
    ActionExtract,  # 补充遗漏的ActionExtract模型
    calculate_model_size
)

# ======================== 1. 配置参数（补充ActionExtract相关） ========================
config = {
    # 训练参数
    "batch_size": 4,
    "epochs": 100,
    "lr": 1e-5,
    "sampling_workers": 20,
    "max_train_batches": 20,  # 可后续改为len(train_loader)
    "max_val_batches": 20,  # 可后续改为len(val_loader)

    # 生成器模型参数
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,  # 两个电机，动作维度为2
    "gen_seq_len": 30,  # 观测序列长度

    # Judge模型参数
    "judge_seq_len": 30,  # 预测序列长度
    "embed_dim_judge": 128,
    "num_layers_judge": 3,
    "nhead_judge": 4,
    "judge_dim": 32,
    "num_candidates": 5,  # 生成的候选动作数量

    # ActionExtract模型参数（新增）
    "action_extract_dim": 32,  # 与judge_dim对齐
    "action_extract_out_dim": 2,  # 输出动作维度（与motor_dim一致）

    # 数据和模型路径
    "data_root_dirs": '/data/cyzhao/collector_cydpo',
    "save_path": "./saved_models",
    "loss_data_path": "./loss_records"
}

# 创建保存目录
os.makedirs(config["save_path"], exist_ok=True)
os.makedirs(config["loss_data_path"], exist_ok=True)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ======================== 2. 加载数据集（完全复用） ========================
data_root = config["data_root_dirs"]
data_dir_list = [
    os.path.join(data_root, file)
    for file in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, file)) and "2025" in file
]

all_dataset = CombinedDataset(
    dir_list=data_dir_list,
    frame_len=config["gen_seq_len"],
    predict_len=config['judge_seq_len'],
    show=True
)

train_dataset = all_dataset.training_dataset
val_dataset = all_dataset.val_dataset
print(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")

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

# ======================== 3. 模型初始化（补充ActionExtract） ========================
# 基础嵌入模型
image_embed = ImageEmbedding(
    embed_dim=config["embed_dim_gen"],
    num_layers=3,
    is_resnet=False
).to(device)

motor_embed = MotorEmbedding(
    motor_dim=config["motor_dim"],
    embed_dim=config["embed_dim_gen"]
).to(device)

# 候选生成模型
candidate_generator = EncoderOnlyCandidateGenerator(
    embed_dim=config["embed_dim_gen"],
    nhead=config["nhead_gen"],
    num_layers=config["num_layers_gen"],
    motor_dim=config["motor_dim"],
    max_seq_length=config["gen_seq_len"]
).to(device)

# Judge模型
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

# 补充ActionExtract模型（核心新增）
action_extract_model = ActionExtract(
    in_dim=config['judge_dim'],  # 输入=judge_image_model的输出维度
    hidden_dim=config['action_extract_dim'],
    out_dim=config['action_extract_out_dim']  # 输出动作维度
).to(device)

# 打印模型大小（含ActionExtract）
print("=" * 60)
print("模型大小统计（MB）：")
print(f"图像嵌入模型：{calculate_model_size(image_embed):.2f}")
print(f"电机嵌入模型：{calculate_model_size(motor_embed):.2f}")
print(f"候选生成模型：{calculate_model_size(candidate_generator):.2f}")
print(f"Judge图像模型：{calculate_model_size(judge_image_model):.2f}")
print(f"Judge电机模型：{calculate_model_size(judge_driver_model):.2f}")
print(f"Judge总模型：{calculate_model_size(judge_total_model):.2f}")
print(f"ActionExtract模型：{calculate_model_size(action_extract_model):.2f}")  # 新增
print("=" * 60)


# ======================== 4. 高斯PDF函数（保留之前的修正） ========================
def gaussian_pdf(x, mean, std):
    eps = 1e-6
    std = std + eps

    two_pi = torch.tensor(2 * np.pi, dtype=std.dtype, device=std.device)
    sqrt_two_pi = torch.sqrt(two_pi)

    pdf_per_dim = (1 / (sqrt_two_pi * std)) * torch.exp(-((x - mean) ** 2) / (2 * std ** 2))
    pdf = torch.prod(pdf_per_dim, dim=1)
    pdf = pdf ** 2  # 放大概率差异，强化最优候选
    return pdf


# ======================== 5. 损失函数（核心调整：Judge用CE，新增ActionExtract的MSE） ========================
def nll_loss(mean, std, target):
    """生成器NLL损失（复用）"""
    eps = 1e-6
    std = std + eps
    nll = torch.log(std) + (target - mean) ** 2 / (2 * std ** 2)
    return nll.mean()


def judge_ce_loss(match_scores, gen_best_indices):
    """Judge交叉熵损失（分类任务，替代MSE）"""
    temperature = 0.1  # 放大分数差异，提升准确率
    match_scores = match_scores / temperature
    return F.cross_entropy(match_scores, gen_best_indices)


def action_extract_mse_loss(pred_action, true_action):
    """ActionExtract的MSE损失（预测动作 vs 真实动作）"""
    return F.mse_loss(pred_action, true_action)


# ======================== 6. 通用计算函数（核心修改） ========================
def compute_loss_and_metrics(batch):
    # 1. 解包数据
    imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
    images = torch.stack([imgs1, imgs2], dim=2).to(device)
    future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
    driver = driver.to(device)
    # 真实动作序列：取future_driver的完整序列（而非仅第一个），用于ActionExtract的MSE
    true_action_seq = future_driver.to(device)  # (batch, judge_seq_len, motor_dim)
    next_driver = future_driver[:, 0, :].to(device)  # 生成器用的真实动作
    batch_size = images.shape[0]

    # 2. 基础特征提取
    image_embedded = image_embed(images)
    motor_embedded = motor_embed(driver)
    future_image_embedded = image_embed(future_images)

    # 3. 候选动作生成
    outputs = candidate_generator(
        image_embedded=image_embedded,
        motor_embedded=motor_embedded,
        num_candidates=config["num_candidates"],
        temperature=2.0  # 提高温度，增大候选动作差异
    )
    mean = outputs['mean']
    std = outputs['std']

    # 4. 计算生成模型的候选概率
    candidate_probs = []
    for candidate in outputs['candidates']:
        candidate_2d = candidate.squeeze(1)
        prob = gaussian_pdf(candidate_2d, mean, std)
        candidate_probs.append(prob)
    candidate_probs = torch.stack(candidate_probs, dim=0).T
    gen_best_indices = torch.argmax(candidate_probs, dim=1)  # 生成模型最优索引

    # 5. Judge模型前向
    judge_img_feat = judge_image_model(future_image_embedded)  # (batch, judge_dim)
    # ===== 核心新增：ActionExtract前向 =====
    # judge_image_model的输出 → ActionExtract → 预测动作序列
    pred_action_seq = action_extract_model(judge_img_feat)  # (batch, action_extract_out_dim)
    # 适配维度：扩展为(batch, judge_seq_len, motor_dim)（与真实动作序列对齐）
    pred_action_seq = pred_action_seq.unsqueeze(1).expand(-1, config['judge_seq_len'], -1)

    # 6. Judge总模型前向
    candidate_emb_list = []
    for candidate in outputs['candidates']:
        candidate_emb = motor_embed(candidate)
        candidate_emb = candidate_emb[:, -1, :]
        candidate_emb_list.append(candidate_emb)
    judge_driver_feats = [judge_driver_model(emb) for emb in candidate_emb_list]
    match_scores = judge_total_model(judge_img_feat, judge_driver_feats)

    # 7. 计算指标
    pred_best_indices = torch.argmax(match_scores, dim=1)
    judge_correct = (pred_best_indices == gen_best_indices).sum().item()
    judge_acc = judge_correct / batch_size
    # 新增：ActionExtract的Top-K准确率（可选）
    top_2_acc = torch.sum(
        torch.topk(match_scores, 2, dim=1).indices == gen_best_indices.unsqueeze(1)).item() / batch_size

    # 8. 损失计算（核心调整）
    gen_loss = nll_loss(mean, std, next_driver)  # 生成器损失
    judge_loss = judge_ce_loss(match_scores, gen_best_indices)  # Judge交叉熵损失（分类）
    action_loss = action_extract_mse_loss(pred_action_seq, true_action_seq)  # ActionExtract的MSE损失
    # 总损失：平衡三个损失，约束judge_image_model不受动作序列干扰
    total_loss = 0.5 * gen_loss + 1.0 * judge_loss + 1.0 * action_loss

    return {
        "total_loss": total_loss,
        "gen_loss": gen_loss,
        "judge_loss": judge_loss,
        "action_loss": action_loss,  # 新增
        "judge_acc": judge_acc,
        "top_2_acc": top_2_acc,  # 新增
        "batch_size": batch_size,
        "gen_best_idx": gen_best_indices,
        "pred_best_idx": pred_best_indices
    }


# ======================== 7. 训练/验证函数（补充ActionExtract的模式切换） ========================
def train_one_epoch(epoch, optimizer):
    # 所有模型设为训练模式（含ActionExtract）
    image_embed.train()
    motor_embed.train()
    candidate_generator.train()
    judge_image_model.train()
    judge_driver_model.train()
    judge_total_model.train()
    action_extract_model.train()  # 新增

    total_loss = 0.0
    gen_loss_total = 0.0
    judge_loss_total = 0.0
    action_loss_total = 0.0  # 新增
    judge_acc_total = 0.0
    top_2_acc_total = 0.0  # 新增
    total_samples = 0
    batch_count = 0

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
        metrics = compute_loss_and_metrics(batch)
        metrics["total_loss"].backward()  # 反向传播：action_loss会约束judge_image_model
        optimizer.step()

        # 累计指标
        total_loss += metrics["total_loss"].item() * metrics["batch_size"]
        gen_loss_total += metrics["gen_loss"].item() * metrics["batch_size"]
        judge_loss_total += metrics["judge_loss"].item() * metrics["batch_size"]
        action_loss_total += metrics["action_loss"].item() * metrics["batch_size"]  # 新增
        judge_acc_total += metrics["judge_acc"] * metrics["batch_size"]
        top_2_acc_total += metrics["top_2_acc"] * metrics["batch_size"]  # 新增
        total_samples += metrics["batch_size"]

        # 更新进度条（补充action_loss和top_2_acc）
        pbar.set_postfix({
            "总损失": f"{metrics['total_loss'].item():.4f}",
            "生成损失": f"{metrics['gen_loss'].item():.4f}",
            "Judge损失": f"{metrics['judge_loss'].item():.4f}",
            "Action损失": f"{metrics['action_loss'].item():.4f}",  # 新增
            "Judge准确率": f"{metrics['judge_acc']:.2%}",
            "Top-2准确率": f"{metrics['top_2_acc']:.2%}"  # 新增
        })

    # 计算平均指标
    avg_total_loss = total_loss / total_samples
    avg_gen_loss = gen_loss_total / total_samples
    avg_judge_loss = judge_loss_total / total_samples
    avg_action_loss = action_loss_total / total_samples  # 新增
    avg_judge_acc = judge_acc_total / total_samples
    avg_top_2_acc = top_2_acc_total / total_samples  # 新增

    return {
        "total_loss": avg_total_loss,
        "gen_loss": avg_gen_loss,
        "judge_loss": avg_judge_loss,
        "action_loss": avg_action_loss,  # 新增
        "judge_acc": avg_judge_acc,
        "top_2_acc": avg_top_2_acc  # 新增
    }


def validate_one_epoch(epoch):
    # 所有模型设为评估模式（含ActionExtract）
    image_embed.eval()
    motor_embed.eval()
    candidate_generator.eval()
    judge_image_model.eval()
    judge_driver_model.eval()
    judge_total_model.eval()
    action_extract_model.eval()  # 新增

    total_loss = 0.0
    gen_loss_total = 0.0
    judge_loss_total = 0.0
    action_loss_total = 0.0  # 新增
    judge_acc_total = 0.0
    top_2_acc_total = 0.0  # 新增
    total_samples = 0
    batch_count = 0

    pbar = tqdm(
        enumerate(val_loader),
        desc=f"验证 Epoch {epoch + 1}/{config['epochs']}",
        total=min(config["max_val_batches"], len(val_loader))
    )

    with torch.no_grad():
        for batch_idx, batch in pbar:
            if batch_count >= config["max_val_batches"]:
                pbar.write(f"\n已验证{config['max_val_batches']}个batch，提前终止当前epoch")
                break
            batch_count += 1

            metrics = compute_loss_and_metrics(batch)

            # 累计指标
            total_loss += metrics["total_loss"].item() * metrics["batch_size"]
            gen_loss_total += metrics["gen_loss"].item() * metrics["batch_size"]
            judge_loss_total += metrics["judge_loss"].item() * metrics["batch_size"]
            action_loss_total += metrics["action_loss"].item() * metrics["batch_size"]  # 新增
            judge_acc_total += metrics["judge_acc"] * metrics["batch_size"]
            top_2_acc_total += metrics["top_2_acc"] * metrics["batch_size"]  # 新增
            total_samples += metrics["batch_size"]

            # 更新进度条
            pbar.set_postfix({
                "验证总损失": f"{metrics['total_loss'].item():.4f}",
                "验证生成损失": f"{metrics['gen_loss'].item():.4f}",
                "验证Judge损失": f"{metrics['judge_loss'].item():.4f}",
                "验证Action损失": f"{metrics['action_loss'].item():.4f}",  # 新增
                "验证Judge准确率": f"{metrics['judge_acc']:.2%}",
                "验证Top-2准确率": f"{metrics['top_2_acc']:.2%}"  # 新增
            })

    # 计算平均指标
    avg_total_loss = total_loss / total_samples
    avg_gen_loss = gen_loss_total / total_samples
    avg_judge_loss = judge_loss_total / total_samples
    avg_action_loss = action_loss_total / total_samples  # 新增
    avg_judge_acc = judge_acc_total / total_samples
    avg_top_2_acc = top_2_acc_total / total_samples  # 新增

    return {
        "total_loss": avg_total_loss,
        "gen_loss": avg_gen_loss,
        "judge_loss": avg_judge_loss,
        "action_loss": avg_action_loss,  # 新增
        "judge_acc": avg_judge_acc,
        "top_2_acc": avg_top_2_acc  # 新增
    }


# ======================== 8. 主训练流程（补充ActionExtract到优化器） ========================
def main():
    # 优化器：包含ActionExtract模型参数，为Judge模型单独设置更高lr
    optimizer = torch.optim.Adam(
        params=[
            {'params': image_embed.parameters()},
            {'params': motor_embed.parameters()},
            {'params': candidate_generator.parameters()},
            # Judge模型单独高lr
            {'params': judge_image_model.parameters(), 'lr': 1e-4},
            {'params': judge_driver_model.parameters(), 'lr': 1e-4},
            {'params': judge_total_model.parameters(), 'lr': 1e-4},
            # ActionExtract模型参数（新增）
            {'params': action_extract_model.parameters(), 'lr': 1e-4}
        ],
        lr=config['lr']  # 生成模型lr=1e-5
    )

    # 学习率调度器
    sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 初始化最佳指标
    best_val_loss = float('inf')
    best_val_judge_acc = 0.0

    # 记录指标（补充action_loss和top_2_acc）
    records = {
        "train_total": [], "train_gen": [], "train_judge": [], "train_action": [], "train_judge_acc": [],
        "train_top_2_acc": [],
        "val_total": [], "val_gen": [], "val_judge": [], "val_action": [], "val_judge_acc": [], "val_top_2_acc": []
    }

    # 训练开始提示
    print("=" * 50)
    print("开始训练（含ActionExtract + Judge交叉熵损失）")
    print(f"总epoch数：{config['epochs']} | 批量大小：{config['batch_size']} | 设备：{device}")
    print(f"loss数据保存目录：{config['loss_data_path']}")
    print("=" * 50)

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()

        # 训练
        train_metrics = train_one_epoch(epoch, optimizer)
        sch.step()
        # 验证
        val_metrics = validate_one_epoch(epoch)

        # 计算耗时
        epoch_time = time.time() - epoch_start_time

        # 记录指标
        records["train_total"].append(train_metrics["total_loss"])
        records["train_gen"].append(train_metrics["gen_loss"])
        records["train_judge"].append(train_metrics["judge_loss"])
        records["train_action"].append(train_metrics["action_loss"])  # 新增
        records["train_judge_acc"].append(train_metrics["judge_acc"])
        records["train_top_2_acc"].append(train_metrics["top_2_acc"])  # 新增
        records["val_total"].append(val_metrics["total_loss"])
        records["val_gen"].append(val_metrics["gen_loss"])
        records["val_judge"].append(val_metrics["judge_loss"])
        records["val_action"].append(val_metrics["action_loss"])  # 新增
        records["val_judge_acc"].append(val_metrics["judge_acc"])
        records["val_top_2_acc"].append(val_metrics["top_2_acc"])  # 新增

        # 打印epoch信息
        print("\n" + "=" * 30)
        print(f"Epoch {epoch + 1}/{config['epochs']} | 耗时：{epoch_time:.2f}秒")
        print(
            f"【训练集】总损失：{train_metrics['total_loss']:.4f} | 生成损失：{train_metrics['gen_loss']:.4f} | Judge损失：{train_metrics['judge_loss']:.4f} | Action损失：{train_metrics['action_loss']:.4f} | Judge准确率：{train_metrics['judge_acc']:.2%} | Top-2准确率：{train_metrics['top_2_acc']:.2%}")
        print(
            f"【验证集】总损失：{val_metrics['total_loss']:.4f} | 生成损失：{val_metrics['gen_loss']:.4f} | Judge损失：{val_metrics['judge_loss']:.4f} | Action损失：{val_metrics['action_loss']:.4f} | Judge准确率：{val_metrics['judge_acc']:.2%} | Top-2准确率：{val_metrics['top_2_acc']:.2%}")
        print("=" * 30 + "\n")

        # 保存最佳模型（含ActionExtract）
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
                    "judge_total_model": judge_total_model.state_dict(),
                    "action_extract_model": action_extract_model.state_dict()  # 新增
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": sch.state_dict(),
                "config": config
            }, best_model_path)

            print(
                f"✅ 保存最佳模型（验证总损失：{best_val_loss:.4f} | 验证Judge准确率：{best_val_judge_acc:.2%}）至：{best_model_path}")

    # 保存记录
    record_save_path = os.path.join(config["loss_data_path"], "train_val_records.npy")
    np.save(record_save_path, records)

    # 训练结束总结
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳验证总损失：{best_val_loss:.4f}")
    print(f"最佳验证Judge准确率：{best_val_judge_acc:.2%}")
    print(f"最佳模型路径：{os.path.join(config['save_path'], 'best_model')}")
    print(f"记录保存路径：{record_save_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
