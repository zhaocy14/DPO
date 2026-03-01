import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 路径配置（确保能导入Models.py）
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

# 导入自定义模型
from Model.Models import (
    ImageEmbedding, MotorEmbedding, EncoderOnlyCandidateGenerator,
    JudgeModelImage, JudgeModelDriver, JudgeModel, ActionExtract,
    calculate_model_size
)


# ======================== 1. 配置参数 ========================
class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据配置
    batch_size = 4
    seq_length = 30  # 图像序列长度
    motor_dim = 2  # 电机动作维度
    num_candidates = 5  # 生成的候选动作数量

    # 模型配置
    embed_dim_gen = 128  # 基础嵌入维度
    judge_dim = 32  # 判断模型输出维度
    num_layers_gen = 16  # 生成器Transformer层数
    nhead_gen = 8  # 生成器注意力头数
    num_layers_judge = 2  # 判断模型Transformer层数
    nhead_judge = 4  # 判断模型注意力头数

    # 训练配置
    lr = 1e-4
    epochs = 100
    weight_decay = 1e-5
    temperature = 0.5  # 候选生成温度系数
    save_interval = 10  # 模型保存间隔
    val_split = 0.2  # 训练/验证数据分割比例


config = Config()


# ======================== 2. 数据集（训练+验证） ========================
class PretrainDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=30, motor_dim=2, is_train=True):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.motor_dim = motor_dim
        self.is_train = is_train
        # 固定随机种子，确保验证集数据稳定
        torch.manual_seed(42 if is_train else 100)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟输入：(seq, num_cameras=2, 3, H=64, W=64)
        images = torch.randn(self.seq_length, 2, 3, 64, 64)
        # 模拟电机动作：(seq, motor_dim)
        motor_data = torch.randn(self.seq_length, self.motor_dim)
        # 模拟未来图像（用于判断模型）：(seq, 2, 3, 64, 64)
        future_images = torch.randn(self.seq_length, 2, 3, 64, 64)
        # 模拟真实电机信号（仅用于ActionExtract损失）
        real_motor_signal = torch.randn(self.motor_dim)

        return {
            "images": images,
            "motor_data": motor_data,
            "future_images": future_images,
            "real_motor_signal": real_motor_signal
        }


# ======================== 3. 模型初始化 ========================
def init_models():
    """初始化所有模型并移到指定设备"""
    # 嵌入模型
    image_embed_model = ImageEmbedding(
        embed_dim=config.embed_dim_gen,
        num_layers=3,
        is_resnet=False
    ).to(config.device)

    motor_embed_model = MotorEmbedding(
        motor_dim=config.motor_dim,
        embed_dim=config.embed_dim_gen
    ).to(config.device)

    # 候选生成模型
    candidate_generator = EncoderOnlyCandidateGenerator(
        embed_dim=config.embed_dim_gen,
        nhead=config.nhead_gen,
        num_layers=config.num_layers_gen,
        motor_dim=config.motor_dim,
        max_seq_length=config.seq_length
    ).to(config.device)

    # 判断模型
    judge_image_model = JudgeModelImage(
        embed_dim=config.embed_dim_gen,
        num_frames=config.seq_length,
        num_layers=config.num_layers_judge,
        nhead=config.nhead_judge,
        judge_dim=config.judge_dim
    ).to(config.device)

    judge_driver_model = JudgeModelDriver(
        embed_dim=config.embed_dim_gen,
        judge_dim=config.judge_dim
    ).to(config.device)

    judge_total_model = JudgeModel(
        embed_dim=config.embed_dim_gen,
        num_frames=config.seq_length,
        num_layers=config.num_layers_judge,
        nhead=config.nhead_judge,
        judge_dim=config.judge_dim
    ).to(config.device)

    # 动作提取模型
    action_extract_model = ActionExtract(
        in_dim=config.judge_dim,
        hidden_dim=64,
        out_dim=config.motor_dim,
        dropout_rate=0.2
    ).to(config.device)

    # 打印模型大小
    print("=" * 60)
    print("模型大小统计（MB）：")
    print(f"图像嵌入模型：{calculate_model_size(image_embed_model):.2f}")
    print(f"电机嵌入模型：{calculate_model_size(motor_embed_model):.2f}")
    print(f"候选生成模型：{calculate_model_size(candidate_generator):.2f}")
    print(f"图像判断模型：{calculate_model_size(judge_image_model):.2f}")
    print(f"电机判断模型：{calculate_model_size(judge_driver_model):.2f}")
    print(f"总判断模型：{calculate_model_size(judge_total_model):.2f}")
    print(f"动作提取模型：{calculate_model_size(action_extract_model):.2f}")
    print("=" * 60)

    return {
        "image_embed": image_embed_model,
        "motor_embed": motor_embed_model,
        "candidate_generator": candidate_generator,
        "judge_image": judge_image_model,
        "judge_driver": judge_driver_model,
        "judge_total": judge_total_model,
        "action_extract": action_extract_model
    }


# ======================== 4. 训练/验证通用函数（计算损失+准确率） ========================
def compute_loss_and_metrics(model_dict, batch_data):
    """计算损失和Judge准确率（训练/验证通用）"""
    # 1. 数据预处理
    images = batch_data["images"].to(config.device)
    motor_data = batch_data["motor_data"].to(config.device)
    future_images = batch_data["future_images"].to(config.device)
    real_motor_signal = batch_data["real_motor_signal"].to(config.device)

    batch_size = images.shape[0]

    # 2. 基础特征提取
    image_embedded = model_dict["image_embed"](images)
    motor_embedded = model_dict["motor_embed"](motor_data)
    future_image_embedded = model_dict["image_embed"](future_images)

    # 3. 生成候选动作
    gen_outputs = model_dict["candidate_generator"](
        image_embedded=image_embedded,
        motor_embedded=motor_embedded,
        num_candidates=config.num_candidates,
        temperature=config.temperature
    )

    # 4. 构建候选动作特征列表
    candidate_emb_list = []
    for candidate in gen_outputs["candidates"]:
        candidate_emb = model_dict["motor_embed"](candidate)
        candidate_emb = candidate_emb[:, -1, :]
        candidate_emb_list.append(candidate_emb)

    # 5. Judge Model 前向
    judge_img_feat = model_dict["judge_image"](future_image_embedded)
    judge_driver_feats = [model_dict["judge_driver"](emb) for emb in candidate_emb_list]
    match_scores = model_dict["judge_total"](judge_img_feat, judge_driver_feats)

    # 6. 生成Judge标签 + 计算准确率
    # 6.1 生成标签
    max_indices = torch.argmax(match_scores, dim=1)
    judge_labels = torch.zeros_like(match_scores)
    judge_labels.scatter_(1, max_indices.unsqueeze(1), 1.0)

    # 6.2 计算Judge准确率：预测的最大索引 == 标签的最大索引（即模型是否选对最优候选）
    pred_max_indices = torch.argmax(match_scores, dim=1)
    label_max_indices = torch.argmax(judge_labels, dim=1)
    judge_correct = (pred_max_indices == label_max_indices).sum().item()
    judge_acc = judge_correct / batch_size

    # 7. ActionExtract 前向
    motor_pred = model_dict["action_extract"](judge_img_feat)

    # 8. 损失计算
    best_candidate = gen_outputs["candidates"][max_indices[0]]
    gen_loss = F.mse_loss(gen_outputs["mean"], best_candidate.squeeze(1))
    judge_loss = F.mse_loss(match_scores, judge_labels)
    action_loss = F.mse_loss(motor_pred, real_motor_signal)
    total_loss = gen_loss + judge_loss + action_loss

    return {
        "total_loss": total_loss,
        "gen_loss": gen_loss,
        "judge_loss": judge_loss,
        "action_loss": action_loss,
        "judge_acc": judge_acc,
        "batch_size": batch_size
    }


# ======================== 5. 训练函数 ========================
def train_one_epoch(epoch, model_dict, dataloader, optimizer):
    """单轮训练"""
    for model in model_dict.values():
        model.train()

    total_loss = 0.0
    gen_loss_total = 0.0
    judge_loss_total = 0.0
    action_loss_total = 0.0
    judge_acc_total = 0.0
    total_samples = 0

    for batch_idx, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        # 计算损失和准确率
        metrics = compute_loss_and_metrics(model_dict, batch_data)

        # 反向传播
        metrics["total_loss"].backward()
        optimizer.step()

        # 累计指标
        total_loss += metrics["total_loss"].item() * metrics["batch_size"]
        gen_loss_total += metrics["gen_loss"].item() * metrics["batch_size"]
        judge_loss_total += metrics["judge_loss"].item() * metrics["batch_size"]
        action_loss_total += metrics["action_loss"].item() * metrics["batch_size"]
        judge_acc_total += metrics["judge_acc"] * metrics["batch_size"]
        total_samples += metrics["batch_size"]

        # 打印批次信息
        if (batch_idx + 1) % 10 == 0:
            print(f"[Train] Epoch [{epoch + 1}/{config.epochs}] | Batch [{batch_idx + 1}/{len(dataloader)}] | "
                  f"Total Loss: {metrics['total_loss'].item():.4f} | "
                  f"Judge Acc: {metrics['judge_acc']:.2%} | "
                  f"Gen Loss: {metrics['gen_loss'].item():.4f} | "
                  f"Judge Loss: {metrics['judge_loss'].item():.4f}")

    # 计算平均指标
    avg_total_loss = total_loss / total_samples
    avg_gen_loss = gen_loss_total / total_samples
    avg_judge_loss = judge_loss_total / total_samples
    avg_action_loss = action_loss_total / total_samples
    avg_judge_acc = judge_acc_total / total_samples

    return {
        "total_loss": avg_total_loss,
        "gen_loss": avg_gen_loss,
        "judge_loss": avg_judge_loss,
        "action_loss": avg_action_loss,
        "judge_acc": avg_judge_acc
    }


# ======================== 6. 验证函数（新增） ========================
def validate_one_epoch(epoch, model_dict, dataloader):
    """单轮验证（无反向传播）"""
    for model in model_dict.values():
        model.eval()  # 评估模式：关闭dropout/batchnorm

    total_loss = 0.0
    gen_loss_total = 0.0
    judge_loss_total = 0.0
    action_loss_total = 0.0
    judge_acc_total = 0.0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算，节省显存
        for batch_idx, batch_data in enumerate(dataloader):
            # 计算损失和准确率
            metrics = compute_loss_and_metrics(model_dict, batch_data)

            # 累计指标
            total_loss += metrics["total_loss"].item() * metrics["batch_size"]
            gen_loss_total += metrics["gen_loss"].item() * metrics["batch_size"]
            judge_loss_total += metrics["judge_loss"].item() * metrics["batch_size"]
            action_loss_total += metrics["action_loss"].item() * metrics["batch_size"]
            judge_acc_total += metrics["judge_acc"] * metrics["batch_size"]
            total_samples += metrics["batch_size"]

            # 打印批次信息
            if (batch_idx + 1) % 10 == 0:
                print(f"[Val] Epoch [{epoch + 1}/{config.epochs}] | Batch [{batch_idx + 1}/{len(dataloader)}] | "
                      f"Total Loss: {metrics['total_loss'].item():.4f} | "
                      f"Judge Acc: {metrics['judge_acc']:.2%}")

    # 计算平均指标
    avg_total_loss = total_loss / total_samples
    avg_gen_loss = gen_loss_total / total_samples
    avg_judge_loss = judge_loss_total / total_samples
    avg_action_loss = action_loss_total / total_samples
    avg_judge_acc = judge_acc_total / total_samples

    return {
        "total_loss": avg_total_loss,
        "gen_loss": avg_gen_loss,
        "judge_loss": avg_judge_loss,
        "action_loss": avg_action_loss,
        "judge_acc": avg_judge_acc
    }


# ======================== 7. 主训练流程（含验证） ========================
def main():
    # 1. 初始化模型
    model_dict = init_models()

    # 2. 构建数据集（训练+验证）
    total_samples = 1000
    val_samples = int(total_samples * config.val_split)
    train_samples = total_samples - val_samples

    train_dataset = PretrainDataset(num_samples=train_samples, is_train=True)
    val_dataset = PretrainDataset(num_samples=val_samples, is_train=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=0
    )

    # 3. 优化器
    all_params = []
    for model in model_dict.values():
        all_params.extend(model.parameters())

    optimizer = optim.AdamW(
        all_params,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 4. 训练+验证循环
    best_val_acc = 0.0  # 保存最佳验证准确率
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        start_time = time.time()

        # 训练
        train_metrics = train_one_epoch(epoch, model_dict, train_dataloader, optimizer)

        # 验证
        val_metrics = validate_one_epoch(epoch, model_dict, val_dataloader)

        # 计算耗时
        epoch_time = time.time() - start_time

        # 打印Epoch汇总
        print("\n" + "=" * 80)
        print(f"Epoch [{epoch + 1}/{config.epochs}] | Time: {epoch_time:.2f}s")
        print("-" * 40 + " 训练集 " + "-" * 40)
        print(f"Total Loss: {train_metrics['total_loss']:.4f} | "
              f"Gen Loss: {train_metrics['gen_loss']:.4f} | "
              f"Judge Loss: {train_metrics['judge_loss']:.4f} | "
              f"Action Loss: {train_metrics['action_loss']:.4f} | "
              f"Judge Acc: {train_metrics['judge_acc']:.2%}")
        print("-" * 40 + " 验证集 " + "-" * 40)
        print(f"Total Loss: {val_metrics['total_loss']:.4f} | "
              f"Gen Loss: {val_metrics['gen_loss']:.4f} | "
              f"Judge Loss: {val_metrics['judge_loss']:.4f} | "
              f"Action Loss: {val_metrics['action_loss']:.4f} | "
              f"Judge Acc: {val_metrics['judge_acc']:.2%}")
        print("=" * 80 + "\n")

        # 保存最佳模型（基于验证准确率）
        if val_metrics["judge_acc"] > best_val_acc:
            best_val_acc = val_metrics["judge_acc"]
            best_val_loss = val_metrics["total_loss"]
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_dict": {k: v.state_dict() for k, v in model_dict.items()},
                "optimizer_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss
            }, "./checkpoints/best_model.pth")
            print(f"保存最佳模型！当前最佳验证Judge Acc: {best_val_acc:.2%} | Val Loss: {best_val_loss:.4f}\n")

        # 定期保存模型
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_dict": {k: v.state_dict() for k, v in model_dict.items()},
                "optimizer_dict": optimizer.state_dict(),
                "val_acc": val_metrics["judge_acc"],
                "val_loss": val_metrics["total_loss"]
            }, f"./checkpoints/epoch_{epoch + 1}.pth")
            print(f"保存Epoch {epoch + 1} 模型\n")

    print(f"训练完成！")
    print(f"最佳验证Judge准确率：{best_val_acc:.2%} | 最佳验证损失：{best_val_loss:.4f}")
    print(f"最佳模型路径：./checkpoints/best_model.pth")


# ======================== 8. 入口函数 ========================
if __name__ == "__main__":
    main()
