import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 路径配置（确保能导入Models.py）
pwd = os.path.abspath(os.path.dirname(__file__))
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
    num_candidates = 5  # 生成的候选动作数量（仅用这些动作训练Judge Model）

    # 模型配置
    embed_dim_gen = 128  # 基础嵌入维度
    judge_dim = 32  # 判断模型输出维度（核心：解决32/256维度不匹配）
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


config = Config()


# ======================== 2. 模拟数据集（替换为你的真实数据集） ========================
class PretrainDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=30, motor_dim=2):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.motor_dim = motor_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟输入：(seq, num_cameras=2, 3, H=64, W=64)
        images = torch.randn(self.seq_length, 2, 3, 64, 64)
        # 模拟电机动作：(seq, motor_dim)
        motor_data = torch.randn(self.seq_length, self.motor_dim)
        # 模拟未来图像（用于判断模型）：(seq, 2, 3, 64, 64)
        future_images = torch.randn(self.seq_length, 2, 3, 64, 64)
        # 模拟真实电机信号（仅用于ActionExtract损失，非Judge Model）
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

    # 判断模型（核心：指定judge_dim=32）
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

    # 动作提取模型（输入维度=judge_dim=32）
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


# ======================== 4. 训练函数（核心修改：Judge Model 标签逻辑） ========================
def train_one_epoch(epoch, model_dict, dataloader, optimizer):
    """单轮训练（Judge Model 仅基于生成的候选动作训练）"""
    # 模型设为训练模式
    for model in model_dict.values():
        model.train()

    total_loss = 0.0
    gen_loss_total = 0.0
    judge_loss_total = 0.0
    action_loss_total = 0.0

    for batch_idx, batch_data in enumerate(dataloader):
        # 1. 数据预处理（移到设备）
        images = batch_data["images"].to(config.device)  # (batch, seq, 2, 3, 64, 64)
        motor_data = batch_data["motor_data"].to(config.device)  # (batch, seq, 2)
        future_images = batch_data["future_images"].to(config.device)  # (batch, seq, 2, 3, 64, 64)
        real_motor_signal = batch_data["real_motor_signal"].to(config.device)  # (batch, 2)

        batch_size = images.shape[0]
        optimizer.zero_grad()

        # 2. 基础特征提取
        image_embedded = model_dict["image_embed"](images)  # (batch, seq, 256)
        motor_embedded = model_dict["motor_embed"](motor_data)  # (batch, seq, 128)
        future_image_embedded = model_dict["image_embed"](future_images)  # (batch, seq, 256)

        # 3. 生成候选动作（仅生成num_candidates个，无真实动作）
        gen_outputs = model_dict["candidate_generator"](
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=config.num_candidates,
            temperature=config.temperature
        )

        # 4. 构建候选动作的电机特征列表（仅生成的动作，无真实动作）
        candidate_emb_list = []
        for candidate in gen_outputs["candidates"]:
            # candidate: (batch, 1, 2) → 嵌入为 (batch, 1, 128)
            candidate_emb = model_dict["motor_embed"](candidate)
            candidate_emb = candidate_emb[:, -1, :]  # (batch, 128)
            candidate_emb_list.append(candidate_emb)

        # 5. Judge Model 前向（仅基于生成的候选动作）
        # 5.1 图像特征：256维 → 32维
        judge_img_feat = model_dict["judge_image"](future_image_embedded)  # (batch, 32)

        # 5.2 电机特征：128维 → 32维
        judge_driver_feats = []
        for driver_emb in candidate_emb_list:
            driver_feat_32d = model_dict["judge_driver"](driver_emb)  # (batch, 32)
            judge_driver_feats.append(driver_feat_32d)

        # 5.3 计算匹配分数：(batch, num_candidates)
        match_scores = model_dict["judge_total"](judge_img_feat, judge_driver_feats)

        # 6. 生成 Judge Model 标签（核心修改）
        # 规则：匹配分数最大的候选标1，其余标0
        # 6.1 找到每个样本的最大分数索引
        max_indices = torch.argmax(match_scores, dim=1)  # (batch,)

        # 6.2 构建one-hot标签（1表示最优候选，0表示其他）
        judge_labels = torch.zeros_like(match_scores)  # (batch, num_candidates)
        judge_labels.scatter_(1, max_indices.unsqueeze(1), 1.0)  # 最大索引位置设为1

        # 7. ActionExtract 前向
        motor_pred = model_dict["action_extract"](judge_img_feat)  # (batch, 2)

        # 8. 损失计算
        # 8.1 生成器损失：预测均值 vs 生成的最优动作（替代真实动作）
        # 取最优候选动作作为生成器的监督信号
        best_candidate = gen_outputs["candidates"][max_indices[0]]  # 批量第一个样本的最优候选
        gen_loss = F.mse_loss(gen_outputs["mean"], best_candidate.squeeze(1))

        # 8.2 Judge Model 损失：匹配分数 vs one-hot标签（MSE）
        judge_loss = F.mse_loss(match_scores, judge_labels)

        # 8.3 ActionExtract 损失（可选：仍用真实动作，或用最优候选）
        action_loss = F.mse_loss(motor_pred, real_motor_signal)

        # 总损失
        total_batch_loss = gen_loss + judge_loss + action_loss

        # 9. 反向传播
        total_batch_loss.backward()
        optimizer.step()

        # 10. 损失统计
        total_loss += total_batch_loss.item()
        gen_loss_total += gen_loss.item()
        judge_loss_total += judge_loss.item()
        action_loss_total += action_loss.item()

        # 打印批次信息
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{config.epochs}] | Batch [{batch_idx + 1}/{len(dataloader)}] | "
                  f"Total Loss: {total_batch_loss.item():.4f} | "
                  f"Gen Loss: {gen_loss.item():.4f} | "
                  f"Judge Loss: {judge_loss.item():.4f} | "
                  f"Action Loss: {action_loss.item():.4f}")

    # 计算平均损失
    avg_total_loss = total_loss / len(dataloader)
    avg_gen_loss = gen_loss_total / len(dataloader)
    avg_judge_loss = judge_loss_total / len(dataloader)
    avg_action_loss = action_loss_total / len(dataloader)

    return avg_total_loss, avg_gen_loss, avg_judge_loss, avg_action_loss


# ======================== 5. 主训练流程 ========================
def main():
    # 1. 初始化模型
    model_dict = init_models()

    # 2. 构建数据集
    train_dataset = PretrainDataset(num_samples=1000)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # 根据硬件调整
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

    # 4. 训练循环
    best_loss = float("inf")
    for epoch in range(config.epochs):
        start_time = time.time()

        # 单轮训练
        train_total, train_gen, train_judge, train_action = train_one_epoch(
            epoch, model_dict, train_dataloader, optimizer
        )

        # 打印epoch信息
        epoch_time = time.time() - start_time
        print(f"\nEpoch [{epoch + 1}/{config.epochs}] | Time: {epoch_time:.2f}s | "
              f"Avg Total Loss: {train_total:.4f} | "
              f"Avg Gen Loss: {train_gen:.4f} | "
              f"Avg Judge Loss: {train_judge:.4f} | "
              f"Avg Action Loss: {train_action:.4f}")

        # 保存最佳模型
        if train_total < best_loss:
            best_loss = train_total
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_dict": {k: v.state_dict() for k, v in model_dict.items()},
                "optimizer_dict": optimizer.state_dict(),
                "best_loss": best_loss
            }, "./checkpoints/best_model.pth")
            print(f"保存最佳模型，当前最佳损失：{best_loss:.4f}")

        # 定期保存
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_dict": {k: v.state_dict() for k, v in model_dict.items()},
                "optimizer_dict": optimizer.state_dict()
            }, f"./checkpoints/epoch_{epoch + 1}.pth")
            print(f"保存Epoch {epoch + 1} 模型")

    print("\n训练完成！最佳模型已保存至 ./checkpoints/best_model.pth")


# ======================== 6. 入口函数 ========================
if __name__ == "__main__":
    main()
