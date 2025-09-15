# -*- coding: utf-8 -*-
import os
import sys
# 路径配置
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from DataModule.DataModule import CombinedDataset
from Model.Models import (ImageEmbedding, MotorEmbedding,
                          EncoderOnlyCandidateGenerator,
                          SimilarityModelImage, SimilarityModelDriver,
                          PositionalEncoding)

# 设备初始化
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"[初始化] 使用设备: {device}")

# 配置参数（与训练时保持一致）
CONFIG = {
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,
    "gen_seq_len": 30,
    "sim_seq_len": 30,
    "data_root_dirs": '/data/cyzhao/collector_cydpo/dpo_data',
    "trained_model_path": "./saved_models/key_layers_dpo_final_best_model",  # 训练好的模型路径
    "similarity_save_path": "./layer_similarity_matrix.npy",  # 相似度矩阵保存路径
    "batch_size": 1  # 用于生成特征的批次大小
}

# 创建保存目录
os.makedirs(os.path.dirname(CONFIG["similarity_save_path"]), exist_ok=True)


# 定义Transformer编码器模型（与训练时一致）
class TransformerEncoderModel(nn.Module):
    def __init__(self, embed_dim=64, nhead=8, num_layers=16):
        super(TransformerEncoderModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim * 3, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_trainable = nn.Parameter(torch.ones(num_layers, dtype=torch.bool), requires_grad=False)

    def set_layer_trainable(self, layer_idx, trainable):
        if 0 <= layer_idx < len(self.layers):
            self.layer_trainable[layer_idx] = trainable

    def forward(self, src):
        layer_outputs = []
        current = src
        for i, layer in enumerate(self.layers):
            if self.layer_trainable[i]:
                current = layer(current)
            else:
                with torch.no_grad():
                    current = layer(current)
            layer_outputs.append(current)
        return layer_outputs


# 定义候选生成器模型（与训练时一致）
class EncoderOnlyCandidateGenerator(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, motor_dim=2, max_seq_length=100):
        super().__init__()
        self.d_model = embed_dim * 3
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder = TransformerEncoderModel(embed_dim, nhead, num_layers)
        self.fc_mean = nn.Linear(self.d_model, motor_dim)
        self.fc_logvar = nn.Linear(self.d_model, motor_dim)

    def forward(self, image_embedded, motor_embedded, num_candidates=5, temperature=0.5):
        combined = torch.cat([motor_embedded, image_embedded], dim=-1)
        combined = self.positional_encoding(combined)
        encoder_layer_outputs = self.encoder(combined)
        final_encoder_out = encoder_layer_outputs[-1]
        global_feat = final_encoder_out.mean(dim=1)

        mean = self.fc_mean(global_feat)
        logvar = self.fc_logvar(global_feat)
        logvar = torch.clamp(logvar, min=-5, max=5)
        std = torch.exp(0.5 * logvar) * temperature

        candidates = []
        for _ in range(num_candidates):
            eps = torch.randn_like(mean)
            sample = mean + std * eps
            sample = torch.tanh(sample)
            candidates.append(sample.unsqueeze(1))

        return {
            'candidates': candidates,
            'mean': mean,
            'std': std,
            'encoder_layer_outputs': encoder_layer_outputs
        }


# 加载训练好的模型
def load_trained_model(model_path):
    # 初始化模型
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

    # 加载模型权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        image_embed.load_state_dict(
            checkpoint["model_states"]["image_embed"] if "model_states" in checkpoint else checkpoint[
                "policy_generator_state_dict"])
        motor_embed.load_state_dict(
            checkpoint["model_states"]["motor_embed"] if "model_states" in checkpoint else checkpoint[
                "policy_generator_state_dict"])
        policy_generator.load_state_dict(checkpoint["policy_generator_state_dict"])
        print(f"[模型加载] 成功加载训练好的模型：{model_path}")
    except Exception as e:
        raise RuntimeError(f"[模型加载失败] {str(e)}") from e

    # 设置为评估模式
    image_embed.eval()
    motor_embed.eval()
    policy_generator.eval()

    return image_embed, motor_embed, policy_generator


# 加载数据用于生成输入
def load_sample_data():
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
        val_dataset = all_dataset.val_dataset
        print(f"[数据加载] 验证集：{len(val_dataset)}样本")
    except Exception as e:
        raise RuntimeError(f"[数据集加载失败] {str(e)}") from e

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    return val_loader


# 计算层间余弦相似度
def calculate_layer_similarity(layer_outputs):
    num_layers = len(layer_outputs)
    similarity_matrix = torch.zeros(num_layers, num_layers, device=device)

    for i in range(num_layers):
        # 展平特征维度：[batch, seq, dim] -> [batch, seq*dim]
        output_i = layer_outputs[i].view(layer_outputs[i].shape[0], -1)
        output_i_norm = F.normalize(output_i, dim=1)

        for j in range(num_layers):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue

            output_j = layer_outputs[j].view(layer_outputs[j].shape[0], -1)
            output_j_norm = F.normalize(output_j, dim=1)

            # 计算批次平均余弦相似度
            sim = F.cosine_similarity(output_i_norm, output_j_norm, dim=1).mean()
            similarity_matrix[i, j] = sim

    return similarity_matrix.cpu().numpy()


# 主函数
def main():
    print("\n" + "=" * 60)
    print("          计算EncoderOnlyCandidateGenerator层间余弦相似度")
    print("=" * 60)

    # 1. 加载模型
    try:
        image_embed, motor_embed, policy_generator = load_trained_model(CONFIG["trained_model_path"])
    except Exception as e:
        print(f"[模型加载失败] {str(e)}")
        return

    # 2. 加载样本数据
    try:
        val_loader = load_sample_data()
        # 获取一个样本批次
        sample_batch = next(iter(val_loader))
    except Exception as e:
        print(f"[数据加载失败] {str(e)}")
        return

    # 3. 准备输入数据
    imgs1, imgs2, driver, _, _, _ = sample_batch
    images = torch.stack([imgs1, imgs2], dim=2).to(device)
    driver = driver.to(device)

    # 4. 获取嵌入特征
    with torch.no_grad():
        image_embedded = image_embed(images)
        motor_embedded = motor_embed(driver)

    # 5. 获取模型输出和各层特征
    with torch.no_grad():
        model_output = policy_generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=1
        )
        layer_outputs = model_output["encoder_layer_outputs"]
        num_layers = len(layer_outputs)
        print(f"[层信息] 共检测到 {num_layers} 个Encoder层")

    # 6. 计算层间相似度
    print("[计算中] 正在计算层间余弦相似度...")
    similarity_matrix = calculate_layer_similarity(layer_outputs)

    # 7. 保存结果
    np.save(CONFIG["similarity_save_path"], similarity_matrix)
    print(f"[保存成功] 相似度矩阵已保存至：{CONFIG['similarity_save_path']}")
    print(f"[矩阵形状] {similarity_matrix.shape} (层数量 x 层数量)")

    # 8. 打印部分结果（前5x5）
    print("\n[部分结果] 前5层的相似度矩阵：")
    print(similarity_matrix[:5, :5])


if __name__ == "__main__":
    import torch.nn.functional as F
    import torch.nn as nn

    main()
