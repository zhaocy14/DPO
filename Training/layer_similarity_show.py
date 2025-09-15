# -*- coding: utf-8 -*-
import os
import sys
# 路径配置（确保导入正常）
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataModule.DataModule import CombinedDataset
from Model.Models import (
    ImageEmbedding,
    MotorEmbedding,
    SimilarityModelImage,
    SimilarityModelDriver,
    PositionalEncoding  # 仅导入原模型中的基础组件
)

# ---------------------- 1. 基础配置与设备初始化 ----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[初始化] 使用设备: {device}")

# 路径配置（与你的普通训练脚本对应）
BEST_MODEL_PATH = "./saved_models/best_model"  # 你的best_model路径
SIMILARITY_SAVE_PATH = "./layer_similarity/best_model_layer_similarity.npy"
os.makedirs(os.path.dirname(SIMILARITY_SAVE_PATH), exist_ok=True)


# ---------------------- 2. 关键修复：重新定义模型类（新增encoder_layer_outputs返回） ----------------------
# 注意：以下模型结构与你的普通训练脚本完全一致，仅新增返回各层输出
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
        layer_outputs = []  # 保存每一层的输出
        current = src
        for i, layer in enumerate(self.layers):
            if self.layer_trainable[i]:
                current = layer(current)
            else:
                with torch.no_grad():
                    current = layer(current)
            layer_outputs.append(current)  # 记录当前层输出
        return layer_outputs  # 返回所有层输出（原训练中未用到，但此处必须返回）


class EncoderOnlyCandidateGenerator(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, motor_dim=2, max_seq_length=100):
        super().__init__()
        self.d_model = embed_dim * 3  # motor_embed(1*dim) + image_embed(2*dim)
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder = TransformerEncoderModel(embed_dim, nhead, num_layers)  # 调用上面的编码器

        # 单组动作输出层（与普通训练脚本一致）
        self.fc_mean = nn.Linear(self.d_model, motor_dim)
        self.fc_logvar = nn.Linear(self.d_model, motor_dim)

    def forward(self, image_embedded, motor_embedded, num_candidates=5, temperature=0.5):
        # 1. 融合特征（与原训练一致）
        combined = torch.cat([motor_embedded, image_embedded], dim=-1)
        combined = self.positional_encoding(combined)

        # 2. 获取编码器所有层输出（关键修复：原训练中只取最后一层，此处保留所有层）
        encoder_layer_outputs = self.encoder(combined)  # [num_layers, batch, seq, 3*embed_dim]
        final_encoder_out = encoder_layer_outputs[-1]  # 仍用最后一层计算动作（与原训练一致）
        global_feat = final_encoder_out.mean(dim=1)

        # 3. 计算动作分布（与原训练一致）
        mean = self.fc_mean(global_feat)
        logvar = self.fc_logvar(global_feat)
        logvar = torch.clamp(logvar, min=-5, max=5)
        std = torch.exp(0.5 * logvar) * temperature

        # 4. 生成候选动作（与原训练一致）
        candidates = []
        for _ in range(num_candidates):
            eps = torch.randn_like(mean)
            sample = mean + std * eps
            sample = torch.tanh(sample)
            candidates.append(sample.unsqueeze(1))  # (batch, 1, motor_dim)

        # 【关键新增】返回encoder_layer_outputs，用于计算层相似度
        return {
            'candidates': candidates,
            'mean': mean,
            'std': std,
            'encoder_layer_outputs': encoder_layer_outputs  # 新增此键，解决KeyError
        }


# ---------------------- 3. 工具函数（不变） ----------------------
def calculate_layer_cosine_similarity(layer_outputs):
    """计算层间余弦相似度矩阵"""
    num_layers = len(layer_outputs)
    similarity_matrix = torch.zeros((num_layers, num_layers), device=device)

    for i in range(num_layers):
        # 展平特征：[batch, seq, dim] → [batch, seq*dim]
        layer_i = layer_outputs[i].view(layer_outputs[i].shape[0], -1)
        layer_i_norm = F.normalize(layer_i, dim=1)

        for j in range(num_layers):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue
            layer_j = layer_outputs[j].view(layer_outputs[j].shape[0], -1)
            layer_j_norm = F.normalize(layer_j, dim=1)
            batch_sim = F.cosine_similarity(layer_i_norm, layer_j_norm, dim=1).mean()
            similarity_matrix[i, j] = batch_sim

    return similarity_matrix.cpu().numpy()


def load_best_model_and_config(model_path):
    """加载best_model和训练配置，使用上面重新定义的模型类"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint["config"]
        print(f"[配置读取] 嵌入维度：{config['embed_dim_gen']}，Transformer层数：{config['num_layers_gen']}")
    except Exception as e:
        raise RuntimeError(f"[Checkpoint加载失败] {str(e)}") from e

    # 初始化模型（使用重新定义的类，确保能返回encoder_layer_outputs）
    image_embed = ImageEmbedding(
        embed_dim=config["embed_dim_gen"],
        num_layers=3,
        is_resnet=False
    ).to(device)

    motor_embed = MotorEmbedding(
        motor_dim=config["motor_dim"],
        embed_dim=config["embed_dim_gen"]
    ).to(device)

    # 关键：使用重新定义的EncoderOnlyCandidateGenerator
    candidate_generator = EncoderOnlyCandidateGenerator(
        embed_dim=config["embed_dim_gen"],
        nhead=config["nhead_gen"],
        num_layers=config["num_layers_gen"],
        motor_dim=config["motor_dim"],
        max_seq_length=config["gen_seq_len"]
    ).to(device)

    # 加载权重（与原训练模型结构完全匹配，可正常加载）
    image_embed.load_state_dict(checkpoint["model_states"]["image_embed"])
    motor_embed.load_state_dict(checkpoint["model_states"]["motor_embed"])
    candidate_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])

    # 设为评估模式
    image_embed.eval()
    motor_embed.eval()
    candidate_generator.eval()

    print("[模型加载完成] 已使用支持层输出返回的模型类")
    return image_embed, motor_embed, candidate_generator, config


def get_sample_input(image_embed, motor_embed, config):
    """获取模型输入（与原训练一致）"""
    data_root = config["data_root_dirs"]
    data_dir_list = [
        os.path.join(data_root, f) for f in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, f)) and "2025" in f
    ]
    if not data_dir_list:
        raise FileNotFoundError(f"[数据缺失] {data_root}下无含'2025'的子目录")

    # 加载验证集（仅需1个batch）
    all_dataset = CombinedDataset(
        dir_list=data_dir_list,
        frame_len=config["gen_seq_len"],
        predict_len=config["sim_seq_len"],
        show=False
    )
    val_loader = DataLoader(
        all_dataset.val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 预处理样本
    sample_batch = next(iter(val_loader))
    imgs1, imgs2, driver, _, _, _ = sample_batch
    images = torch.stack([imgs1, imgs2], dim=2).to(device)
    driver = driver.to(device)

    # 生成嵌入特征
    with torch.no_grad():
        image_embedded = image_embed(images)  # (1, 30, 2*128)
        motor_embedded = motor_embed(driver)  # (1, 30, 128)

    print(f"[输入准备完成] 图像嵌入：{image_embedded.shape}，电机嵌入：{motor_embedded.shape}")
    return image_embedded, motor_embedded


# ---------------------- 4. 主流程 ----------------------
def main():
    print("\n" + "=" * 70)
    print("                  修复KeyError：计算Transformer层间余弦相似度")
    print("=" * 70)

    # 1. 加载模型（使用修复后的类）
    try:
        image_embed, motor_embed, candidate_generator, config = load_best_model_and_config(BEST_MODEL_PATH)
    except Exception as e:
        print(f"[初始化失败] {str(e)}")
        return

    # 2. 获取输入数据
    try:
        image_embedded, motor_embedded = get_sample_input(image_embed, motor_embed, config)
    except Exception as e:
        print(f"[输入失败] {str(e)}")
        return

    # 3. 前向传播获取层输出（此时已能拿到encoder_layer_outputs）
    print("\n[前向传播] 正在获取各层Transformer输出...")
    with torch.no_grad():
        generator_output = candidate_generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=1,
            temperature=1.0
        )
        # 现在可以正常获取该键，无KeyError
        encoder_layer_outputs = generator_output["encoder_layer_outputs"]
        num_layers = len(encoder_layer_outputs)
    print(f"[层输出获取成功] 共{num_layers}层，每层形状：{encoder_layer_outputs[0].shape}")

    # 4. 计算相似度
    print("\n[相似度计算] 正在计算层间矩阵...")
    similarity_matrix = calculate_layer_cosine_similarity(encoder_layer_outputs)

    # 5. 保存结果
    np.save(SIMILARITY_SAVE_PATH, similarity_matrix)
    print(f"\n[保存成功] 矩阵路径：{SIMILARITY_SAVE_PATH}")
    print(f"[矩阵信息] 形状：{similarity_matrix.shape}，数值范围：[{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

    # 6. 预览前5层
    print("\n[前5层相似度预览]")
    print(np.round(similarity_matrix[:5, :5], 4))

    print("\n" + "=" * 70)
    print("                          流程完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

