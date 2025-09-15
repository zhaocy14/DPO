# -*- coding: utf-8 -*-
import os
import sys
# 路径配置（确保导入正常）
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataModule.DataModule import CombinedDataset
from Model.Models import (
    ImageEmbedding,
    MotorEmbedding,
    EncoderOnlyCandidateGenerator,
    SimilarityModelImage,
    SimilarityModelDriver,
    PositionalEncoding  # 确保导入模型依赖组件
)

# ---------------------- 1. 基础配置与设备初始化 ----------------------
# 设备设置（与训练保持一致）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[初始化] 使用设备: {device}")

# 核心路径配置（需与训练脚本的config对应）
BEST_MODEL_PATH = "./saved_models/best_model"  # 训练好的最佳模型路径
SIMILARITY_SAVE_PATH = "./layer_similarity/best_model_layer_similarity.npy"  # 相似度结果保存路径
os.makedirs(os.path.dirname(SIMILARITY_SAVE_PATH), exist_ok=True)  # 自动创建保存目录


# ---------------------- 2. 关键工具函数 ----------------------
def calculate_layer_cosine_similarity(layer_outputs):
    """
    计算Transformer各层输出的余弦相似度矩阵
    :param layer_outputs: 各层输出列表，形状为 [num_layers, batch, seq_len, dim]
    :return: 相似度矩阵 (num_layers, num_layers)，numpy格式
    """
    num_layers = len(layer_outputs)
    # 初始化相似度矩阵（num_layers x num_layers）
    similarity_matrix = torch.zeros((num_layers, num_layers), device=device)

    for i in range(num_layers):
        # 展平当前层特征：[batch, seq_len, dim] → [batch, seq_len * dim]（消除时序维度，计算整体相似度）
        layer_i = layer_outputs[i].view(layer_outputs[i].shape[0], -1)
        layer_i_norm = F.normalize(layer_i, dim=1)  # L2归一化，避免尺度影响

        for j in range(num_layers):
            if i == j:
                similarity_matrix[i, j] = 1.0  # 自身相似度为1
                continue

            # 展平对比层特征并归一化
            layer_j = layer_outputs[j].view(layer_outputs[j].shape[0], -1)
            layer_j_norm = F.normalize(layer_j, dim=1)

            # 计算批次内平均余弦相似度（避免单样本波动）
            batch_sim = F.cosine_similarity(layer_i_norm, layer_j_norm, dim=1).mean()
            similarity_matrix[i, j] = batch_sim

    # 转换为numpy格式并返回
    return similarity_matrix.cpu().numpy()


def load_best_model_and_config(model_path):
    """
    加载最佳模型及其训练时的配置（从checkpoint中读取，避免手动配置错误）
    :param model_path: best_model路径
    :return: 加载好的模型（image_embed, motor_embed, candidate_generator）、训练配置config
    """
    # 加载checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"[模型加载] 成功读取checkpoint：{model_path}")
    except Exception as e:
        raise RuntimeError(f"[模型加载失败] 无法读取checkpoint：{str(e)}") from e

    # 从checkpoint中获取训练时的配置（关键！确保模型初始化参数与训练一致）
    config = checkpoint["config"]
    print(f"[配置读取] 从checkpoint获取训练配置：embed_dim_gen={config['embed_dim_gen']}, num_layers_gen={config['num_layers_gen']}")

    # 初始化模型（参数与训练时完全一致）
    # 1. 图像嵌入模型
    image_embed = ImageEmbedding(
        embed_dim=config["embed_dim_gen"],
        num_layers=3,  # 训练时固定为3层
        is_resnet=False  # 训练时关闭ResNet
    ).to(device)

    # 2. 电机嵌入模型
    motor_embed = MotorEmbedding(
        motor_dim=config["motor_dim"],  # 训练时为2（两个电机）
        embed_dim=config["embed_dim_gen"]
    ).to(device)

    # 3. 核心：候选生成器（含Transformer编码器）
    candidate_generator = EncoderOnlyCandidateGenerator(
        embed_dim=config["embed_dim_gen"],
        nhead=config["nhead_gen"],  # 训练时为8
        num_layers=config["num_layers_gen"],  # 训练时为16层
        motor_dim=config["motor_dim"],
        max_seq_length=config["gen_seq_len"]  # 训练时为30
    ).to(device)

    # 从checkpoint加载模型权重（对应训练脚本的"model_states"键）
    image_embed.load_state_dict(checkpoint["model_states"]["image_embed"])
    motor_embed.load_state_dict(checkpoint["model_states"]["motor_embed"])
    candidate_generator.load_state_dict(checkpoint["model_states"]["candidate_generator"])

    # 设置模型为评估模式（关闭Dropout等训练特有的层）
    image_embed.eval()
    motor_embed.eval()
    candidate_generator.eval()

    print("[模型加载完成] 所有模型已设置为eval模式")
    return image_embed, motor_embed, candidate_generator, config


def get_sample_input(image_embed, motor_embed, config):
    """
    从训练数据集加载一个样本批次（用于模型前向传播获取层输出）
    :param image_embed: 图像嵌入模型
    :param motor_embed: 电机嵌入模型
    :param config: 训练配置
    :return: 预处理后的嵌入特征（image_embedded, motor_embedded）
    """
    # 加载数据集（与训练脚本的CombinedDataset配置一致）
    data_root = config["data_root_dirs"]
    data_dir_list = [
        os.path.join(data_root, file)
        for file in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, file)) and "2025" in file
    ]
    if not data_dir_list:
        raise FileNotFoundError(f"[数据加载失败] {data_root}下无含'2025'的子目录")

    # 初始化验证集（仅需一个样本批次，无需训练集）
    all_dataset = CombinedDataset(
        dir_list=data_dir_list,
        frame_len=config["gen_seq_len"],  # 观测序列长度30
        predict_len=config["sim_seq_len"],  # 预测序列长度30
        show=False  # 关闭数据集打印信息
    )
    val_loader = DataLoader(
        dataset=all_dataset.val_dataset,
        batch_size=1,  # 仅需1个batch（减少计算量）
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print(f"[数据加载] 成功获取验证集，样本数：{len(all_dataset.val_dataset)}")

    # 获取一个样本批次并预处理
    sample_batch = next(iter(val_loader))
    imgs1, imgs2, driver, _, _, _ = sample_batch  # 仅需图像和电机数据

    # 转换为模型输入格式并移动到设备
    images = torch.stack([imgs1, imgs2], dim=2).to(device)  # (1, 30, 2, 3, H, W)
    driver = driver.to(device)  # (1, 30, 2)

    # 生成嵌入特征（无梯度计算）
    with torch.no_grad():
        image_embedded = image_embed(images)  # (1, 30, 2*128)
        motor_embedded = motor_embed(driver)  # (1, 30, 128)

    print(f"[输入准备完成] 图像嵌入形状：{image_embedded.shape}，电机嵌入形状：{motor_embedded.shape}")
    return image_embedded, motor_embedded


# ---------------------- 3. 主流程：加载模型 → 计算相似度 → 保存结果 ----------------------
def main():
    print("\n" + "=" * 70)
    print("                  读取best_model并计算Transformer层间余弦相似度")
    print("=" * 70)

    # 1. 加载最佳模型和训练配置
    try:
        image_embed, motor_embed, candidate_generator, config = load_best_model_and_config(BEST_MODEL_PATH)
    except Exception as e:
        print(f"[初始化失败] {str(e)}")
        return

    # 2. 获取模型输入（嵌入特征）
    try:
        image_embedded, motor_embedded = get_sample_input(image_embed, motor_embed, config)
    except Exception as e:
        print(f"[输入准备失败] {str(e)}")
        return

    # 3. 前向传播获取Transformer各层输出
    print("\n[前向传播] 正在获取Encoder各层输出...")
    with torch.no_grad():  # 关闭梯度，加速计算
        generator_output = candidate_generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=1,  # 仅需1个候选（无需生成多个，减少计算）
            temperature=1.0
        )
        # 提取各层输出（关键！训练时的模型forward会返回该键）
        encoder_layer_outputs = generator_output["encoder_layer_outputs"]
        num_layers = len(encoder_layer_outputs)
    print(f"[层输出获取完成] 共{num_layers}个Transformer Encoder层，每层输出形状：{encoder_layer_outputs[0].shape}")

    # 4. 计算层间余弦相似度
    print("\n[相似度计算] 正在计算层间余弦相似度矩阵...")
    similarity_matrix = calculate_layer_cosine_similarity(encoder_layer_outputs)

    # 5. 保存结果
    np.save(SIMILARITY_SAVE_PATH, similarity_matrix)
    print(f"\n[结果保存] 相似度矩阵已保存至：{SIMILARITY_SAVE_PATH}")
    print(f"[矩阵信息] 形状：{similarity_matrix.shape}（{num_layers}层 × {num_layers}层），数值范围：[{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

    # 6. 打印前5层预览（方便快速查看）
    print("\n[结果预览] 前5层的余弦相似度矩阵：")
    print(np.round(similarity_matrix[:5, :5], 4))  # 保留4位小数

    print("\n" + "=" * 70)
    print("                          计算流程完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
