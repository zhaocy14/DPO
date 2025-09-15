# -*- coding: utf-8 -*-
import os
import sys
# 路径配置
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import time
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataModule.DataModule import CombinedDataset
from Model.Models import (ImageEmbedding, MotorEmbedding,
                          EncoderOnlyCandidateGenerator,
                          SimilarityModelImage, SimilarityModelDriver,
                          PositionalEncoding)  # 确保导入PositionalEncoding
from tqdm import tqdm



# 设备初始化
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"[初始化] 使用设备: {device}")

# ---------------------- 核心配置参数（新增关键层识别参数） ----------------------
CONFIG = {
    # 验证控制
    "max_val_batches": 50,
    "val_batch_size": 8,

    # 相似度加权系数
    "alpha": 0.9,
    "action_match_tolerance": 1e-4,

    # 关键层识别参数
    "P": 3,  # 倒数P层特殊处理
    "Q": 5,  # 每次训练随机选择Q个关键层更新
    "similarity_threshold": 0.8,  # 80%相似度阈值

    "batch_size": 1,
    "epochs": 5,
    "lr": 5e-7,
    "num_candidates": 5,
    "sampling_workers": 2,
    "max_train_samples_per_epoch": 500,
    "dpo_beta": 0.1,
    "repeat_threshold": 0.95,
    "history_cache_size": 10,
    "use_candidates": "candidates1",
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,  # Transformer层数
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
    "dpo_loss_path": "./loss_records/dpo_final_loss.npy",
    "key_layers_record_path": "./loss_records/key_layers_history.npy"  # 记录关键层历史
}

# 创建保存目录
os.makedirs(os.path.dirname(CONFIG["dpo_save_path"]), exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["dpo_loss_path"]), exist_ok=True)


# ---------------------- Transformer相关模型定义（确保与提供的代码一致） ----------------------
class TransformerEncoderModel(nn.Module):
    def __init__(self, embed_dim=64, nhead=8, num_layers=16):
        super(TransformerEncoderModel, self).__init__()
        # 存储每一层的编码器
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim * 3, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        # 为每一层创建训练控制开关，默认都参与训练
        self.layer_trainable = nn.Parameter(torch.ones(num_layers, dtype=torch.bool), requires_grad=False)

    def set_layer_trainable(self, layer_idx, trainable):
        """设置指定层是否参与训练"""
        if 0 <= layer_idx < len(self.layers):
            self.layer_trainable[layer_idx] = trainable

    def forward(self, src):
        """返回每一层的输出结果列表"""
        layer_outputs = []
        current = src

        for i, layer in enumerate(self.layers):
            # 根据开关决定是否让该层参与训练
            if self.layer_trainable[i]:
                # 该层参与训练，正常计算
                current = layer(current)
            else:
                # 该层不参与训练，关闭梯度计算
                with torch.no_grad():
                    current = layer(current)

            # 保存当前层的输出
            layer_outputs.append(current)

        return layer_outputs  # 返回所有层的输出列表，最后一个元素是最终输出


class EncoderOnlyCandidateGenerator(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, motor_dim=2, max_seq_length=100):
        super().__init__()
        self.d_model = embed_dim * 3  # 融合后的特征维度
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder = TransformerEncoderModel(embed_dim, nhead, num_layers)

        # 输出层：为两种数据分别预测分布参数
        # 第一组数据的分布参数
        self.fc_mean1 = nn.Linear(self.d_model, motor_dim)  # 第一组均值
        self.fc_logvar1 = nn.Linear(self.d_model, motor_dim)  # 第一组对数方差

        # 第二组数据的分布参数
        self.fc_mean2 = nn.Linear(self.d_model, motor_dim)  # 第二组均值
        self.fc_logvar2 = nn.Linear(self.d_model, motor_dim)  # 第二组对数方差

    def forward(self, image_embedded, motor_embedded, num_candidates=5, temperature=0.5):
        """
        生成多个1帧动作候选，返回两组数据及其分布参数
        :param image_embedded: 图像嵌入 (batch, seq, 2*embed_dim)
        :param motor_embedded: 历史电机嵌入 (batch, seq, embed_dim)
        :param num_candidates: 候选数量
        :param temperature: 温度参数（控制采样随机性，>0，越小越集中）
        :return: 两组候选动作列表及对应的均值和标准差
        """
        # 1. 融合输入特征
        combined = torch.cat([motor_embedded, image_embedded], dim=-1)  # (batch, seq, 3*embed_dim)
        combined = self.positional_encoding(combined)  # 加位置编码

        # 2. Encoder提取全局特征
        encoder_out = self.encoder(combined)  # 得到每一层的输出列表
        final_out = encoder_out[-1]
        global_feat = final_out.mean(dim=1)  # 取序列均值作为全局特征 (batch, 3*embed_dim)

        # 3. 预测第一组数据的分布参数（均值+标准差）
        mean1 = self.fc_mean1(global_feat)  # (batch, motor_dim)
        logvar1 = self.fc_logvar1(global_feat)  # (batch, motor_dim)
        logvar1 = torch.clamp(logvar1, min=-5, max=5)
        std1 = torch.exp(0.5 * logvar1) * temperature  # 温度调整标准差

        # 预测第二组数据的分布参数（均值+标准差）
        mean2 = self.fc_mean2(global_feat)  # (batch, motor_dim)
        logvar2 = self.fc_logvar2(global_feat)  # (batch, motor_dim)
        logvar2 = torch.clamp(logvar2, min=-5, max=5)
        std2 = torch.exp(0.5 * logvar2) * temperature  # 温度调整标准差

        # 4. 从高斯分布中多次采样，生成候选，并使用tanh限制在[-1, 1]
        candidates1 = []
        candidates2 = []
        for _ in range(num_candidates):
            # 第一组采样
            eps1 = torch.randn_like(mean1)  # 标准正态分布噪声
            sample1 = mean1 + std1 * eps1  # (batch, motor_dim)
            sample1 = torch.tanh(sample1)  # 将输出限制在[-1, 1]之间
            candidates1.append(sample1.unsqueeze(1))  # 增加时间维度 (batch, 1, motor_dim)

            # 第二组采样
            eps2 = torch.randn_like(mean2)  # 标准正态分布噪声
            sample2 = mean2 + std2 * eps2  # (batch, motor_dim)
            sample2 = torch.tanh(sample2)  # 将输出限制在[-1, 1]之间
            candidates2.append(sample2.unsqueeze(1))  # 增加时间维度 (batch, 1, motor_dim)

        # 返回两组候选以及它们的均值和标准差，同时返回所有层的输出用于关键层识别
        return {
            'candidates1': candidates1,
            'mean1': mean1,
            'std1': std1,
            'candidates2': candidates2,
            'mean2': mean2,
            'std2': std2,
            'encoder_layer_outputs': encoder_out  # 新增：返回每一层的输出
        }


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
    def check_batch_size(dataloader, name, expected_size):
        for batch in dataloader:
            imgs1 = batch[0]
            if imgs1.shape[0] != expected_size:
                raise RuntimeError(f"[BatchSize错误] {name}的batch_size={imgs1.shape[0]}，需为{expected_size}")
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

    check_batch_size(train_loader, "训练集", CONFIG["batch_size"])
    check_batch_size(val_loader, "验证集", CONFIG["val_batch_size"])
    print(f"[数据配置] 训练集batch_size={CONFIG['batch_size']} | 验证集batch_size={CONFIG['val_batch_size']}")

    return train_loader, val_loader


# ---------------------- 3. 核心工具函数（新增关键层识别相关函数） ----------------------
def calculate_layer_similarity(layer_outputs):
    """
    计算每一层输出与其他层输出的余弦相似度
    :param layer_outputs: 每一层的输出列表，形状为 [num_layers, batch, seq, dim]
    :return: 相似度矩阵 [num_layers, num_layers]
    """
    num_layers = len(layer_outputs)
    similarity_matrix = torch.zeros(num_layers, num_layers, device=device)

    # 对每一层计算与其他层的相似度
    for i in range(num_layers):
        # 展平特征维度：[batch, seq, dim] -> [batch, seq*dim]
        output_i = layer_outputs[i].view(layer_outputs[i].shape[0], -1)
        # 标准化
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

    return similarity_matrix


def identify_key_layers(policy_generator, sample_batch, image_embed, motor_embed):
    """
    识别Transformer编码器中的关键层
    :param policy_generator: EncoderOnlyCandidateGenerator模型
    :param sample_batch: 用于分析的样本批次
    :param image_embed: 图像嵌入模型
    :param motor_embed: 电机嵌入模型
    :return: 关键层索引列表
    """
    # 解包样本批次
    imgs1, imgs2, driver, _, _, _ = sample_batch
    images = torch.stack([imgs1, imgs2], dim=2).to(device)
    driver = driver.to(device)

    # 获取嵌入特征
    with torch.no_grad():
        image_embedded = image_embed(images)
        motor_embedded = motor_embed(driver)

    # 获取模型输出，包括每一层的编码器输出
    model_output = policy_generator(
        image_embedded=image_embedded,
        motor_embedded=motor_embedded,
        num_candidates=1  # 只需要1个候选即可用于分析
    )
    layer_outputs = model_output["encoder_layer_outputs"]
    num_layers = len(layer_outputs)
    P = CONFIG["P"]

    # 计算层间相似度矩阵
    similarity_matrix = calculate_layer_similarity(layer_outputs)

    # 初始化关键层集合
    key_layers = set()

    # 最后一层直接设为关键层
    key_layers.add(num_layers - 1)

    # 识别关键层
    for i in range(num_layers):
        # 对于倒数P层，特殊处理：与所有后续层比较
        if i >= num_layers - P:
            # 后续层是i+1到最后一层
            next_layers = range(i + 1, num_layers)
            if not next_layers:  # 最后一层已经处理过
                continue

            # 计算与后续层的平均相似度
            avg_sim = torch.mean(similarity_matrix[i, list(next_layers)]).item()

            if avg_sim >= CONFIG["similarity_threshold"]:
                key_layers.add(i)
        else:
            # 对于其他层，与后面P层比较
            next_layers = range(i + 1, min(i + 1 + P, num_layers))
            if not next_layers:
                continue

            # 计算与后面P层的平均相似度
            avg_sim = torch.mean(similarity_matrix[i, list(next_layers)]).item()

            if avg_sim >= CONFIG["similarity_threshold"]:
                key_layers.add(i)

    # 转换为排序后的列表
    key_layers_list = sorted(list(key_layers))
    print(f"[关键层识别] 共识别到 {len(key_layers_list)} 个关键层: {key_layers_list}")
    return key_layers_list


def select_layers_for_training(policy_generator, key_layers):
    """
    从关键层中随机选择Q个层进行训练，其他层不训练
    :param policy_generator: EncoderOnlyCandidateGenerator模型
    :param key_layers: 关键层索引列表
    :return: 被选中进行训练的层索引
    """
    num_layers = len(policy_generator.encoder.layers)
    Q = CONFIG["Q"]

    # 确保关键层数量不小于Q，若小于则全部选中
    if len(key_layers) <= Q:
        selected_layers = key_layers
    else:
        # 随机选择Q个关键层
        selected_indices = np.random.choice(len(key_layers), Q, replace=False)
        selected_layers = [key_layers[i] for i in selected_indices]

    # 先将所有层设置为不可训练
    for i in range(num_layers):
        policy_generator.encoder.set_layer_trainable(i, False)

    # 再将选中的层设置为可训练
    for layer_idx in selected_layers:
        policy_generator.encoder.set_layer_trainable(layer_idx, True)

    print(
        f"[训练层选择] 从 {len(key_layers)} 个关键层中随机选择了 {len(selected_layers)} 个进行训练: {sorted(selected_layers)}")
    return selected_layers


def gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    std = std + eps
    log_prob = -0.5 * torch.log(2 * torch.tensor(np.pi, device=device)) - torch.log(std) - (action - mean) ** 2 / (
                2 * std ** 2)
    return log_prob.sum(dim=-1)


def get_generator_distribution(generator: EncoderOnlyCandidateGenerator,
                               image_embedded: torch.Tensor,
                               motor_embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
                              motor_embed: MotorEmbedding,
                              batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """选择偏好和非偏好动作，返回sim_total用于后续分析"""
    # 1. 维度检查
    assert len(candidates) == CONFIG["num_candidates"], f"候选数={len(candidates)}，需为{CONFIG['num_candidates']}"
    for i, cand in enumerate(candidates):
        assert cand.shape == (batch_size, CONFIG["motor_dim"]), \
            f"候选{i}维度错误：{cand.shape}，需为({batch_size},{CONFIG['motor_dim']})"
    assert future_driver_last.shape == (batch_size, CONFIG["motor_dim"]), \
        f"future_driver_last维度错误：{future_driver_last.shape}"

    # 2. 候选动作嵌入
    candidate_embeddings = []
    for cand in candidates:
        cand_with_seq = cand.unsqueeze(1)  # (B,1,2)
        emb = motor_embed(cand_with_seq)  # (B,1,128)
        candidate_embeddings.append(emb)

    # 3. 计算图像相似度
    sim_img = []
    for emb in candidate_embeddings:
        cand_proj = driver_sim_model(emb).squeeze(1)  # (B,32)
        img_proj_future_squeezed = img_proj_future.squeeze(1)  # (B,32)
        sim = F.cosine_similarity(cand_proj, img_proj_future_squeezed, dim=1)  # (B,)
        sim_img.append(sim)

    # 4. 计算动作相似度
    future_driver_with_seq = future_driver_last.unsqueeze(1)  # (B,1,2)
    future_driver_emb = motor_embed(future_driver_with_seq)  # (B,1,128)
    future_emb_squeezed = future_driver_emb.squeeze(1)  # (B,128)
    future_norm = F.normalize(future_emb_squeezed, dim=1)  # (B,128)

    sim_driver = []
    for emb in candidate_embeddings:
        emb_squeezed = emb.squeeze(1)  # (B,128)
        emb_norm = F.normalize(emb_squeezed, dim=1)  # (B,128)
        sim = F.cosine_similarity(emb_norm, future_norm, dim=1)  # (B,)
        sim_driver.append(sim)

    # 5. 计算总相似度
    alpha = CONFIG["alpha"]
    sim_img_tensor = torch.stack(sim_img).T  # (B, 5)
    sim_driver_tensor = torch.stack(sim_driver).T  # (B, 5)
    sim_total = alpha * sim_img_tensor + (1 - alpha) * sim_driver_tensor  # (B, 5)

    # 6. 选择偏好/非偏好动作
    preferred_idx = sim_total.argmax(dim=1)  # (B,)
    rejected_idx = sim_total.argmin(dim=1)  # (B,)

    # 7. 提取动作
    candidates_tensor = torch.stack(candidates).permute(1, 0, 2)  # (B,5,2)
    preferred = torch.gather(
        candidates_tensor, 1,
        preferred_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, CONFIG["motor_dim"])
    ).squeeze(1)  # (B,2)
    rejected = torch.gather(
        candidates_tensor, 1,
        rejected_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, CONFIG["motor_dim"])
    ).squeeze(1)  # (B,2)

    return preferred, rejected, sim_total


def get_model_highest_prob_action(candidates: list[torch.Tensor],
                                  candidates_tensor: torch.Tensor,
                                  image_embedded: torch.Tensor,
                                  motor_embedded: torch.Tensor,
                                  policy_gen: EncoderOnlyCandidateGenerator) -> torch.Tensor:
    """获取模型最高概率的动作"""
    batch_size = image_embedded.shape[0]
    num_candidates = len(candidates)

    # 获取模型的分布参数
    mean, std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)  # (B,2), (B,2)

    # 计算每个候选动作的概率
    candidate_probs = []
    for cand in candidates:
        log_prob = gaussian_log_prob(mean, std, cand)
        candidate_probs.append(log_prob)

    # 转换为概率分布
    probs_tensor = torch.stack(candidate_probs).T  # (B,5)
    probs_tensor = F.softmax(probs_tensor, dim=1)  # 归一化

    # 找到最高概率的候选索引
    highest_prob_idx = probs_tensor.argmax(dim=1)  # (B,)

    # 提取最高概率动作
    highest_prob_action = torch.gather(
        candidates_tensor, 1,
        highest_prob_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, CONFIG["motor_dim"])
    ).squeeze(1)  # (B,2)

    return highest_prob_action, probs_tensor


def is_action_repeated(current_action: torch.Tensor, history_actions: list[torch.Tensor]) -> bool:
    assert current_action.shape == (CONFIG["motor_dim"],), f"current_action维度错误：{current_action.shape}"
    if not history_actions:
        return False
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
    batch_size = preferred.shape[0]
    assert preferred.shape == (batch_size, CONFIG["motor_dim"]), f"preferred维度错误：{preferred.shape}"
    assert rejected.shape == (batch_size, CONFIG["motor_dim"]), f"rejected维度错误：{rejected.shape}"

    policy_mean, policy_std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)
    log_p_theta_pref = gaussian_log_prob(policy_mean, policy_std, preferred)  # (B,)
    log_p_theta_rej = gaussian_log_prob(policy_mean, policy_std, rejected)  # (B,)

    with torch.no_grad():
        ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)
        log_p_ref_pref = gaussian_log_prob(ref_mean, ref_std, preferred)
        log_p_ref_rej = gaussian_log_prob(ref_mean, ref_std, rejected)

    advantage = (log_p_theta_pref - log_p_ref_pref) - (log_p_theta_rej - log_p_ref_rej)  # (B,)
    return -F.logsigmoid(CONFIG["dpo_beta"] * advantage).mean()


# ---------------------- 4. 训练/验证函数 ----------------------
def train_one_epoch(epoch: int,
                    train_loader: DataLoader,
                    policy_gen: EncoderOnlyCandidateGenerator,
                    ref_gen: EncoderOnlyCandidateGenerator,
                    optimizer: torch.optim.Optimizer,
                    motor_embed: MotorEmbedding,
                    selected_layers: list) -> tuple[float, int]:
    policy_gen.train()
    total_loss = 0.0
    optimized_count = 0
    history_actions = []
    batch_size = CONFIG["batch_size"]

    pbar = tqdm(enumerate(train_loader),
                desc=f"[训练] Epoch {epoch + 1}/{CONFIG['epochs']} (训练层: {selected_layers})",
                total=min(CONFIG["max_train_samples_per_epoch"], len(train_loader)))

    for sample_idx, batch in pbar:
        if sample_idx >= CONFIG["max_train_samples_per_epoch"]:
            print(f"\n[训练] 已达样本上限（{CONFIG['max_train_samples_per_epoch']}），终止")
            break

        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        assert imgs1.shape[0] == batch_size, f"训练集batch_size错误：{imgs1.shape[0]}"

        images = torch.stack([imgs1, imgs2], dim=2).to(device)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        driver = driver.to(device)
        future_driver = future_driver.to(device)
        future_driver_last = future_driver[:, -1, :]

        with torch.no_grad():
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)
            future_image_embedded = image_embed(future_images)
            img_proj_future = img_sim_model(future_image_embedded)

        generator_output = policy_gen(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=CONFIG["num_candidates"],
            temperature=1.0
        )
        candidates = generator_output[CONFIG["use_candidates"]]
        candidates = [cand.squeeze(1) for cand in candidates]

        preferred, rejected, _ = select_preferred_rejected(
            candidates=candidates,
            img_proj_future=img_proj_future,
            future_driver_last=future_driver_last,
            motor_embed=motor_embed,
            batch_size=batch_size
        )

        current_action = preferred.squeeze(0)
        if is_action_repeated(current_action, history_actions):
            pbar.set_postfix({"状态": "跳过重复动作", "优化样本数": optimized_count})
            continue

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

        history_actions.append(current_action.detach())
        if len(history_actions) > CONFIG["history_cache_size"]:
            history_actions.pop(0)

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
                  motor_embed: MotorEmbedding) -> tuple[float, int, int]:
    policy_gen.eval()
    total_loss = 0.0
    sample_count = 0
    match_count = 0  # 记录匹配次数
    max_batches = CONFIG["max_val_batches"]
    batch_size = CONFIG["val_batch_size"]
    tolerance = CONFIG["action_match_tolerance"]

    with torch.no_grad():
        pbar_total = min(max_batches, len(val_loader))
        pbar = tqdm(enumerate(val_loader),
                    desc=f"[验证] Epoch {epoch + 1}",
                    total=pbar_total)

        for sample_idx, batch in pbar:
            if sample_idx >= max_batches:
                print(f"\n[验证] 已达最大batch数（{max_batches}），终止验证")
                break

            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
            assert imgs1.shape[0] == batch_size, f"验证集batch_size错误：{imgs1.shape[0]}"

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

            # 生成候选动作
            generator_output = policy_gen(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=CONFIG["num_candidates"],
                temperature=1.0
            )
            candidates = generator_output[CONFIG["use_candidates"]]
            candidates = [cand.squeeze(1) for cand in candidates]  # [5, (B,2)]
            candidates_tensor = torch.stack(candidates).permute(1, 0, 2)  # (B,5,2)

            # 1. 选择preferred action
            preferred, rejected, _ = select_preferred_rejected(
                candidates=candidates,
                img_proj_future=img_proj_future,
                future_driver_last=future_driver_last,
                motor_embed=motor_embed,
                batch_size=batch_size
            )  # (B,2)

            # 2. 获取模型最高概率action
            highest_prob_action, _ = get_model_highest_prob_action(
                candidates=candidates,
                candidates_tensor=candidates_tensor,
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                policy_gen=policy_gen
            )  # (B,2)

            # 3. 比较两个动作是否匹配
            for i in range(batch_size):
                if torch.allclose(
                        preferred[i],
                        highest_prob_action[i],
                        atol=tolerance,
                        rtol=0
                ):
                    match_count += 1

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
            sample_count += batch_size

            # 计算当前批次的匹配率
            batch_match_rate = (match_count - (sample_count - batch_size)) / batch_size
            pbar.set_postfix({
                "验证损失": f"{loss.item():.4f}",
                "已验证样本": sample_count,
                "批次匹配率": f"{batch_match_rate:.2%}"
            })

    avg_loss = total_loss / (sample_count / batch_size) if sample_count > 0 else 0.0
    match_rate = match_count / sample_count if sample_count > 0 else 0.0
    print(f"[验证] Epoch {epoch + 1} | 平均损失：{avg_loss:.4f} | 总样本数：{sample_count}")
    print(f"[验证] 匹配统计：prefer与模型最高概率动作匹配 {match_count}/{sample_count}（{match_rate:.2%}）")
    return avg_loss, sample_count, match_count


# ---------------------- 5. 主函数（新增关键层识别和训练层选择逻辑） ----------------------
def main():
    start_total_time = time.time()
    print("\n" + "=" * 60)
    print("          EncoderOnlyCandidateGenerator DPO优化（带关键层识别功能）")
    print("=" * 60)
    print(
        f"[配置信息] 关键层参数: P={CONFIG['P']}, Q={CONFIG['Q']}, 相似度阈值={CONFIG['similarity_threshold'] * 100}%")

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

    # 优化器配置（只优化policy_generator的参数）
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

    # 损失记录（新增关键层相关记录）
    loss_records = {
        "train_loss": [],
        "val_loss": [],
        "train_optimized_samples": [],
        "val_total_samples": [],
        "val_prefer_matches_model": [],
        "val_match_rate": [],
        "key_layers": [],  # 记录每个epoch的关键层
        "selected_layers": [],  # 记录每个epoch选中训练的层
        "lr": []
    }
    best_val_loss = float("inf")

    # 训练循环
    for epoch in range(CONFIG["epochs"]):
        print("\n" + "-" * 50)
        epoch_start_time = time.time()
        print(f"[Epoch {epoch + 1}] 开始关键层识别和训练层选择...")

        # 获取一个样本批次用于关键层识别
        sample_batch = next(iter(train_loader))

        # 1. 识别关键层
        key_layers = identify_key_layers(
            policy_generator,
            sample_batch,
            image_embed,
            motor_embed
        )

        # 2. 从关键层中随机选择Q个层进行训练
        selected_layers = select_layers_for_training(
            policy_generator,
            key_layers
        )

        # 3. 训练（传入选中的层信息用于日志）
        train_loss, optimized_samples = train_one_epoch(
            epoch=epoch,
            train_loader=train_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            optimizer=optimizer,
            motor_embed=motor_embed,
            selected_layers=selected_layers
        )

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 4. 验证
        val_loss, val_samples, val_matches = validate_full(
            epoch=epoch,
            val_loader=val_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            motor_embed=motor_embed
        )
        val_match_rate = val_matches / val_samples if val_samples > 0 else 0.0

        # 5. 记录（新增关键层相关记录）
        loss_records["train_loss"].append(train_loss)
        loss_records["val_loss"].append(val_loss)
        loss_records["train_optimized_samples"].append(optimized_samples)
        loss_records["val_total_samples"].append(val_samples)
        loss_records["val_prefer_matches_model"].append(val_matches)
        loss_records["val_match_rate"].append(val_match_rate)
        loss_records["key_layers"].append(key_layers)  # 记录关键层
        loss_records["selected_layers"].append(selected_layers)  # 记录选中的训练层
        loss_records["lr"].append(current_lr)

        # 打印总结
        epoch_time = time.time() - epoch_start_time
        print(f"\n[Epoch 总结] Epoch {epoch + 1}/{CONFIG['epochs']}")
        print(f"  - 耗时：{epoch_time:.2f}秒 | 学习率：{current_lr:.7f}")
        print(f"  - 训练：{train_loss:.4f}（{optimized_samples}样本）")
        print(f"  - 验证：{val_loss:.4f}（{val_samples}样本）")
        print(f"  - 匹配统计：{val_matches}/{val_samples}（{val_match_rate:.2%}）")
        print(f"  - 关键层：{key_layers}")
        print(f"  - 选中训练层：{selected_layers}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "policy_generator_state_dict": policy_generator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": CONFIG,
                "loss_records": loss_records
            }, CONFIG["dpo_save_path"])
            print(f"  - ✅ 保存最佳模型至：{CONFIG['dpo_save_path']}")

    # 训练完成，保存关键层历史记录
    np.save(CONFIG["key_layers_record_path"], loss_records)
    total_time = time.time() - start_total_time

    print("\n" + "=" * 60)
    print("                          DPO优化训练完成")
    print("=" * 60)
    print(f"总耗时：{total_time:.2f}秒 | 最佳验证损失：{best_val_loss:.4f}")
    print(f"总匹配统计：{sum(loss_records['val_prefer_matches_model'])}/{sum(loss_records['val_total_samples'])}")
    print(f"模型路径：{CONFIG['dpo_save_path']} | 关键层记录：{CONFIG['key_layers_record_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
