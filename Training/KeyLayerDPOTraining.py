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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"[初始化] 使用设备: {device}")

# ---------------------- 核心配置参数（删除use_candidates，适配单组动作） ----------------------
CONFIG = {
    # 验证控制
    "max_val_batches": 20,
    "val_batch_size": 6,

    # 相似度加权系数
    "alpha": 0.9,
    "action_match_tolerance": 1e-4,

    # 关键层识别参数
    "P": 3,  # 倒数P层特殊处理
    "Q": 5,  # 每次训练随机选择Q个关键层更新
    "similarity_threshold": 0.85,  # 80%相似度阈值

    "batch_size": 1,
    "epochs": 10,
    "lr": 5e-3,
    "num_candidates": 5,  # 单组候选动作数量
    "sampling_workers": 2,
    "max_train_samples_per_epoch": 100,
    "dpo_beta": 0.1,
    "repeat_threshold": 0.97,
    "history_cache_size": 0,
    # 【核心删除】移除use_candidates（不再区分candidates1/candidates2）
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,  # Transformer层数
    "motor_dim": 2,  # 单组动作含2个电机信号
    "gen_seq_len": 30,
    "sim_seq_len": 30,
    "embed_dim_sim": 128,
    "num_layers_sim": 3,
    "nhead_sim": 4,
    "similarity_dim": 32,
    "data_root_dirs": '/data/cyzhao/collector_cydpo/dpo_data',
    "pretrained_model_path": "./saved_models/best_model",
    "dpo_save_path": "./saved_models/key_layers_dpo_final_best_model",
    "dpo_loss_path": "./loss_records/key_layers_dpo_final_loss.npy",
    "key_layers_record_path": "./loss_records/key_layers_history.npy"  # 记录关键层历史
}

# 创建保存目录
os.makedirs(os.path.dirname(CONFIG["dpo_save_path"]), exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["dpo_loss_path"]), exist_ok=True)


# # ---------------------- Transformer相关模型定义（适配单组动作参数） ----------------------
class TransformerEncoderModel(nn.Module):
    def __init__(self, embed_dim=64, nhead=8, num_layers=16):
        super(TransformerEncoderModel, self).__init__()
        # 存储每一层的编码器（输入维度：embed_dim*3，与融合特征匹配）
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim * 3, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        # 层训练控制开关（默认全部可训练）
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
            # 根据开关控制梯度计算
            if self.layer_trainable[i]:
                current = layer(current)
            else:
                with torch.no_grad():
                    current = layer(current)
            layer_outputs.append(current)

        return layer_outputs  # [num_layers, batch, seq, 3*embed_dim]


class EncoderOnlyCandidateGenerator(nn.Module):
    """【核心修改】适配单组动作参数，删除双组输出层"""
    def __init__(self, embed_dim, nhead, num_layers, motor_dim=2, max_seq_length=100):
        super().__init__()
        self.d_model = embed_dim * 3  # 融合特征维度（motor_embed + image_embed）
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder = TransformerEncoderModel(embed_dim, nhead, num_layers)

        # 【核心修改】仅保留一组输出层（对应2个电机信号）
        self.fc_mean = nn.Linear(self.d_model, motor_dim)  # 动作均值（batch, 2）
        self.fc_logvar = nn.Linear(self.d_model, motor_dim)  # 动作对数方差（batch, 2）

    def forward(self, image_embedded, motor_embedded, num_candidates=5, temperature=0.5):
        """
        生成单组候选动作，返回分布参数和每一层编码器输出（用于关键层识别）
        :param image_embedded: (batch, seq, 2*embed_dim)
        :param motor_embedded: (batch, seq, embed_dim)
        :return: dict含单组候选、分布参数、层输出
        """
        # 1. 融合图像与电机特征
        combined = torch.cat([motor_embedded, image_embedded], dim=-1)  # (batch, seq, 3*embed_dim)
        combined = self.positional_encoding(combined)  # 加位置编码

        # 2. Transformer编码器（返回所有层输出）
        encoder_layer_outputs = self.encoder(combined)  # [num_layers, batch, seq, 3*embed_dim]
        final_encoder_out = encoder_layer_outputs[-1]  # 最后一层输出
        global_feat = final_encoder_out.mean(dim=1)  # 序列均值作为全局特征（batch, 3*embed_dim）

        # 3. 预测单组动作分布参数
        mean = self.fc_mean(global_feat)  # (batch, 2)
        logvar = self.fc_logvar(global_feat)  # (batch, 2)
        logvar = torch.clamp(logvar, min=-5, max=5)  # 限制方差范围
        std = torch.exp(0.5 * logvar) * temperature  # 温度调整标准差

        # 4. 生成单组候选动作
        candidates = []
        for _ in range(num_candidates):
            eps = torch.randn_like(mean)  # 标准正态噪声（batch, 2）
            sample = mean + std * eps  # 重参数化采样
            sample = torch.tanh(sample)  # 动作范围限制在[-1,1]
            candidates.append(sample.unsqueeze(1))  # 增加时间维度（batch, 1, 2）

        # 【核心修改】返回单组候选和参数，保留层输出用于关键层识别
        return {
            'candidates': candidates,  # 单组候选动作列表：[num_candidates, (batch,1,2)]
            'mean': mean,              # 单组均值：(batch,2)
            'std': std,                # 单组标准差：(batch,2)
            'encoder_layer_outputs': encoder_layer_outputs  # 所有层输出：用于关键层识别
        }


# ---------------------- 1. 模型加载（适配单组动作模型权重） ----------------------
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

    # 加载单组动作模型权重（键为mean/std，无mean1/mean2）
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

    # 冻结非策略模型（仅policy_generator可训练）
    for model in [image_embed, motor_embed, ref_generator, img_sim_model, driver_sim_model]:
        for param in model.parameters():
            param.requires_grad = False
    print("[模型配置] 仅EncoderOnlyCandidateGenerator可训练")

    return image_embed, motor_embed, policy_generator, ref_generator, img_sim_model, driver_sim_model


# ---------------------- 2. 数据加载（无需修改） ----------------------
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

    # 检查batch_size一致性
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
        pin_memory=False,
        drop_last=False
    )

    check_batch_size(train_loader, "训练集", CONFIG["batch_size"])
    check_batch_size(val_loader, "验证集", CONFIG["val_batch_size"])
    print(f"[数据配置] 训练集batch_size={CONFIG['batch_size']} | 验证集batch_size={CONFIG['val_batch_size']}")

    return train_loader, val_loader


# ---------------------- 3. 核心工具函数（适配单组动作，保留关键层逻辑） ----------------------
def calculate_layer_similarity(layer_outputs):
    """计算层间余弦相似度（关键层识别核心，逻辑不变）"""
    num_layers = len(layer_outputs)
    similarity_matrix = torch.zeros(num_layers, num_layers, device=device)

    for i in range(num_layers):
        # 展平特征：[batch, seq, dim] → [batch, seq*dim]
        output_i = layer_outputs[i].view(layer_outputs[i].shape[0], -1)
        output_i_norm = F.normalize(output_i, dim=1)

        for j in range(num_layers):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue

            output_j = layer_outputs[j].view(layer_outputs[j].shape[0], -1)
            output_j_norm = F.normalize(output_j, dim=1)

            # 批次平均相似度
            sim = F.cosine_similarity(output_i_norm, output_j_norm, dim=1).mean()
            similarity_matrix[i, j] = sim

    return similarity_matrix


def identify_key_layers(policy_generator, sample_batch, image_embed, motor_embed):
    """识别关键层（逻辑不变，适配单组动作模型输出）"""
    # 解包样本
    imgs1, imgs2, driver, _, _, _ = sample_batch
    images = torch.stack([imgs1, imgs2], dim=2).to(device)
    driver = driver.to(device)

    # 特征嵌入（无梯度）
    with torch.no_grad():
        image_embedded = image_embed(images)
        motor_embedded = motor_embed(driver)

    # 【适配修改】获取单组动作模型输出（含层输出）
    model_output = policy_generator(
        image_embedded=image_embedded,
        motor_embedded=motor_embedded,
        num_candidates=1  # 仅需1个候选用于层分析
    )
    layer_outputs = model_output["encoder_layer_outputs"]
    num_layers = len(layer_outputs)
    P = CONFIG["P"]

    # 计算层间相似度矩阵
    similarity_matrix = calculate_layer_similarity(layer_outputs)

    # 识别关键层
    key_layers = set()
    key_layers.add(num_layers - 1)  # 最后一层默认是关键层

    for i in range(num_layers):
        if i >= num_layers - P:
            # 倒数P层：与后续所有层比较
            next_layers = range(i + 1, num_layers)
            if not next_layers:
                continue
            avg_sim = torch.mean(similarity_matrix[i, list(next_layers)]).item()
            if avg_sim >= CONFIG["similarity_threshold"]:
                key_layers.add(i)
        else:
            # 其他层：与后续P层比较
            next_layers = range(i + 1, min(i + 1 + P, num_layers))
            if not next_layers:
                continue
            avg_sim = torch.mean(similarity_matrix[i, list(next_layers)]).item()
            if avg_sim >= CONFIG["similarity_threshold"]:
                key_layers.add(i)

    key_layers_list = sorted(list(key_layers))
    print(f"[关键层识别] 共识别到 {len(key_layers_list)} 个关键层: {key_layers_list}")
    return key_layers_list


def select_layers_for_training(policy_generator, key_layers):
    """从关键层选Q个训练（逻辑不变）"""
    num_layers = len(policy_generator.encoder.layers)
    Q = CONFIG["Q"]

    # 选Q个关键层（不足则全选）
    if len(key_layers) <= Q:
        selected_layers = key_layers
    else:
        selected_indices = np.random.choice(len(key_layers), Q, replace=False)
        selected_layers = [key_layers[i] for i in selected_indices]

    # 先冻结所有层，再激活选中层
    for i in range(num_layers):
        policy_generator.encoder.set_layer_trainable(i, False)
    for layer_idx in selected_layers:
        policy_generator.encoder.set_layer_trainable(layer_idx, True)

    print(
        f"[训练层选择] 从 {len(key_layers)} 个关键层中随机选择了 {len(selected_layers)} 个进行训练: {sorted(selected_layers)}")
    return selected_layers


def gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """计算高斯对数概率（适配单组mean/std）"""
    eps = 1e-6
    std = std + eps  # 避免log(0)
    log_prob = -0.5 * torch.log(2 * torch.tensor(np.pi, device=device)) - torch.log(std) - (action - mean) ** 2 / (
                2 * std ** 2)
    return log_prob.sum(dim=-1)


def get_generator_distribution(generator: EncoderOnlyCandidateGenerator,
                               image_embedded: torch.Tensor,
                               motor_embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """【核心修改】获取单组动作分布参数（删除双组分支）"""
    combined = torch.cat([motor_embedded, image_embedded], dim=-1)  # (batch, seq, 3*embed_dim)
    combined = generator.positional_encoding(combined)
    encoder_out = generator.encoder(combined)  # 层输出列表
    final_encoder_out = encoder_out[-1]  # 最后一层输出
    global_feat = final_encoder_out.mean(dim=1)  # (batch, 3*embed_dim)

    # 【核心修改】仅获取单组mean/std
    mean = generator.fc_mean(global_feat)  # (batch, 2)
    logvar = generator.fc_logvar(global_feat)  # (batch, 2)
    logvar = torch.clamp(logvar, min=-5, max=5)
    std = torch.exp(0.5 * logvar)  # (batch, 2)

    return mean, std


def select_preferred_rejected(candidates: list[torch.Tensor],
                              img_proj_future: torch.Tensor,
                              future_driver_last: torch.Tensor,
                              motor_embed: MotorEmbedding,
                              batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """选择偏好/非偏好动作（适配单组候选，逻辑不变）"""
    # 维度检查（单组候选）
    assert len(candidates) == CONFIG["num_candidates"], f"候选数={len(candidates)}，需为{CONFIG['num_candidates']}"
    for i, cand in enumerate(candidates):
        assert cand.shape == (batch_size, 1, CONFIG["motor_dim"]), \
            f"候选{i}维度错误：{cand.shape}，需为({batch_size},1,{CONFIG['motor_dim']})"
    assert future_driver_last.shape == (batch_size, CONFIG["motor_dim"]), \
        f"future_driver_last维度错误：{future_driver_last.shape}"

    # 候选动作嵌入
    candidate_embeddings = []
    for cand in candidates:
        emb = motor_embed(cand)  # (batch, 1, embed_dim_gen)
        candidate_embeddings.append(emb)

    # 图像相似度计算
    sim_img = []
    for emb in candidate_embeddings:
        cand_proj = driver_sim_model(emb).squeeze(1)  # (batch, similarity_dim)
        img_proj_squeezed = img_proj_future.squeeze(1)  # (batch, similarity_dim)
        sim = F.cosine_similarity(cand_proj, img_proj_squeezed, dim=1)  # (batch,)
        sim_img.append(sim)

    # 动作相似度计算
    future_driver_seq = future_driver_last.unsqueeze(1)  # (batch, 1, 2)
    future_driver_emb = motor_embed(future_driver_seq)  # (batch, 1, embed_dim_gen)
    future_emb_squeezed = future_driver_emb.squeeze(1)  # (batch, embed_dim_gen)
    future_norm = F.normalize(future_emb_squeezed, dim=1)  # (batch, embed_dim_gen)

    sim_driver = []
    for emb in candidate_embeddings:
        emb_squeezed = emb.squeeze(1)  # (batch, embed_dim_gen)
        emb_norm = F.normalize(emb_squeezed, dim=1)  # (batch, embed_dim_gen)
        sim = F.cosine_similarity(emb_norm, future_norm, dim=1)  # (batch,)
        sim_driver.append(sim)

    # 总相似度（加权融合）
    alpha = CONFIG["alpha"]
    sim_img_tensor = torch.stack(sim_img).T  # (batch, num_candidates)
    sim_driver_tensor = torch.stack(sim_driver).T  # (batch, num_candidates)
    sim_total = alpha * sim_img_tensor + (1 - alpha) * sim_driver_tensor  # (batch, num_candidates)

    # 选择偏好/非偏好动作
    preferred_idx = sim_total.argmax(dim=1)  # (batch,)
    rejected_idx = sim_total.argmin(dim=1)  # (batch,)

    # 提取动作
    candidates_tensor = torch.stack(candidates).permute(1, 0, 2, 3)  # (batch, num_candidates, 1, 2)
    preferred = torch.gather(
        candidates_tensor, 1,
        preferred_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, CONFIG["motor_dim"])
    ).squeeze(1).squeeze(1)  # (batch, 2)
    rejected = torch.gather(
        candidates_tensor, 1,
        rejected_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, CONFIG["motor_dim"])
    ).squeeze(1).squeeze(1)  # (batch, 2)

    return preferred, rejected, sim_total


def get_model_highest_prob_action(candidates: list[torch.Tensor],
                                  candidates_tensor: torch.Tensor,
                                  image_embedded: torch.Tensor,
                                  motor_embedded: torch.Tensor,
                                  policy_gen: EncoderOnlyCandidateGenerator) -> torch.Tensor:
    """获取模型最高概率动作（适配单组分布参数）"""
    batch_size = image_embedded.shape[0]
    num_candidates = len(candidates)

    # 获取单组分布参数
    mean, std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)  # (batch,2), (batch,2)

    # 计算每个候选的概率
    candidate_probs = []
    for cand in candidates:
        cand_squeezed = cand.squeeze(1)  # (batch, 2)
        log_prob = gaussian_log_prob(mean, std, cand_squeezed)  # (batch,)
        candidate_probs.append(log_prob)

    # 概率分布归一化
    probs_tensor = torch.stack(candidate_probs).T  # (batch, num_candidates)
    probs_tensor = F.softmax(probs_tensor, dim=1)  # (batch, num_candidates)

    # 提取最高概率动作
    highest_prob_idx = probs_tensor.argmax(dim=1)  # (batch,)
    highest_prob_action = torch.gather(
        candidates_tensor, 1,
        highest_prob_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, CONFIG["motor_dim"])
    ).squeeze(1)  # (batch, 2)

    return highest_prob_action, probs_tensor


def is_action_repeated(current_action: torch.Tensor, history_actions: list[torch.Tensor]) -> bool:
    """动作重复检查（逻辑不变）"""
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
    """DPO损失计算（适配单组动作参数）"""
    batch_size = preferred.shape[0]
    assert preferred.shape == (batch_size, CONFIG["motor_dim"]), f"preferred维度错误：{preferred.shape}"
    assert rejected.shape == (batch_size, CONFIG["motor_dim"]), f"rejected维度错误：{rejected.shape}"

    # 策略模型概率（单组参数）
    policy_mean, policy_std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)
    log_p_theta_pref = gaussian_log_prob(policy_mean, policy_std, preferred)  # (batch,)
    log_p_theta_rej = gaussian_log_prob(policy_mean, policy_std, rejected)  # (batch,)

    # 参考模型概率（单组参数，无梯度）
    with torch.no_grad():
        ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)
        log_p_ref_pref = gaussian_log_prob(ref_mean, ref_std, preferred)
        log_p_ref_rej = gaussian_log_prob(ref_mean, ref_std, rejected)

    # 计算DPO优势与损失
    advantage = (log_p_theta_pref - log_p_ref_pref) - (log_p_theta_rej - log_p_ref_rej)  # (batch,)
    return -F.logsigmoid(CONFIG["dpo_beta"] * advantage).mean()


# ---------------------- 4. 训练/验证函数（适配单组候选动作） ----------------------
def train_one_epoch(epoch: int,
                    train_loader: DataLoader,
                    policy_gen: EncoderOnlyCandidateGenerator,
                    ref_gen: EncoderOnlyCandidateGenerator,
                    optimizer: torch.optim.Optimizer,
                    motor_embed: MotorEmbedding,
                    selected_layers: list) -> tuple[float, int]:
    """训练单epoch（适配单组候选动作）"""
    policy_gen.train()
    total_loss = 0.0
    optimized_count = 0
    history_actions = []
    batch_size = CONFIG["batch_size"]

    # 进度条显示训练层信息
    pbar = tqdm(enumerate(train_loader),
                desc=f"[训练] Epoch {epoch + 1}/{CONFIG['epochs']} (训练层: {selected_layers})",
                total=min(CONFIG["max_train_samples_per_epoch"], len(train_loader)))

    for sample_idx, batch in pbar:
        if sample_idx >= CONFIG["max_train_samples_per_epoch"]:
            print(f"\n[训练] 已达样本上限（{CONFIG['max_train_samples_per_epoch']}），终止")
            break

        # 解包数据
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        assert imgs1.shape[0] == batch_size, f"训练集batch_size错误：{imgs1.shape[0]}"

        images = torch.stack([imgs1, imgs2], dim=2).to(device)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        driver = driver.to(device)
        future_driver = future_driver.to(device)
        future_driver_last = future_driver[:, -1, :]  # (batch, 2)

        # 冻结特征提取模块梯度
        with torch.no_grad():
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)
            future_image_embedded = image_embed(future_images)
            img_proj_future = img_sim_model(future_image_embedded)  # (batch, 1, similarity_dim)

        # 【核心修改】获取单组候选动作
        generator_output = policy_gen(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=CONFIG["num_candidates"],
            temperature=1.0
        )
        candidates = generator_output["candidates"]  # 单组候选列表：[num_candidates, (batch,1,2)]

        # 选择偏好/非偏好动作
        preferred, rejected, _ = select_preferred_rejected(
            candidates=candidates,
            img_proj_future=img_proj_future,
            future_driver_last=future_driver_last,
            motor_embed=motor_embed,
            batch_size=batch_size
        )  # (batch,2), (batch,2)

        # 跳过重复动作
        current_action = preferred.squeeze(0)  # (2,)
        if is_action_repeated(current_action, history_actions):
            pbar.set_postfix({"状态": "跳过重复动作", "优化样本数": optimized_count})
            continue

        # DPO损失计算与优化
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

        # 更新历史动作缓存
        history_actions.append(current_action.detach())
        if len(history_actions) > CONFIG["history_cache_size"]:
            history_actions.pop(0)

        # 累计损失与计数
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
    """验证函数（适配单组候选动作）"""
    policy_gen.eval()
    total_loss = 0.0
    sample_count = 0
    match_count = 0  # 偏好动作与最高概率动作匹配次数
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

            # 解包数据
            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
            assert imgs1.shape[0] == batch_size, f"验证集batch_size错误：{imgs1.shape[0]}"

            images = torch.stack([imgs1, imgs2], dim=2).to(device)
            future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
            driver = driver.to(device)
            future_driver = future_driver.to(device)
            future_driver_last = future_driver[:, -1, :]  # (batch, 2)

            # 特征嵌入
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)
            future_image_embedded = image_embed(future_images)
            img_proj_future = img_sim_model(future_image_embedded)  # (batch, 1, similarity_dim)

            # 【核心修改】获取单组候选动作
            generator_output = policy_gen(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=CONFIG["num_candidates"],
                temperature=1.0
            )
            candidates = generator_output["candidates"]  # 单组候选列表
            candidates_squeezed = [cand.squeeze(1) for cand in candidates]  # [num_candidates, (batch,2)]
            candidates_tensor = torch.stack(candidates_squeezed).permute(1, 0, 2)  # (batch, num_candidates, 2)

            # 1. 选择偏好动作
            preferred, rejected, _ = select_preferred_rejected(
                candidates=candidates,
                img_proj_future=img_proj_future,
                future_driver_last=future_driver_last,
                motor_embed=motor_embed,
                batch_size=batch_size
            )  # (batch,2)

            # 2. 获取模型最高概率动作
            highest_prob_action, _ = get_model_highest_prob_action(
                candidates=candidates_squeezed,
                candidates_tensor=candidates_tensor,
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                policy_gen=policy_gen
            )  # (batch,2)

            # 3. 统计匹配次数（逐样本）
            for i in range(batch_size):
                if torch.allclose(
                        preferred[i],
                        highest_prob_action[i],
                        atol=tolerance,
                        rtol=0
                ):
                    match_count += 1

            # 4. 计算DPO损失
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

            # 进度条更新
            batch_match_rate = (match_count - (sample_count - batch_size)) / batch_size
            pbar.set_postfix({
                "验证损失": f"{loss.item():.4f}",
                "已验证样本": sample_count,
                "批次匹配率": f"{batch_match_rate:.2%}"
            })

    # 计算验证指标
    avg_loss = total_loss / (sample_count / batch_size) if sample_count > 0 else 0.0
    match_rate = match_count / sample_count if sample_count > 0 else 0.0
    print(f"[验证] Epoch {epoch + 1} | 平均损失：{avg_loss:.4f} | 总样本数：{sample_count}")
    print(f"[验证] 匹配统计：prefer与模型最高概率动作匹配 {match_count}/{sample_count}（{match_rate:.2%}）")
    return avg_loss, sample_count, match_count


# ---------------------- 5. 主函数（逻辑不变，适配修改后的子函数） ----------------------
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

    # 优化器配置（仅优化policy_generator）
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

    # 损失记录（含关键层历史）
    loss_records = {
        "train_loss": [],
        "val_loss": [],
        "train_optimized_samples": [],
        "val_total_samples": [],
        "val_prefer_matches_model": [],
        "val_match_rate": [],
        "key_layers": [],  # 记录每个epoch的关键层
        "selected_layers": [],  # 记录每个epoch选中的训练层
        "lr": []
    }
    best_val_loss = float("inf")

    # 训练循环（含关键层识别）
    for epoch in range(CONFIG["epochs"]):
        print("\n" + "-" * 50)
        epoch_start_time = time.time()
        print(f"[Epoch {epoch + 1}] 开始关键层识别和训练层选择...")

        # 获取一个样本批次用于关键层识别
        sample_batch = next(iter(train_loader))

        # 1. 识别关键层（适配单组动作模型）
        key_layers = identify_key_layers(
            policy_generator,
            sample_batch,
            image_embed,
            motor_embed
        )

        # 2. 从关键层选Q个训练
        selected_layers = select_layers_for_training(
            policy_generator,
            key_layers
        )

        # 3. 训练（传入选中图层信息）
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
        current_lr = optimizer.param_groups[0]["lr"]

        # 4. 验证（适配单组动作）
        val_loss, val_samples, val_matches = validate_full(
            epoch=epoch,
            val_loader=val_loader,
            policy_gen=policy_generator,
            ref_gen=ref_generator,
            motor_embed=motor_embed
        )
        val_match_rate = val_matches / val_samples if val_samples > 0 else 0.0

        # 5. 记录指标（含关键层信息）
        loss_records["train_loss"].append(train_loss)
        loss_records["val_loss"].append(val_loss)
        loss_records["train_optimized_samples"].append(optimized_samples)
        loss_records["val_total_samples"].append(val_samples)
        loss_records["val_prefer_matches_model"].append(val_matches)
        loss_records["val_match_rate"].append(val_match_rate)
        loss_records["key_layers"].append(key_layers)
        loss_records["selected_layers"].append(selected_layers)
        loss_records["lr"].append(current_lr)

        # 打印Epoch总结
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

    # 训练完成：保存关键层历史记录
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
