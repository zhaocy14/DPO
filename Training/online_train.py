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
                          PositionalEncoding)
from tqdm import tqdm

# 设备初始化
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"[初始化] 使用设备: {device}")

# ---------------------- 核心配置参数 ----------------------
CONFIG = {
    "alpha": 0.9,
    "action_match_tolerance": 1e-4,
    "P": 3,
    "Q": 5,
    "similarity_threshold": 0.85,
    "batch_size": 1,
    "total_frames": 6000,
    "lr": 5e-4,
    "num_candidates": 5,
    "sampling_workers": 2,
    "dpo_beta": 0.1,
    "repeat_threshold": 0.97,  # 动作重复判断阈值
    "history_cache_size": 0,  # 增加历史缓存大小，更准确判断重复
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,
    "gen_seq_len": 30,
    "sim_seq_len": 30,
    "embed_dim_sim": 128,
    "num_layers_sim": 3,
    "nhead_sim": 4,
    "similarity_dim": 32,
    "data_root_dirs": '/data/cyzhao/collector_cydpo/dpo_data',
    "pretrained_model_path": "./saved_models/best_model",
    "save_path": "./saved_models/inference_train_loop_model",
    "stats_path": "./loss_records/inference_train_stats.npy",
    "save_interval": 500,
    "window_size": 5,  # 将窗口大小设为配置参数，方便调整
    "debug_mode": False  # 调试模式开关
}

# 创建保存目录
os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["stats_path"]), exist_ok=True)


# ---------------------- 模型定义 ----------------------
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


# ---------------------- 模型加载 ----------------------
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

    # 加载模型权重
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


# ---------------------- 数据加载 ----------------------
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
        dataset = all_dataset.training_dataset
        print(f"[数据加载] 数据集规模：{len(dataset)}样本")
    except Exception as e:
        raise RuntimeError(f"[数据集加载失败] {str(e)}") from e

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=CONFIG["sampling_workers"],
        pin_memory=True,
        drop_last=False
    )

    return data_loader


def split_sample_into_frames(sample):
    imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = sample
    seq_len = imgs1.shape[1]

    frames = []
    for i in range(seq_len):
        frame = (
            imgs1[:, i:i + 1, ...],
            imgs2[:, i:i + 1, ...],
            driver[:, i:i + 1, ...],
            future_imgs1[:, i:i + 1, ...],
            future_imgs2[:, i:i + 1, ...],
            future_driver[:, i:i + 1, ...]
        )
        frames.append(frame)

    return frames


# ---------------------- 核心工具函数 ----------------------
def calculate_layer_similarity(layer_outputs):
    num_layers = len(layer_outputs)
    similarity_matrix = torch.zeros(num_layers, num_layers, device=device)

    for i in range(num_layers):
        output_i = layer_outputs[i].view(layer_outputs[i].shape[0], -1)
        output_i_norm = F.normalize(output_i, dim=1)

        for j in range(num_layers):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue

            output_j = layer_outputs[j].view(layer_outputs[j].shape[0], -1)
            output_j_norm = F.normalize(output_j, dim=1)
            sim = F.cosine_similarity(output_i_norm, output_j_norm, dim=1).mean()
            similarity_matrix[i, j] = sim

    return similarity_matrix


def identify_key_layers(policy_generator, frame, image_embed, motor_embed):
    imgs1, imgs2, driver, _, _, _ = frame
    images = torch.stack([imgs1, imgs2], dim=2).to(device)
    driver = driver.to(device)

    with torch.no_grad():
        image_embedded = image_embed(images)
        motor_embedded = motor_embed(driver)

    model_output = policy_generator(
        image_embedded=image_embedded,
        motor_embedded=motor_embedded,
        num_candidates=1
    )
    layer_outputs = model_output["encoder_layer_outputs"]
    num_layers = len(layer_outputs)
    P = CONFIG["P"]

    similarity_matrix = calculate_layer_similarity(layer_outputs)

    key_layers = set()
    key_layers.add(num_layers - 1)

    for i in range(num_layers):
        if i >= num_layers - P:
            next_layers = range(i + 1, num_layers)
            if not next_layers:
                continue
            avg_sim = torch.mean(similarity_matrix[i, list(next_layers)]).item()
            if avg_sim >= CONFIG["similarity_threshold"]:
                key_layers.add(i)
        else:
            next_layers = range(i + 1, min(i + 1 + P, num_layers))
            if not next_layers:
                continue
            avg_sim = torch.mean(similarity_matrix[i, list(next_layers)]).item()
            if avg_sim >= CONFIG["similarity_threshold"]:
                key_layers.add(i)

    return sorted(list(key_layers))


def select_layers_for_training(policy_generator, key_layers):
    num_layers = len(policy_generator.encoder.layers)
    Q = CONFIG["Q"]

    if len(key_layers) <= Q:
        selected_layers = key_layers
    else:
        selected_indices = np.random.choice(len(key_layers), Q, replace=False)
        selected_layers = [key_layers[i] for i in selected_indices]

    for i in range(num_layers):
        policy_generator.encoder.set_layer_trainable(i, False)
    for layer_idx in selected_layers:
        policy_generator.encoder.set_layer_trainable(layer_idx, True)

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
    final_encoder_out = encoder_out[-1]
    global_feat = final_encoder_out.mean(dim=1)

    mean = generator.fc_mean(global_feat)
    logvar = generator.fc_logvar(global_feat)
    logvar = torch.clamp(logvar, min=-5, max=5)
    std = torch.exp(0.5 * logvar)

    return mean, std


def select_preferred_rejected(candidates: list[torch.Tensor],
                              img_proj_future: torch.Tensor,
                              future_driver_last: torch.Tensor,
                              motor_embed: MotorEmbedding,
                              driver_sim_model: SimilarityModelDriver,
                              batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(candidates) == CONFIG["num_candidates"], f"候选数={len(candidates)}，需为{CONFIG['num_candidates']}"
    for i, cand in enumerate(candidates):
        assert cand.shape == (batch_size, 1, CONFIG["motor_dim"]), \
            f"候选{i}维度错误：{cand.shape}，需为({batch_size},1,{CONFIG['motor_dim']})"
    assert future_driver_last.shape == (batch_size, CONFIG["motor_dim"]), \
        f"future_driver_last维度错误：{future_driver_last.shape}"

    candidate_embeddings = []
    for cand in candidates:
        emb = motor_embed(cand)
        candidate_embeddings.append(emb)

    sim_img = []
    for emb in candidate_embeddings:
        cand_proj = driver_sim_model(emb).squeeze(1)
        img_proj_squeezed = img_proj_future.squeeze(1)
        sim = F.cosine_similarity(cand_proj, img_proj_squeezed, dim=1)
        sim_img.append(sim)

    future_driver_seq = future_driver_last.unsqueeze(1)
    future_driver_emb = motor_embed(future_driver_seq)
    future_emb_squeezed = future_driver_emb.squeeze(1)
    future_norm = F.normalize(future_emb_squeezed, dim=1)

    sim_driver = []
    for emb in candidate_embeddings:
        emb_squeezed = emb.squeeze(1)
        emb_norm = F.normalize(emb_squeezed, dim=1)
        sim = F.cosine_similarity(emb_norm, future_norm, dim=1)
        sim_driver.append(sim)

    alpha = CONFIG["alpha"]
    sim_img_tensor = torch.stack(sim_img).T
    sim_driver_tensor = torch.stack(sim_driver).T
    sim_total = alpha * sim_img_tensor + (1 - alpha) * sim_driver_tensor

    preferred_idx = sim_total.argmax(dim=1)
    rejected_idx = sim_total.argmin(dim=1)

    candidates_tensor = torch.stack(candidates).permute(1, 0, 2, 3)
    preferred = torch.gather(
        candidates_tensor, 1,
        preferred_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, CONFIG["motor_dim"])
    ).squeeze(1).squeeze(1)
    rejected = torch.gather(
        candidates_tensor, 1,
        rejected_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, CONFIG["motor_dim"])
    ).squeeze(1).squeeze(1)

    return preferred, rejected, sim_total


def get_model_highest_prob_action(candidates: list[torch.Tensor],
                                  candidates_tensor: torch.Tensor,
                                  image_embedded: torch.Tensor,
                                  motor_embedded: torch.Tensor,
                                  policy_gen: EncoderOnlyCandidateGenerator) -> torch.Tensor:
    batch_size = image_embedded.shape[0]
    num_candidates = len(candidates)

    mean, std = get_generator_distribution(policy_gen, image_embedded, motor_embedded)

    candidate_probs = []
    for cand in candidates:
        cand_squeezed = cand.squeeze(1)
        log_prob = gaussian_log_prob(mean, std, cand_squeezed)
        candidate_probs.append(log_prob)

    probs_tensor = torch.stack(candidate_probs).T
    probs_tensor = F.softmax(probs_tensor, dim=1)

    highest_prob_idx = probs_tensor.argmax(dim=1)
    highest_prob_action = torch.gather(
        candidates_tensor, 1,
        highest_prob_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, CONFIG["motor_dim"])
    ).squeeze(1)

    return highest_prob_action, probs_tensor


def is_action_repeated(current_action: torch.Tensor, history_actions: list[torch.Tensor]) -> tuple[bool, float]:
    """
    判断当前动作是否与历史动作重复

    返回值:
        - 布尔值: 是否重复
        - 浮点数: 最高相似度值（用于调试）
    """
    assert current_action.shape == (CONFIG["motor_dim"],), f"current_action维度错误：{current_action.shape}"

    if not history_actions:
        return False, 0.0  # 无历史动作，不重复

    current_norm = F.normalize(current_action.unsqueeze(0), dim=1)
    max_similarity = 0.0

    for hist_action in history_actions:
        assert hist_action.shape == (CONFIG["motor_dim"],), f"历史动作维度错误：{hist_action.shape}"
        hist_norm = F.normalize(hist_action.unsqueeze(0), dim=1)
        sim = F.cosine_similarity(current_norm, hist_norm, dim=1).item()

        if sim > max_similarity:
            max_similarity = sim

        if sim > CONFIG["repeat_threshold"]:
            return True, sim  # 找到重复动作，返回

    return False, max_similarity  # 不重复，返回最高相似度


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
    log_p_theta_pref = gaussian_log_prob(policy_mean, policy_std, preferred)
    log_p_theta_rej = gaussian_log_prob(policy_mean, policy_std, rejected)

    with torch.no_grad():
        ref_mean, ref_std = get_generator_distribution(ref_gen, image_embedded, motor_embedded)
        log_p_ref_pref = gaussian_log_prob(ref_mean, ref_std, preferred)
        log_p_ref_rej = gaussian_log_prob(ref_mean, ref_std, rejected)

    advantage = (log_p_theta_pref - log_p_ref_pref) - (log_p_theta_rej - log_p_ref_rej)
    return -F.logsigmoid(CONFIG["dpo_beta"] * advantage).mean()


# ---------------------- 推理-训练循环 ----------------------
def inference_train_loop(data_loader,
                         policy_gen,
                         ref_gen,
                         optimizer,
                         motor_embed,
                         image_embed,
                         img_sim_model,
                         driver_sim_model):
    policy_gen.train()
    total_loss = 0.0
    processed_frames = 0
    optimized_count = 0
    history_actions = []  # 存储历史动作，用于判断重复
    batch_size = CONFIG["batch_size"]
    window_size = CONFIG["window_size"]

    # 滑动窗口统计优化
    match_window = []  # 存储最近window_size帧的匹配结果
    match_stats = []  # 存储窗口统计信息
    total_matches = 0  # 总匹配次数

    data_iter = iter(data_loader)
    pbar = tqdm(total=CONFIG["total_frames"], desc="[推理-训练循环] 处理帧")

    while processed_frames < CONFIG["total_frames"]:
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            sample = next(data_iter)

        frames = split_sample_into_frames(sample)

        for frame in frames:
            if processed_frames >= CONFIG["total_frames"]:
                break

            # 1. 提取当前帧数据
            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = frame

            images = torch.stack([imgs1, imgs2], dim=2).to(device)
            future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
            driver = driver.to(device)
            future_driver = future_driver.to(device)
            future_driver_last = future_driver[:, -1, :]

            # 2. 特征嵌入
            with torch.no_grad():
                image_embedded = image_embed(images)
                motor_embedded = motor_embed(driver)
                future_image_embedded = image_embed(future_images)
                img_proj_future = img_sim_model(future_image_embedded)

            # 3. 生成候选动作
            generator_output = policy_gen(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=CONFIG["num_candidates"],
                temperature=1.0
            )
            candidates = generator_output["candidates"]

            # 4. 选择偏好动作和拒绝动作
            preferred, rejected, _ = select_preferred_rejected(
                candidates=candidates,
                img_proj_future=img_proj_future,
                future_driver_last=future_driver_last,
                motor_embed=motor_embed,
                driver_sim_model=driver_sim_model,
                batch_size=batch_size
            )

            # 5. 获取模型最高概率动作
            candidates_squeezed = [cand.squeeze(1) for cand in candidates]
            candidates_tensor = torch.stack(candidates_squeezed).permute(1, 0, 2)
            highest_prob_action, _ = get_model_highest_prob_action(
                candidates=candidates_squeezed,
                candidates_tensor=candidates_tensor,
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                policy_gen=policy_gen
            )

            # 6. 判断preference match
            is_match = bool(torch.allclose(
                preferred,
                highest_prob_action,
                atol=CONFIG["action_match_tolerance"],
                rtol=0
            ))

            if is_match:
                total_matches += 1

            # 7. 更新滑动窗口并记录统计信息
            # 确保窗口始终保持最新的window_size帧
            match_window.append(1 if is_match else 0)
            while len(match_window) > window_size:
                match_window.pop(0)  # 移除最旧的元素

            # 窗口填满后才记录统计信息
            if len(match_window) == window_size:
                window_match_count = sum(match_window)
                current_window_rate = window_match_count / window_size
                total_rate = total_matches / (processed_frames + 1) if processed_frames > 0 else 0

                # 记录详细的窗口统计信息
                window_stats = {
                    "frame_idx": processed_frames,
                    "window_content": match_window.copy(),  # 记录窗口具体内容
                    "window_match_count": window_match_count,
                    "window_size": window_size,
                    "window_match_rate": current_window_rate,
                    "total_match_rate": total_rate
                }
                match_stats.append(window_stats)

                # 调试模式下打印窗口变化
                if CONFIG["debug_mode"]:
                    print(f"\n[窗口更新] 帧 {processed_frames}: 窗口内容 {match_window}, "
                          f"匹配率 {current_window_rate:.2%}")

            # 8. 检查动作是否重复
            current_action = preferred.squeeze(0)
            action_repeated, max_sim = is_action_repeated(current_action, history_actions)

            # 调试模式下打印动作重复信息
            if CONFIG["debug_mode"] and action_repeated:
                print(f"[动作重复] 帧 {processed_frames}: 最高相似度 {max_sim:.4f} "
                      f"(阈值 {CONFIG['repeat_threshold']})")

            # 如果动作重复，跳过训练
            if action_repeated:
                # 仍然将当前动作加入历史缓存，用于后续判断
                history_actions.append(current_action.detach())
                # 保持历史缓存大小
                while len(history_actions) > CONFIG["history_cache_size"]:
                    history_actions.pop(0)

                processed_frames += 1
                pbar.update(1)
                continue

            # 9. 执行训练步骤
            # 识别关键层并选择训练层
            key_layers = identify_key_layers(
                policy_generator=policy_gen,
                frame=frame,
                image_embed=image_embed,
                motor_embed=motor_embed
            )

            selected_layers = select_layers_for_training(
                policy_generator=policy_gen,
                key_layers=key_layers
            )

            # 计算损失并优化
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

            # 10. 更新历史动作缓存
            history_actions.append(current_action.detach())
            while len(history_actions) > CONFIG["history_cache_size"]:
                history_actions.pop(0)  # 保持缓存大小

            # 11. 累计训练统计
            total_loss += loss.item()
            optimized_count += 1

            # 12. 定期保存模型
            if (processed_frames + 1) % CONFIG["save_interval"] == 0:
                torch.save({
                    "frame": processed_frames + 1,
                    "policy_generator_state_dict": policy_gen.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": CONFIG,
                    "total_loss": total_loss,
                    "optimized_count": optimized_count,
                    "total_matches": total_matches,
                    "match_stats": match_stats
                }, f"{CONFIG['save_path']}_frame_{processed_frames + 1}.pth")
                print(f"\n[保存模型] 已保存第{processed_frames + 1}帧模型")

            # 13. 更新进度条
            processed_frames += 1
            # 显示窗口内容和统计信息
            window_content = f"窗口: {match_window}" if match_window else ""
            pbar.set_postfix({
                "DPO损失": f"{loss.item():.4f}",
                "优化帧数": optimized_count,
                "总匹配率": f"{(total_matches / processed_frames):.2%}" if processed_frames > 0 else "N/A",
                window_content: f"{sum(match_window)}/{len(match_window)}" if match_window else ""
            })
            pbar.update(1)

    pbar.close()
    avg_loss = total_loss / optimized_count if optimized_count > 0 else 0.0
    overall_match_rate = total_matches / processed_frames if processed_frames > 0 else 0.0

    # 保存最终模型和统计结果
    torch.save({
        "frame": processed_frames,
        "policy_generator_state_dict": policy_gen.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": CONFIG,
        "total_loss": total_loss,
        "optimized_count": optimized_count,
        "total_matches": total_matches,
        "match_stats": match_stats
    }, f"{CONFIG['save_path']}_final.pth")

    stats = {
        "total_frames": processed_frames,
        "optimized_frames": optimized_count,
        "total_loss": total_loss,
        "avg_loss": avg_loss,
        "total_matches": total_matches,
        "overall_match_rate": overall_match_rate,
        "match_stats": match_stats,
        "config_used": CONFIG  # 保存使用的配置参数
    }
    np.save(CONFIG["stats_path"], stats)

    return stats


# ---------------------- 主函数 ----------------------
def main():
    start_time = time.time()
    print("\n" + "=" * 60)
    print("          推理-训练循环模式：连续处理帧数据")
    print("=" * 60)
    print(f"[配置信息] 总帧数: {CONFIG['total_frames']}, 窗口大小: {CONFIG['window_size']}")
    print(f"[配置信息] 关键层参数: P={CONFIG['P']}, Q={CONFIG['Q']}")
    print(f"[配置信息] 动作重复阈值: {CONFIG['repeat_threshold']}, 历史缓存大小: {CONFIG['history_cache_size']}")
    if CONFIG["debug_mode"]:
        print("[注意] 调试模式已开启，将输出详细过程信息")

    try:
        image_embed, motor_embed, policy_generator, ref_generator, img_sim_model, driver_sim_model = load_pretrained_models(
            CONFIG["pretrained_model_path"]
        )
        data_loader = load_dataset()
    except Exception as e:
        print(f"[初始化失败] {str(e)}")
        return

    optimizer = torch.optim.Adam(
        params=policy_generator.parameters(),
        lr=CONFIG["lr"],
        weight_decay=1e-6
    )

    print("\n[开始循环] 进入推理-训练模式...")
    stats = inference_train_loop(
        data_loader=data_loader,
        policy_gen=policy_generator,
        ref_gen=ref_generator,
        optimizer=optimizer,
        motor_embed=motor_embed,
        image_embed=image_embed,
        img_sim_model=img_sim_model,
        driver_sim_model=driver_sim_model
    )

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("                          推理-训练循环完成")
    print("=" * 60)
    print(f"总耗时：{total_time:.2f}秒 | 总处理帧数：{stats['total_frames']}")
    print(f"优化帧数：{stats['optimized_frames']} | 平均损失：{stats['avg_loss']:.4f}")
    print(f"总匹配统计：{stats['total_matches']}/{stats['total_frames']}（{stats['overall_match_rate']:.2%}）")
    print(f"最终模型路径：{CONFIG['save_path']}_final.pth")
    print(f"统计信息路径：{CONFIG['stats_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
