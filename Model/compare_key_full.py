import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# ---------------------- 模型定义 ----------------------
class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 注册为非参数缓冲区

    def forward(self, x):
        """x: (seq_len, batch_size, d_model)"""
        x = x + self.pe[:x.size(0)]
        return x


class TransformerEncoderModel(nn.Module):
    def __init__(self, embed_dim=64, nhead=8, num_layers=16):
        super(TransformerEncoderModel, self).__init__()
        # 存储每一层的编码器
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim * 3, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        # 为每一层创建训练控制开关（布尔值张量，不需要梯度）
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
                current = layer(current)
            else:
                with torch.no_grad():
                    current = layer(current)

            layer_outputs.append(current)

        return layer_outputs


class EncoderOnlyCandidateGenerator(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, motor_dim=2, max_seq_length=100):
        super().__init__()
        self.d_model = embed_dim * 3  # 融合后的特征维度
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder = TransformerEncoderModel(embed_dim, nhead, num_layers)

        # 输出层
        self.fc_mean1 = nn.Linear(self.d_model, motor_dim)
        self.fc_logvar1 = nn.Linear(self.d_model, motor_dim)
        self.fc_mean2 = nn.Linear(self.d_model, motor_dim)
        self.fc_logvar2 = nn.Linear(self.d_model, motor_dim)

    def forward(self, image_embedded, motor_embedded):
        """生成动作候选（简化版，仅用于训练时间对比）"""
        # 融合输入特征
        combined = torch.cat([motor_embedded, image_embedded], dim=-1)
        combined = self.positional_encoding(combined)

        # Encoder提取特征
        encoder_out = self.encoder(combined)
        final_out = encoder_out[-1]
        global_feat = final_out.mean(dim=1)

        # 预测分布参数
        mean1 = self.fc_mean1(global_feat)
        logvar1 = self.fc_logvar1(global_feat)
        mean2 = self.fc_mean2(global_feat)
        logvar2 = self.fc_logvar2(global_feat)

        return {
            'mean1': mean1,
            'logvar1': logvar1,
            'mean2': mean2,
            'logvar2': logvar2
        }


# ---------------------- 关键层识别函数 ----------------------
def calculate_layer_similarity(layer_outputs):
    """计算每一层输出与其他层输出的余弦相似度"""
    num_layers = len(layer_outputs)
    similarity_matrix = torch.zeros(num_layers, num_layers, device=layer_outputs[0].device)

    for i in range(num_layers):
        # 展平特征维度并标准化
        output_i = layer_outputs[i].view(layer_outputs[i].shape[0], -1)
        output_i_norm = torch.nn.functional.normalize(output_i, dim=1)

        for j in range(num_layers):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue

            output_j = layer_outputs[j].view(layer_outputs[j].shape[0], -1)
            output_j_norm = torch.nn.functional.normalize(output_j, dim=1)

            # 计算平均余弦相似度
            sim = torch.nn.functional.cosine_similarity(output_i_norm, output_j_norm, dim=1).mean()
            similarity_matrix[i, j] = sim

    return similarity_matrix


def identify_key_layers(model, sample_image_emb, sample_motor_emb, P=3, threshold=0.8):
    """识别Transformer编码器中的关键层"""

    # 临时修改模型以获取层输出
    def temp_forward(image_embedded, motor_embedded):
        combined = torch.cat([motor_embedded, image_embedded], dim=-1)
        combined = model.positional_encoding(combined)
        encoder_out = model.encoder(combined)
        return encoder_out

    # 获取模型输出
    with torch.no_grad():
        layer_outputs = temp_forward(sample_image_emb, sample_motor_emb)
    num_layers = len(layer_outputs)

    # 计算层间相似度矩阵
    similarity_matrix = calculate_layer_similarity(layer_outputs)

    # 初始化关键层集合
    key_layers = set()

    # 最后一层直接设为关键层
    key_layers.add(num_layers - 1)

    # 识别关键层
    for i in range(num_layers):
        # 对于倒数P层，与所有后续层比较
        if i >= num_layers - P:
            next_layers = range(i + 1, num_layers)
            if not next_layers:  # 最后一层已经处理过
                continue

            # 计算与后续层的平均相似度
            avg_sim = torch.mean(similarity_matrix[i, list(next_layers)]).item()

            if avg_sim >= threshold:
                key_layers.add(i)
        else:
            # 对于其他层，与后面P层比较
            next_layers = range(i + 1, min(i + 1 + P, num_layers))
            if not next_layers:
                continue

            # 计算与后面P层的平均相似度
            avg_sim = torch.mean(similarity_matrix[i, list(next_layers)]).item()

            if avg_sim >= threshold:
                key_layers.add(i)

    return sorted(list(key_layers))


# ---------------------- 训练时间测量函数 ----------------------
def count_trainable_parameters(model):
    """计算模型中可训练的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_training_time(model, input_image, input_motor, iterations=100):
    """测量训练时间（前向+反向传播）"""
    model.train()
    # 获取可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 检查是否有可训练参数
    if not trainable_params:
        raise ValueError("没有可训练的参数，请检查模型配置")

    # 只优化可训练参数
    optimizer = torch.optim.Adam(trainable_params, lr=5e-6)
    start_time = time.time()

    for _ in tqdm(range(iterations), desc="测量训练时间"):
        optimizer.zero_grad()
        outputs = model(input_image, input_motor)
        # 简单的损失函数
        loss = outputs['mean1'].sum() + outputs['mean2'].sum()
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    avg_time = total_time / iterations
    steps_per_sec = iterations / total_time  # 每秒训练步骤数

    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'steps_per_sec': steps_per_sec
    }


def compare_training_strategies(embed_dim=128, nhead=8, num_layers=16,
                                batch_size=1, seq_length=30,
                                P=3, Q=5, threshold=0.8,
                                training_iter=100):
    """比较训练状态下全模型与关键层训练的性能差异"""
    print("=" * 70)
    print(f"训练时间对比 - 模型配置: 嵌入维度={embed_dim}, 注意力头数={nhead}, 层数={num_layers}")
    print(f"输入配置: 批次大小={batch_size}, 序列长度={seq_length}")
    print(f"关键层参数: P={P}, Q={Q}, 相似度阈值={threshold * 100}%")
    print("=" * 70)

    # 1. 创建模型和设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = EncoderOnlyCandidateGenerator(
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_length=seq_length
    ).to(device)

    # 创建模拟输入
    image_embedded = torch.randn(batch_size, seq_length, 2 * embed_dim, device=device)
    motor_embedded = torch.randn(batch_size, seq_length, embed_dim, device=device)

    # 2. 识别关键层
    print("\n识别关键层中...")
    key_layers = identify_key_layers(model, image_embedded, motor_embedded, P, threshold)
    print(f"识别到的关键层: {key_layers} (共{len(key_layers)}层)")

    # 从关键层中随机选择Q个层
    if len(key_layers) <= Q:
        selected_layers = key_layers
    else:
        selected_indices = np.random.choice(len(key_layers), Q, replace=False)
        selected_layers = [key_layers[i] for i in selected_indices]
    print(f"从关键层中选择的训练层: {selected_layers} (共{len(selected_layers)}层)")

    # 3. 全模型训练性能评估
    print("\n=== 全模型训练性能 ===")
    # 设置所有层可训练
    for i in range(num_layers):
        model.encoder.set_layer_trainable(i, True)

    # 只对浮点和复数类型的张量启用梯度
    for param in model.parameters():
        if param.dtype.is_floating_point or param.dtype.is_complex:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 检查全模型是否有可训练参数
    full_train_params = count_trainable_parameters(model)
    if full_train_params == 0:
        print("警告: 全模型没有可训练参数，请检查模型定义")
        return None

    full_train_stats = measure_training_time(
        model, image_embedded, motor_embedded, training_iter
    )
    print(f"可训练参数: {full_train_params:,}")
    print(f"总训练时间 ({training_iter}步): {full_train_stats['total_time']:.4f}秒")
    print(f"单步训练平均时间: {full_train_stats['avg_time']:.6f}秒")
    print(f"每秒训练步数: {full_train_stats['steps_per_sec']:.2f}")

    # 4. 关键层训练性能评估
    print("\n=== 关键层训练性能 ===")
    # 重置所有参数的requires_grad状态
    for param in model.parameters():
        param.requires_grad = False

    # 只设置选中的层可训练
    for layer_idx in selected_layers:
        if 0 <= layer_idx < len(model.encoder.layers):
            # 为选中层的所有参数启用梯度
            for param in model.encoder.layers[layer_idx].parameters():
                if param.dtype.is_floating_point or param.dtype.is_complex:
                    param.requires_grad = True

    # 确保输出层始终可训练（可选，根据需求调整）
    for param in model.fc_mean1.parameters():
        param.requires_grad = True
    for param in model.fc_logvar1.parameters():
        param.requires_grad = True
    for param in model.fc_mean2.parameters():
        param.requires_grad = True
    for param in model.fc_logvar2.parameters():
        param.requires_grad = True

    # 检查关键层模型是否有可训练参数
    key_train_params = count_trainable_parameters(model)
    if key_train_params == 0:
        print("警告: 关键层模型没有可训练参数，将使用全模型参数")
        # 回退到全模型训练
        for param in model.parameters():
            if param.dtype.is_floating_point or param.dtype.is_complex:
                param.requires_grad = True
        key_train_params = count_trainable_parameters(model)

    key_train_stats = measure_training_time(
        model, image_embedded, motor_embedded, training_iter
    )
    print(f"可训练参数: {key_train_params:,}")
    print(f"总训练时间 ({training_iter}步): {key_train_stats['total_time']:.4f}秒")
    print(f"单步训练平均时间: {key_train_stats['avg_time']:.6f}秒")
    print(f"每秒训练步数: {key_train_stats['steps_per_sec']:.2f}")

    # 5. 计算性能提升比例
    print("\n=== 性能提升对比 ===")
    param_reduction = (1 - key_train_params / full_train_params) * 100
    time_reduction = (1 - key_train_stats['avg_time'] / full_train_stats['avg_time']) * 100
    speedup = key_train_stats['steps_per_sec'] / full_train_stats['steps_per_sec']

    print(f"可训练参数减少: {param_reduction:.2f}%")
    print(f"单步训练时间减少: {time_reduction:.2f}%")
    print(f"训练速度提升: {speedup:.2f}x")

    return {
        'full_training': full_train_stats,
        'full_train_params': full_train_params,
        'key_training': key_train_stats,
        'key_train_params': key_train_params,
        'key_layers': key_layers,
        'selected_layers': selected_layers,
        'param_reduction': param_reduction,
        'time_reduction': time_reduction,
        'speedup': speedup
    }


# ---------------------- 主函数 ----------------------
def main():
    # 配置参数
    config = {
        "embed_dim": 128,
        "nhead": 8,
        "num_layers": 16,
        "batch_size": 1,  # 实时输入
        "seq_length": 30,
        "P": 3,
        "Q": 5,
        "threshold": 0.8,
        "training_iter": 200  # 增加迭代次数以获得更稳定的时间测量
    }

    # 运行训练时间对比
    print(f"{'#' * 20} 训练时间对比实验 {'#' * 20}")
    result = compare_training_strategies(**config)

    # 汇总结果
    if result:
        print("\n\n" + "=" * 70)
        print("训练时间对比总结")
        print("=" * 70)
        print(f"可训练参数减少: {result['param_reduction']:.2f}%")
        print(f"单步训练时间减少: {result['time_reduction']:.2f}%")
        print(f"训练速度提升: {result['speedup']:.2f}x")
        print(f"关键层数量: {len(result['key_layers'])} (从中选择{len(result['selected_layers'])}层训练)")
        print("=" * 70)


if __name__ == "__main__":
    main()
