import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights


# 复用之前定义的ImageEmbedding和MotorEmbedding
class ImageEmbedding(nn.Module):
    def __init__(self, embed_dim=64, c=256, num_layers=3, dropout_rate=0.5, is_resnet=False):
        super(ImageEmbedding, self).__init__()
        self.c = c
        self.is_resnet = is_resnet
        self.embed_dim = embed_dim

        if self.is_resnet:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)
        else:
            self.cnn_layers = nn.ModuleList()
            self.residual_layers = nn.ModuleList()
            in_channels = 3
            out_channels = 16  # 初始输出通道设为16

            for i in range(num_layers):
                self.cnn_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ))

                if in_channels != out_channels:
                    self.residual_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
                else:
                    self.residual_layers.append(None)

                self.cnn_layers.append(nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(dropout_rate)
                ))

                in_channels = out_channels  # 不再将out_channels翻倍，保持16不变

            h, w = c // (2 ** num_layers), c // (2 ** num_layers)
            self.fc = nn.Linear(in_channels * h * w, embed_dim)

    def forward(self, images):
        batch_size, seq_length, num_cameras, channels, H, W = images.shape
        images = images.view(-1, channels, H, W)
        images = F.interpolate(images, size=(self.c, self.c), mode='bilinear', align_corners=False)

        out = images
        if not self.is_resnet:
            for i in range(len(self.residual_layers)):
                residual = out
                out = self.cnn_layers[2 * i](out)
                if self.residual_layers[i] is not None:
                    residual = self.residual_layers[i](residual)
                out += residual
                out = self.cnn_layers[2 * i + 1](out)
            out = out.view(batch_size * seq_length * num_cameras, -1)
            embedded = self.fc(out)
        else:
            embedded = self.resnet(out)

        embedded = embedded.view(batch_size, seq_length, num_cameras, -1)
        embedded = embedded.view(batch_size, seq_length, -1)  # (batch, seq, 2*embed_dim)
        return embedded


class MotorEmbedding(nn.Module):
    def __init__(self, motor_dim=2, embed_dim=64, num_fc_layers=3, dropout_rate=0.2):
        super(MotorEmbedding, self).__init__()
        self.fc_layers = nn.ModuleList()
        in_dim = motor_dim
        hidden_dim = 16

        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(in_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.fc_layers.append(nn.Linear(in_dim, embed_dim))

    def forward(self, motor_data):
        for layer in self.fc_layers:
            motor_data = layer(motor_data)
        return motor_data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_length = x.size(1)
        pe = self.pe[:, :seq_length, :].repeat(x.size(0), 1, 1)
        x = x + pe
        return x

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
        encoder_out = self.encoder(combined)  # (batch, seq, 3*embed_dim)
        encoder_out = encoder_out[-1]
        global_feat = encoder_out.mean(dim=1)  # 取序列均值作为全局特征 (batch, 3*embed_dim)

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

        # 返回两组候选以及它们的均值和标准差
        return {
            'candidates1': candidates1,
            'mean1': mean1,
            'std1': std1,
            'candidates2': candidates2,
            'mean2': mean2,
            'std2': std2
        }

# 新增：计算模型大小的函数
def calculate_model_size(model):
    """计算模型参数和缓冲区的总大小（MB）"""
    # 计算参数大小
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # 计算缓冲区大小（如positional encoding中的pe）
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    # 转换为MB（1MB = 1024^2字节）
    total_size = (param_size + buffer_size) / (1024 ** 2)
    return total_size


class SimilarityModelImage(nn.Module):
    """
    处理连续几帧的两张图像embedding的投射模型
    使用Transformer encoder + 全连接层
    """

    def __init__(self, embed_dim=32, num_frames=30, num_layers=3, nhead=4, similarity_dim=128):
        super(SimilarityModelImage, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # Transformer编码器层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers
        )

        # 全连接层，将输出映射到相似度空间
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, similarity_dim)
        )

    def forward(self, img_embedding1, img_embedding2):
        """
        输入: 两个连续帧序列的图像嵌入
        img_embedding1: (batch_size, num_frames, embed_dim)
        img_embedding2: (batch_size, num_frames, embed_dim)
        输出: 投射后的特征向量 (batch_size, similarity_dim)
        """
        # 将两个图像嵌入序列拼接
        combined = torch.cat([img_embedding1, img_embedding2], dim=1)  # (batch, 2*num_frames, embed_dim)

        # 通过Transformer编码器
        transformer_out = self.transformer_encoder(combined)  # (batch, 2*num_frames, embed_dim)

        # 取最后一个时间步的输出作为特征
        feature = transformer_out[:, -1, :]  # (batch, embed_dim)

        # 通过全连接层投射到相似度空间
        return self.fc(feature)  # (batch, similarity_dim)


class SimilarityModelDriver(nn.Module):
    """
    处理单独一帧的driver embedding的投射模型
    仅使用全连接层
    """

    def __init__(self, embed_dim=32, similarity_dim=128):
        super(SimilarityModelDriver, self).__init__()
        # 全连接层，将输出映射到相似度空间
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, similarity_dim)
        )

    def forward(self, driver_embedding):
        """
        输入: 单帧的驾驶员嵌入 (batch_size, embed_dim)
        输出: 投射后的特征向量 (batch_size, similarity_dim)
        """
        return self.fc(driver_embedding)  # (batch, similarity_dim)

# 示例使用
if __name__ == "__main__":
    embed_dim = 128
    nhead = 8
    num_layers = 16
    motor_dim = 2
    seq_length = 30

    # 初始化模型
    image_embed = ImageEmbedding(embed_dim, num_layers=3, is_resnet=False)
    motor_embed = MotorEmbedding(motor_dim=motor_dim, embed_dim=embed_dim)
    generator = EncoderOnlyCandidateGenerator(
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=num_layers,
        motor_dim=motor_dim,
        max_seq_length=seq_length
    )

    # 构造输入
    batch_size = 1
    images = torch.randn(batch_size, seq_length, 2, 3, 256, 256)  # (batch, seq, 2相机, 3, H, W)
    motor_data = torch.randn(batch_size, seq_length, motor_dim)  # (batch, seq, motor_dim)

    # 生成候选动作
    image_embedded = image_embed(images)  # (batch, seq, 2*embed_dim)
    motor_embedded = motor_embed(motor_data)  # (batch, seq, embed_dim)
    for i in range(10):
        time_start = time.time()
        outputs = generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=5,  # 生成5个候选
            temperature=0.3  # 低温度：候选更集中；高温度：更发散
        )
        print(f"生成时间: {time.time() - time_start:.4f} 秒")
    # 输出结果形状
    print(f"生成候选数量：{len(outputs['candidates1'])}")
    print(f"单个候选形状：{outputs['candidates1'][0].shape}")  # (batch=2, 1, motor_dim=12)

    # 新增：打印模型大小
    image_size = calculate_model_size(image_embed)
    motor_size = calculate_model_size(motor_embed)
    generator_size = calculate_model_size(generator)
    total_size = image_size + motor_size + generator_size

    print("\n模型大小统计（MB）：")
    print(f"图像嵌入模块：{image_size:.2f} MB")
    print(f"电机嵌入模块：{motor_size:.2f} MB")
    print(f"Transformer模块：{generator_size:.2f} MB")
    print(f"总模型大小：{total_size:.2f} MB")


    # similarity model test
    # 配置参数
    batch_size = 1
    embed_dim = 32
    num_frames = 15  # 连续帧数
    similarity_dim = 32  # 投射后的维度

    # 创建模型实例
    img_sim_model = SimilarityModelImage(
        embed_dim=embed_dim,
        num_frames=num_frames,
        num_layers=3,
        nhead=4,
        similarity_dim=similarity_dim
    )

    driver_sim_model = SimilarityModelDriver(
        embed_dim=embed_dim,
        similarity_dim=similarity_dim
    )

    # 创建随机输入数据
    # 连续帧图像嵌入1: (batch, num_frames, embed_dim)
    img_emb1 = torch.randn(batch_size, num_frames, embed_dim)
    # 连续帧图像嵌入2: (batch, num_frames, embed_dim)
    img_emb2 = torch.randn(batch_size, num_frames, embed_dim)
    # 驾驶员嵌入: (batch, embed_dim)
    driver_emb = torch.randn(batch_size, embed_dim)

    # 模型前向传播
    img_proj = img_sim_model(img_emb1, img_emb2)  # (batch, similarity_dim)
    driver_proj = driver_sim_model(driver_emb)  # (batch, similarity_dim)

    # 计算余弦相似度
    cos_sim = F.cosine_similarity(img_proj, driver_proj, dim=1)

    # 输出结果信息
    print(f"图像投射输出形状: {img_proj.shape}")
    print(f"电机投射输出形状: {driver_proj.shape}")
    print(f"余弦相似度输出形状: {cos_sim.shape}")
    print(f"余弦相似度值范围: [{cos_sim.min():.4f}, {cos_sim.max():.4f}]")

    # 新增：打印模型大小
    img_sim_model_size = calculate_model_size(img_sim_model)
    driver_sim_model_size = calculate_model_size(driver_sim_model)

    print("\n相似度模型大小统计（MB）：")
    print(f"图像相似度模型：{img_sim_model_size:.2f} MB")
    print(f"驾驶员相似度模型：{driver_sim_model_size:.2f} MB")
    print(f"总相似度模型大小：{(img_sim_model_size + driver_sim_model_size):.2f} MB")