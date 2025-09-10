import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights


# 复用之前定义的ImageEmbedding和MotorEmbedding
class ImageEmbedding(nn.Module):
    def __init__(self, embed_dim, c=256, num_layers=1, dropout_rate=0.5, is_resnet=False):
        super(ImageEmbedding, self).__init__()
        self.c = c
        self.is_resnet = is_resnet
        self.embed_dim = embed_dim

        if self.is_resnet:
            pass
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)
        else:
            self.cnn_layers = nn.ModuleList()
            self.residual_layers = nn.ModuleList()
            in_channels = 3
            out_channels = 8

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

                in_channels = out_channels
                out_channels *= 2

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
    def __init__(self, motor_dim=12, embed_dim=128, num_fc_layers=3, dropout_rate=0.2):
        super(MotorEmbedding, self).__init__()
        self.fc_layers = nn.ModuleList()
        in_dim = motor_dim
        hidden_dim = 64

        for _ in range(num_fc_layers - 1):
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
    def __init__(self, embed_dim, nhead, num_layers):
        super(TransformerEncoderModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim * 3, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)  # 返回最终层输出（简化，无需保存所有层）


class EncoderOnlyCandidateGenerator(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, motor_dim=12, max_seq_length=100):
        super().__init__()
        self.d_model = embed_dim * 3  # 融合后的特征维度
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder = TransformerEncoderModel(embed_dim, nhead, num_layers)

        # 输出层：预测1帧动作的分布参数（以连续动作为例，预测均值和标准差）
        self.fc_mean = nn.Linear(self.d_model, motor_dim)  # 动作均值
        self.fc_logvar = nn.Linear(self.d_model, motor_dim)  # 动作对数方差（便于计算标准差）
        # （如果是离散动作，可用nn.Linear(self.d_model, num_classes) + softmax）

    def forward(self, image_embedded, motor_embedded, num_candidates=5, temperature=0.5):
        """
        生成多个1帧动作候选
        :param image_embedded: 图像嵌入 (batch, seq, 2*embed_dim)
        :param motor_embedded: 历史电机嵌入 (batch, seq, embed_dim)
        :param num_candidates: 候选数量
        :param temperature: 温度参数（控制采样随机性，>0，越小越集中）
        :return: 候选动作列表 (num_candidates, batch, 1, motor_dim)
        """
        # 1. 融合输入特征
        combined = torch.cat([motor_embedded, image_embedded], dim=-1)  # (batch, seq, 3*embed_dim)
        combined = self.positional_encoding(combined)  # 加位置编码

        # 2. Encoder提取全局特征
        encoder_out = self.encoder(combined)  # (batch, seq, 3*embed_dim)
        global_feat = encoder_out.mean(dim=1)  # 取序列均值作为全局特征 (batch, 3*embed_dim)

        # 3. 预测动作的分布参数（均值+标准差）
        mean = self.fc_mean(global_feat)  # (batch, motor_dim)
        logvar = self.fc_logvar(global_feat)  # (batch, motor_dim)
        # 限制logvar范围，避免标准差过大/过小
        logvar = torch.clamp(logvar, min=-5, max=5)
        std = torch.exp(0.5 * logvar) * temperature  # 温度调整标准差（温度越高，随机性越强）

        # 4. 从高斯分布中多次采样，生成候选
        candidates = []
        for _ in range(num_candidates):
            # 采样：mean + std * 随机噪声
            eps = torch.randn_like(mean)  # 标准正态分布噪声
            sample = mean + std * eps  # (batch, motor_dim)
            candidates.append(sample.unsqueeze(1))  # 增加时间维度 (batch, 1, motor_dim)

        return candidates  # 列表长度为num_candidates

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


# 示例使用
if __name__ == "__main__":
    embed_dim = 64
    nhead = 4
    num_layers = 15
    motor_dim = 12
    max_seq_length = 10

    # 初始化模型
    image_embed = ImageEmbedding(embed_dim, num_layers=3, is_resnet=False)
    motor_embed = MotorEmbedding(motor_dim=motor_dim, embed_dim=embed_dim)
    generator = EncoderOnlyCandidateGenerator(
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=num_layers,
        motor_dim=motor_dim,
        max_seq_length=max_seq_length
    )

    # 构造输入
    batch_size = 2
    images = torch.randn(batch_size, max_seq_length, 2, 3, 256, 256)  # (batch, seq, 2相机, 3, H, W)
    motor_data = torch.randn(batch_size, max_seq_length, motor_dim)  # (batch, seq, motor_dim)

    # 生成候选动作
    image_embedded = image_embed(images)  # (batch, seq, 2*embed_dim)
    motor_embedded = motor_embed(motor_data)  # (batch, seq, embed_dim)
    for i in range(10):
        time_start = time.time()
        candidates = generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=5,  # 生成5个候选
            temperature=0.3  # 低温度：候选更集中；高温度：更发散
        )
        print(f"生成时间: {time.time() - time_start:.4f} 秒")
    # 输出结果形状
    print(f"生成候选数量：{len(candidates)}")
    print(f"单个候选形状：{candidates[0].shape}")  # (batch=2, 1, motor_dim=12)

    # 新增：打印模型大小
    image_size = calculate_model_size(image_embed)
    motor_size = calculate_model_size(motor_embed)
    generator_size = calculate_model_size(generator)
    total_size = image_size + motor_size + generator_size

    print("\n模型大小统计（MB）：")
    print(f"图像嵌入模块：{image_size:.2f} MB")
    print(f"电机嵌入模块：{motor_size:.2f} MB")
    print(f"生成器模块（含Transformer）：{generator_size:.2f} MB")
    print(f"总模型大小：{total_size:.2f} MB")

