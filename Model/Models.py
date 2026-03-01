import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

# 路径配置（确保导入正常）
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)


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
            out_channels = 16  # 固定输出通道，避免维度膨胀

            for i in range(num_layers):
                # 卷积块（3层卷积+BN+ReLU）
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

                # 残差连接（通道不匹配时用1x1卷积调整）
                if in_channels != out_channels:
                    self.residual_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
                else:
                    self.residual_layers.append(None)

                # 下采样+ dropout
                self.cnn_layers.append(nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(dropout_rate)
                ))

                in_channels = out_channels  # 更新输入通道

            # 全连接层输入维度计算（根据下采样次数）
            h, w = c // (2 ** num_layers), c // (2 ** num_layers)
            self.fc = nn.Linear(in_channels * h * w, embed_dim)

    def forward(self, images):
        # 输入形状：(batch, seq, num_cameras, 3, H, W) → 展平为2D图像用于CNN
        batch_size, seq_length, num_cameras, channels, H, W = images.shape
        images_flat = images.view(-1, channels, H, W)  # (batch*seq*num_cameras, 3, H, W)

        # 统一图像尺寸到指定大小
        images_flat = F.interpolate(images_flat, size=(self.c, self.c), mode='bilinear', align_corners=False)

        if not self.is_resnet:
            out = images_flat
            for i in range(len(self.residual_layers)):
                # 残差连接计算
                residual = out
                out = self.cnn_layers[2 * i](out)  # 卷积块
                if self.residual_layers[i] is not None:
                    residual = self.residual_layers[i](residual)
                out = out + residual  # 残差相加
                out = self.cnn_layers[2 * i + 1](out)  # 下采样+dropout
            # 展平后过全连接层
            out_flat = out.view(batch_size * seq_length * num_cameras, -1)
            embedded = self.fc(out_flat)
        else:
            # ResNet直接提取特征
            embedded = self.resnet(images_flat)

        # 恢复维度：(batch, seq, num_cameras*embed_dim) → 双摄像头时为2*embed_dim
        embedded = embedded.view(batch_size, seq_length, num_cameras, -1)
        embedded = embedded.view(batch_size, seq_length, -1)  # 最终形状：(batch, seq, 2*embed_dim)
        return embedded


class MotorEmbedding(nn.Module):
    def __init__(self, motor_dim=2, embed_dim=64, num_fc_layers=3, dropout_rate=0.2):
        super(MotorEmbedding, self).__init__()
        self.fc_layers = nn.ModuleList()
        in_dim = motor_dim
        hidden_dim = 16  # 隐藏层维度

        # 堆叠全连接层
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(in_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        # 输出层（映射到目标嵌入维度）
        self.fc_layers.append(nn.Linear(in_dim, embed_dim))

    def forward(self, motor_data):
        # 输入形状支持：(batch, motor_dim) 或 (batch, seq, motor_dim)
        out = motor_data
        for layer in self.fc_layers:
            out = layer(out)
        return out  # 输出形状：(batch, embed_dim) 或 (batch, seq, embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 预计算位置编码（注册为缓冲区，不参与训练）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # 偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # 形状：(1, max_len, d_model)

    def forward(self, x):
        # x形状：(batch, seq_len, d_model)
        seq_length = x.size(1)
        # 取对应长度的位置编码（无需repeat，广播机制自动匹配batch）
        pe = self.pe[:, :seq_length, :]
        x = x + pe  # 位置编码与输入相加
        return x


class TransformerEncoderModel(nn.Module):
    def __init__(self, embed_dim=64, nhead=8, num_layers=16):
        super(TransformerEncoderModel, self).__init__()
        self.d_model = embed_dim * 3  # 固定：motor_embed(1*dim) + image_embed(2*dim)
        # 堆叠Transformer编码器层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        # 层训练开关（默认全部参与训练）
        self.layer_trainable = nn.Parameter(torch.ones(num_layers, dtype=torch.bool), requires_grad=False)

    def set_layer_trainable(self, layer_idx, trainable):
        """设置指定层是否参与训练（边界检查）"""
        if 0 <= layer_idx < len(self.layers):
            self.layer_trainable[layer_idx] = trainable

    def forward(self, src):
        # src形状：(batch, seq_len, d_model)
        layer_outputs = []
        current = src

        for i, layer in enumerate(self.layers):
            # 根据开关决定是否计算梯度
            if self.layer_trainable[i]:
                current = layer(current)
            else:
                with torch.no_grad():
                    current = layer(current)
            layer_outputs.append(current)  # 保存每一层的输出

        return layer_outputs  # 返回所有层输出，最后一个为最终输出


class EncoderOnlyCandidateGenerator(nn.Module):
    """
    修正核心：只生成一组动作参数（对应2个电机的动作值）
    输出：(batch, motor_dim=2) 的均值/标准差 + 多个候选动作
    """

    def __init__(self, embed_dim, nhead, num_layers, motor_dim=2, max_seq_length=100):
        super().__init__()
        self.d_model = embed_dim * 3  # motor_embed(1*dim) + image_embed(2*dim)
        self.motor_dim = motor_dim  # 固定为2（两个电机）
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder = TransformerEncoderModel(embed_dim, nhead, num_layers)

        # 【核心修改】只保留一组分布参数输出层（删除mean2/std2相关）
        self.fc_mean = nn.Linear(self.d_model, motor_dim)  # 动作均值预测（2维）
        self.fc_logvar = nn.Linear(self.d_model, motor_dim)  # 动作对数方差预测（2维）

    def forward(self, image_embedded, motor_embedded, num_candidates=5, temperature=0.5):
        """
        输入：
            image_embedded: (batch, seq, 2*embed_dim) → 双摄像头图像嵌入
            motor_embedded: (batch, seq, embed_dim)   → 历史电机动作嵌入
        输出：
            dict: 包含候选动作列表、均值、标准差
        """
        # 1. 融合图像和电机特征（维度：3*embed_dim）
        combined = torch.cat([motor_embedded, image_embedded], dim=-1)  # (batch, seq, 3*embed_dim)
        combined = self.positional_encoding(combined)  # 加入位置编码

        # 2. Transformer编码器提取全局特征
        encoder_out_list = self.encoder(combined)  # 所有层输出列表
        encoder_final_out = encoder_out_list[-1]  # 最后一层输出（最终特征）
        global_feat = encoder_final_out.mean(dim=1)  # 序列维度取平均 → (batch, 3*embed_dim)

        # 3. 预测动作分布参数（均值+标准差）
        mean = self.fc_mean(global_feat)  # (batch, motor_dim=2) → 两个电机的均值
        logvar = self.fc_logvar(global_feat)  # (batch, motor_dim=2) → 两个电机的对数方差

        # 限制对数方差范围，避免标准差过大/过小
        logvar = torch.clamp(logvar, min=-5, max=5)
        std = torch.exp(0.5 * logvar) * temperature  # 标准差（温度调整随机性）

        # 4. 生成多个候选动作（从高斯分布采样）
        candidates = []
        for _ in range(num_candidates):
            eps = torch.randn_like(mean)  # 标准正态噪声 → (batch, 2)
            sample = mean + std * eps  # 重参数化采样
            sample = torch.tanh(sample)  # 限制动作范围到[-1,1]
            sample = sample.unsqueeze(1)  # 增加时间维度 → (batch, 1, 2)
            candidates.append(sample)

        # 【核心修改】只返回一组参数（删除candidates2/mean2/std2）
        return {
            'candidates': candidates,  # 候选动作列表：[ (batch,1,2), ... ]
            'mean': mean,  # 动作均值：(batch, 2)
            'std': std  # 动作标准差：(batch, 2)
        }


class SimilarityModelImage(nn.Module):
    """处理连续帧图像嵌入的投射模型（输入已拼接双摄像头特征）"""

    def __init__(self, embed_dim=128, num_frames=30, num_layers=3, nhead=4, similarity_dim=128):
        super(SimilarityModelImage, self).__init__()
        self.input_dim = embed_dim * 2  # 双摄像头图像嵌入维度（2*embed_dim）
        self.num_frames = num_frames
        self.positional_encoding = PositionalEncoding(self.input_dim, max_len=num_frames)

        # Transformer编码器（提取时序特征）
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers
        )

        # 全连接层（映射到相似度空间）
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, similarity_dim)
        )

    def forward(self, imgs_embedding):
        """
        输入：imgs_embedding → (batch, num_frames, 2*embed_dim)（双摄像头连续帧嵌入）
        输出：image_proj → (batch, similarity_dim)（投射到相似度空间的特征）
        """
        # 加入位置编码
        imgs_embedding_pe = self.positional_encoding(imgs_embedding)
        # Transformer编码时序特征
        transformer_out = self.transformer_encoder(imgs_embedding_pe)  # (batch, num_frames, 2*embed_dim)
        # 取最后一帧特征作为全局表示
        final_feat = transformer_out[:, -1, :]  # (batch, 2*embed_dim)
        # 投射到相似度空间
        return self.fc(final_feat)


class SimilarityModelDriver(nn.Module):
    """处理单帧电机动作嵌入的投射模型"""

    def __init__(self, embed_dim=64, similarity_dim=128):
        super(SimilarityModelDriver, self).__init__()
        # 全连接层（映射到相似度空间）
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, similarity_dim)
        )

    def forward(self, driver_embedding):
        """
        输入：driver_embedding → (batch, embed_dim)（单帧电机动作嵌入）
        输出：driver_proj → (batch, similarity_dim)（投射到相似度空间的特征）
        """
        return self.fc(driver_embedding)


class JudgeModelImage(nn.Module):
    """基于图像序列嵌入的判断模型（输出图像序列的评分/判断结果）"""
    def __init__(self, embed_dim=128, num_frames=30, num_layers=3, nhead=4, judge_dim=32):
        super(JudgeModelImage, self).__init__()
        self.input_dim = embed_dim * 2  # 双摄像头图像嵌入维度（2*embed_dim=256）
        self.num_frames = num_frames
        self.judge_dim = judge_dim  # 匹配训练配置的32维
        self.positional_encoding = PositionalEncoding(self.input_dim, max_len=num_frames)

        # Transformer编码器提取时序特征
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers
        )

        # 评分预测全连接层（新增投影层，将256维→32维）
        self.fc_judge = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),  # 256→128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.input_dim // 2, self.judge_dim),  # 128→32
            nn.Sigmoid()  # 输出归一化到[0,1]
        )

    def forward(self, imgs_embedding):
        """
        输入：imgs_embedding → (batch, num_frames, 2*embed_dim)（256维图像嵌入）
        输出：judge_score → (batch, judge_dim)（32维判断分数）
        """
        # 加入位置编码（维度256，匹配输入）
        imgs_embedding_pe = self.positional_encoding(imgs_embedding)
        # Transformer编码时序特征
        transformer_out = self.transformer_encoder(imgs_embedding_pe)  # (batch, 30, 256)
        # 取最后一帧特征作为全局表示
        final_feat = transformer_out[:, -1, :]  # (batch, 256)
        # 投影到32维判断空间
        judge_score = self.fc_judge(final_feat)  # (batch, 32)
        return judge_score


class JudgeModelDriver(nn.Module):
    """基于电机动作嵌入的判断模型（输出电机动作的评分/判断结果）"""
    def __init__(self, embed_dim=128, judge_dim=32):
        super(JudgeModelDriver, self).__init__()
        self.judge_dim = judge_dim  # 匹配训练配置的32维
        # 评分预测全连接层（将128维电机嵌入→32维判断空间）
        self.fc_judge = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),  # 128→64
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, self.judge_dim),  # 64→32
            nn.Sigmoid()  # 输出归一化到[0,1]
        )

    def forward(self, driver_embedding):
        """
        输入：driver_embedding → (batch, embed_dim)（128维电机嵌入）
        输出：judge_score → (batch, judge_dim)（32维判断分数）
        """
        judge_score = self.fc_judge(driver_embedding)  # (batch, 32)
        return judge_score


class JudgeModel(nn.Module):
    """重构：融合单张图像特征和多个电机特征，输出匹配分数（适配训练代码调用）"""
    def __init__(self, embed_dim=128, num_frames=30, num_layers=3, nhead=4, judge_dim=32):
        super(JudgeModel, self).__init__()
        self.judge_dim = judge_dim
        # 融合层：计算图像特征与每个电机特征的匹配分数
        self.fc_fusion = nn.Sequential(
            nn.Linear(judge_dim * 2, judge_dim),  # 32*2=64 → 32
            nn.ReLU(),
            nn.Linear(judge_dim, 1),  # 32 → 1（单维度匹配分数）
            nn.Sigmoid()
        )

    def forward(self, img_judge_feat, driver_judge_feats):
        """
        输入：
            img_judge_feat: (batch, judge_dim) → JudgeModelImage的输出（32维）
            driver_judge_feats: list → 多个JudgeModelDriver的输出，每个元素是(batch, judge_dim)
        输出：
            match_scores: (batch, num_candidates+1) → 每个电机动作与图像的匹配分数
        """
        match_scores = []
        for driver_feat in driver_judge_feats:
            # 拼接图像特征和单个电机特征
            fusion_feat = torch.cat([img_judge_feat, driver_feat], dim=-1)  # (batch, 64)
            # 计算匹配分数
            score = self.fc_fusion(fusion_feat)  # (batch, 1)
            match_scores.append(score)
        # 拼接所有分数 → (batch, num_candidates+1)
        match_scores = torch.cat(match_scores, dim=-1)
        return match_scores


class ActionExtract(nn.Module):
    """
    3层全连接网络：将JudgeModelImage的输出映射为电机信号预测值
    输入：JudgeModelImage的输出 (batch, judge_dim)
    输出：电机信号预测 (batch, motor_dim)，可与真实电机信号计算损失
    """
    def __init__(self, in_dim=32, hidden_dim=64, out_dim=2, dropout_rate=0.2):
        super(ActionExtract, self).__init__()
        # 修正输入维度为32（匹配JudgeModelImage的输出）
        self.fc_layers = nn.Sequential(
            # 第1层全连接（32→64）
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # 第2层全连接（64→64）
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # 第3层全连接（64→2）
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, judge_image_output):
        """
        前向传播：输入JudgeModelImage的输出，输出电机信号预测
        :param judge_image_output: (batch, judge_dim) → JudgeModelImage的输出（32维）
        :return: motor_pred: (batch, motor_dim) → 预测的电机信号
        """
        motor_pred = self.fc_layers(judge_image_output)
        # 限制电机信号范围到[-1,1]
        motor_pred = torch.tanh(motor_pred)
        return motor_pred


def calculate_model_size(model):
    """计算模型总参数量和缓冲区大小（单位：MB）"""
    # 参数量大小（字节）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # 缓冲区大小（字节，如位置编码pe）
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    # 转换为MB（1MB = 1024^2 字节）
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    return total_size_mb


# ------------------------------
# 示例使用（验证新增ActionExtract类功能）
# ------------------------------
if __name__ == "__main__":
    # 1. 配置参数
    batch_size = 1
    seq_length = 30  # 观测序列长度
    motor_dim = 2  # 两个电机（动作维度=2）
    embed_dim = 128  # 嵌入维度
    num_candidates = 5  # 生成候选动作数量
    judge_dim = 32  # JudgeModelImage输出维度（适配训练配置）
    action_extract_hidden = 64  # ActionExtract隐藏层维度

    # 2. 初始化模型
    image_embed = ImageEmbedding(embed_dim=embed_dim, num_layers=3, is_resnet=False)
    motor_embed = MotorEmbedding(motor_dim=motor_dim, embed_dim=embed_dim)
    candidate_generator = EncoderOnlyCandidateGenerator(
        embed_dim=embed_dim,
        nhead=8,
        num_layers=16,  # 简化层数便于测试
        motor_dim=motor_dim,
        max_seq_length=seq_length
    )
    img_sim_model = SimilarityModelImage(
        embed_dim=embed_dim,
        num_frames=seq_length,
        num_layers=2,
        nhead=4,
        similarity_dim=32
    )
    driver_sim_model = SimilarityModelDriver(
        embed_dim=embed_dim,
        similarity_dim=32
    )
    # 初始化判断模型
    judge_image_model = JudgeModelImage(
        embed_dim=embed_dim,
        num_frames=seq_length,
        num_layers=2,
        nhead=4,
        judge_dim=judge_dim
    )
    judge_driver_model = JudgeModelDriver(
        embed_dim=embed_dim,
        judge_dim=judge_dim
    )
    judge_total_model = JudgeModel(
        embed_dim=embed_dim,
        num_frames=seq_length,
        num_layers=2,
        nhead=4,
        judge_dim=judge_dim
    )
    # 初始化新增的ActionExtract模型
    action_extract_model = ActionExtract(
        in_dim=judge_dim,  # 输入维度=JudgeModelImage输出维度
        hidden_dim=action_extract_hidden,
        out_dim=motor_dim,  # 输出维度=电机信号维度
        dropout_rate=0.2
    )

    # 3. 构造输入数据
    # 图像输入：(batch, seq, num_cameras=2, 3, H=64, W=64)
    images = torch.randn(batch_size, seq_length, 2, 3, 64, 64)
    # 电机动作输入：(batch, seq, motor_dim=2)
    motor_data = torch.randn(batch_size, seq_length, motor_dim)
    # 未来图像输入（用于相似度/判断模型）：(batch, num_frames=seq_length, 2, 3, 64, 64)
    future_images = torch.randn(batch_size, seq_length, 2, 3, 64, 64)
    # 构造真实电机信号（用于对比）
    real_motor_signal = torch.randn(batch_size, motor_dim)  # (batch, 2)

    # 4. 模型前向传播（验证流程）
    # 4.1 图像和电机嵌入
    image_embedded = image_embed(images)  # (batch, seq, 2*embed_dim)
    motor_embedded = motor_embed(motor_data)  # (batch, seq, embed_dim)
    future_image_embedded = image_embed(future_images)  # (batch, seq, 2*embed_dim)
    # 取最后一帧电机嵌入用于判断/相似度模型
    last_motor_embedded = motor_embedded[:, -1, :]  # (batch, embed_dim)

    # 4.2 生成候选动作（验证修改后输出）
    gen_outputs = candidate_generator(
        image_embedded=image_embedded,
        motor_embedded=motor_embedded,
        num_candidates=num_candidates,
        temperature=0.5
    )

    # 4.3 相似度模型前向（验证维度匹配）
    img_proj = img_sim_model(future_image_embedded)  # (batch, 32)
    driver_proj = driver_sim_model(last_motor_embedded)  # (batch, 32)

    # 4.4 判断模型前向（验证新增类功能）
    img_judge_score = judge_image_model(future_image_embedded)  # (batch, judge_dim)
    driver_judge_score = judge_driver_model(last_motor_embedded)  # (batch, judge_dim)
    # 模拟多个电机特征列表（适配重构后的JudgeModel）
    judge_driver_feats = [driver_judge_score for _ in range(num_candidates+1)]
    match_scores = judge_total_model(img_judge_score, judge_driver_feats)  # (batch, num_candidates+1)

    # 4.5 ActionExtract前向（验证新增类）
    motor_pred = action_extract_model(img_judge_score)  # (batch, motor_dim)
    # 计算预测电机信号与真实信号的MSE损失（示例）
    mse_loss = F.mse_loss(motor_pred, real_motor_signal)

    # 5. 打印结果（验证修改正确性）
    print("=" * 60)
    print("1. 候选生成器输出检查（核心修改验证）")
    print(f"   - 候选动作数量：{len(gen_outputs['candidates'])}（应等于{num_candidates}）")
    print(f"   - 单个候选形状：{gen_outputs['candidates'][0].shape}（应是(batch,1,2)）")
    print(f"   - 动作均值形状：{gen_outputs['mean'].shape}（应是(batch,2)）")
    print(f"   - 动作标准差形状：{gen_outputs['std'].shape}（应是(batch,2)）")

    print("\n2. 相似度模型输出检查")
    print(f"   - 图像投射特征形状：{img_proj.shape}（应是(batch,32)）")
    print(f"   - 电机投射特征形状：{driver_proj.shape}（应是(batch,32)）")
    print(f"   - 余弦相似度：{F.cosine_similarity(img_proj, driver_proj, dim=1).tolist()}")

    print("\n3. 判断模型输出检查（新增类验证）")
    print(f"   - 图像单独判断分数形状：{img_judge_score.shape}（应是(batch,{judge_dim})）")
    print(f"   - 电机单独判断分数形状：{driver_judge_score.shape}（应是(batch,{judge_dim})）")
    print(f"   - 匹配分数形状：{match_scores.shape}（应是(batch,{num_candidates+1})）")
    print(f"   - 图像判断分数：{img_judge_score.squeeze()[0].item():.4f}（归一化到[0,1]）")
    print(f"   - 电机判断分数：{driver_judge_score.squeeze()[0].item():.4f}（归一化到[0,1]）")

    print("\n4. ActionExtract输出检查（新增类验证）")
    print(f"   - JudgeModelImage输出形状：{img_judge_score.shape}")
    print(f"   - ActionExtract预测电机信号形状：{motor_pred.shape}（应是(batch,{motor_dim})）")
    print(f"   - 预测电机信号值：{motor_pred.squeeze().tolist()}（归一化到[-1,1]）")
    print(f"   - 真实电机信号值：{real_motor_signal.squeeze().tolist()}")
    print(f"   - 预测与真实信号的MSE损失：{mse_loss.item():.4f}")

    print("\n5. 模型大小统计（MB）")
    print(f"   - 图像嵌入模型：{calculate_model_size(image_embed):.2f} MB")
    print(f"   - 电机嵌入模型：{calculate_model_size(motor_embed):.2f} MB")
    print(f"   - 候选生成模型：{calculate_model_size(candidate_generator):.2f} MB")
    print(f"   - 图像相似度模型：{calculate_model_size(img_sim_model):.2f} MB")
    print(f"   - 电机相似度模型：{calculate_model_size(driver_sim_model):.2f} MB")
    print(f"   - 图像判断模型：{calculate_model_size(judge_image_model):.2f} MB")
    print(f"   - 电机判断模型：{calculate_model_size(judge_driver_model):.2f} MB")
    print(f"   - 总判断模型：{calculate_model_size(judge_total_model):.2f} MB")
    print(f"   - ActionExtract模型：{calculate_model_size(action_extract_model):.2f} MB")
    print("=" * 60)
