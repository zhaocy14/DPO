import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 路径配置
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

# 导入模型和数据集类
from DataModule.DataModule import CombinedDataset
from Model.Models import (
    ImageEmbedding,
    MotorEmbedding,
    EncoderOnlyCandidateGenerator,
    SimilarityModelImage,
    SimilarityModelDriver,
    calculate_model_size
)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置参数
config = {
    # 训练参数
    "batch_size": 4,
    "epochs": 100,
    "lr": 1e-5,
    "sampling_workers": 20,
    "reverse_mse_weight": 1.0,  # 新增：反向MSE损失权重

    # 生成器模型参数
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,  # 两个电机，动作维度为2
    "gen_seq_len": 15,  # 观测序列长度

    # 相似度模型参数
    "sim_seq_len": 15,  # 预测序列长度
    "embed_dim_sim": 128,
    "num_layers_sim": 3,
    "nhead_sim": 4,
    "similarity_dim": 32,

    # 数据和模型路径
    "data_root_dirs": '/data/cyzhao/collector_cydpo',
    "save_path": "./saved_models",
    "loss_data_path": "./loss_records"
}

# 创建保存目录
os.makedirs(config["save_path"], exist_ok=True)
os.makedirs(config["loss_data_path"], exist_ok=True)

# 初始化模型
image_embed = ImageEmbedding(
    embed_dim=config["embed_dim_gen"],
    num_layers=3,
    is_resnet=False
).to(device)

motor_embed = MotorEmbedding(
    motor_dim=config["motor_dim"],
    embed_dim=config["embed_dim_gen"]
).to(device)

candidate_generator = EncoderOnlyCandidateGenerator(
    embed_dim=config["embed_dim_gen"],
    nhead=config["nhead_gen"],
    num_layers=config["num_layers_gen"],
    motor_dim=config["motor_dim"],
    max_seq_length=config["gen_seq_len"]
).to(device)

# 【修改1】初始化img_sim_model时补充motor_dim参数（适配修改后的模型）
img_sim_model = SimilarityModelImage(
    embed_dim=config['embed_dim_sim'],
    num_frames=config['sim_seq_len'],
    num_layers=config['num_layers_sim'],
    nhead=config['nhead_sim'],
    similarity_dim=config['similarity_dim'],
    motor_dim=config['motor_dim'],  # 新增：指定电机维度
    dropout_rate=0.2  # 新增：dropout参数（与模型定义一致）
).to(device)

driver_sim_model = SimilarityModelDriver(
    embed_dim=config['embed_dim_sim'],
    similarity_dim=config['similarity_dim'],
).to(device)

# 加载数据集
data_root = config["data_root_dirs"]
data_dir_list = [
    os.path.join(data_root, file)
    for file in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, file)) and "2025" in file
]

all_dataset = CombinedDataset(
    dir_list=data_dir_list,
    frame_len=config["gen_seq_len"],
    predict_len=config['sim_seq_len'],
    show=True
)

train_dataset = all_dataset.training_dataset
val_dataset = all_dataset.val_dataset
print(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")

# 数据加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['sampling_workers']
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['sampling_workers']
)

# 优化器和学习率调度器
optimizer = torch.optim.Adam(
    params=[
        {'params': image_embed.parameters()},
        {'params': motor_embed.parameters()},
        {'params': candidate_generator.parameters()},
        {'params': img_sim_model.parameters()},
        {'params': driver_sim_model.parameters()}
    ],
    lr=config['lr']
)
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


def nll_loss(mean, std, target):
    """
    计算高斯分布的负对数似然损失
    :param mean: 预测均值 (batch_size, motor_dim=2)
    :param std: 预测标准差 (batch_size, motor_dim=2)
    :param target: 真实动作 (batch_size, motor_dim=2)
    :return: 平均损失值
    """
    eps = 1e-6
    std = std + eps  # 避免log(0)

    # 确保所有张量形状匹配
    nll = torch.log(std) + (target - mean) ** 2 / (2 * std ** 2)
    return nll.mean()  # 对batch和电机维度取平均


def info_ce_loss(img_proj, candidate_projections, temperature=0.1):
    """
    计算Info Noise Contrastive Estimation Loss
    :param img_proj: 未来图像序列投影 (batch, similarity_dim)
    :param candidate_projections: 候选动作投影列表，第一个为正样本
    :return: InfoCE损失
    """
    batch_size = img_proj.shape[0]
    candidates = torch.stack(candidate_projections, dim=1)  # (batch, num_candidates, similarity_dim)

    img_proj = F.normalize(img_proj, p=2, dim=-1)
    candidates = F.normalize(candidates, p=2, dim=-1)
    # 计算图像投影与候选动作投影的余弦相似度
    similarities = F.cosine_similarity(
        img_proj.unsqueeze(1),  # (batch, 1, similarity_dim)
        candidates,  # (batch, num_candidates, similarity_dim)
        dim=2
    )  # (batch, num_candidates)

    # 温度缩放 + 交叉熵损失（第一个候选为正样本）
    similarities = similarities / temperature
    loss = F.cross_entropy(
        similarities,
        torch.zeros(batch_size, dtype=torch.long, device=img_proj.device)
    )
    return loss


# 【新增】定义反向MSE损失函数
def reverse_mse_loss(pred, target, weight=1.0, eps=1e-6):
    """
    反向MSE损失：让预测值与真实值尽可能不相似（通过最小化负的MSE实现）
    :param pred: 预测动作 (batch, motor_dim)
    :param target: 真实动作 (batch, motor_dim)
    :param weight: 损失权重
    :param eps: 防止除零
    :return: 反向MSE损失值
    """
    mse = F.mse_loss(pred, target, reduction='mean')
    # 核心逻辑：负的MSE → 优化器最小化该损失时，会让原始MSE最大化
    reverse_loss = weight * (-mse)
    return reverse_loss


def train_one_epoch(epoch):
    """训练单个epoch"""
    image_embed.train()
    motor_embed.train()
    candidate_generator.train()
    img_sim_model.train()
    driver_sim_model.train()

    total_gen_loss = 0.0
    total_sim_loss = 0.0
    total_reverse_mse_loss = 0.0  # 新增：记录反向MSE损失
    total_loss = 0.0

    max_train_batches = 20  # 限制训练批次（可根据需要调整）
    batch_count = 0

    pbar = tqdm(
        enumerate(train_loader),
        desc=f"训练 Epoch {epoch + 1}/{config['epochs']}",
        total=min(max_train_batches, len(train_loader))
    )

    for batch_idx, batch in pbar:
        if batch_count >= max_train_batches:
            print(f"\n已训练{max_train_batches}个batch，提前终止当前epoch")
            break
        batch_count += 1

        # 解包数据并移动到设备
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        images = torch.stack([imgs1, imgs2], dim=2).to(device)  # (batch, seq, 2, 3, H, W)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        driver = driver.to(device)  # (batch, seq, motor_dim=2)
        next_driver = future_driver[:, 0, :].to(device)  # 下一个动作 (batch, 2)

        # 清零梯度
        optimizer.zero_grad()

        # 1. 特征嵌入
        image_embedded = image_embed(images)  # (batch, seq, 2*embed_dim_gen)
        motor_embedded = motor_embed(driver)  # (batch, seq, embed_dim_gen)

        # 2. 动作生成（使用修正后的模型输出）
        num_candidates = 5
        outputs = candidate_generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=num_candidates,
            temperature=1.0
        )

        # 提取均值和标准差（已修正为单组参数）
        mean = outputs['mean']  # (batch, 2)
        std = outputs['std']  # (batch, 2)

        # 计算生成损失
        gen_loss = nll_loss(mean, std, next_driver)

        # 3. 相似度学习
        # 未来图像嵌入
        future_images_emb = image_embed(future_images)  # (batch, sim_seq_len, 2*embed_dim_gen)
        # 【修改2】接收双输出：相似度嵌入 + 预测电机动作
        img_proj, img_motor_pred = img_sim_model(future_images_emb)  # (batch, similarity_dim), (batch, motor_dim)

        # 【修改3】计算反向MSE损失（让预测动作远离真实动作）
        reverse_mse_loss_val = reverse_mse_loss(
            pred=img_motor_pred,
            target=next_driver,
            weight=config["reverse_mse_weight"]
        )

        # 均值动作作为正样本
        mean_action = mean.unsqueeze(1)  # 增加时间维度 (batch, 1, 2)
        mean_embedded = motor_embed(mean_action)  # (batch, 1, embed_dim_gen)
        mean_proj = driver_sim_model(mean_embedded[:, -1, :])  # (batch, similarity_dim)
        candidate_projections = [mean_proj]  # 第一个元素为正样本

        # 候选动作作为负样本
        for candidate in outputs['candidates']:  # 直接遍历candidates列表（修正点）
            # candidate形状: (batch, 1, 2)
            candidate_embedded = motor_embed(candidate)  # (batch, 1, embed_dim_gen)
            candidate_proj = driver_sim_model(candidate_embedded[:, -1, :])  # (batch, similarity_dim)
            candidate_projections.append(candidate_proj)

        # 计算相似度损失
        sim_loss = info_ce_loss(img_proj, candidate_projections)

        # 【修改4】总损失 = 生成损失 + 相似度损失 + 反向MSE损失
        loss = gen_loss + sim_loss + reverse_mse_loss_val * 0
        loss.backward()
        optimizer.step()

        # 累计损失
        total_gen_loss += gen_loss.item()
        total_sim_loss += sim_loss.item()
        total_reverse_mse_loss += reverse_mse_loss_val.item()  # 新增
        total_loss += loss.item()

        # 更新进度条（新增反向MSE损失显示）
        pbar.set_postfix({
            "总损失": f"{loss.item():.4f}",
            "生成损失": f"{gen_loss.item():.4f}",
            "相似度损失": f"{sim_loss.item():.4f}",
            "反向MSE损失": f"{reverse_mse_loss_val.item():.4f}"  # 新增
        })

    # 计算平均损失
    avg_gen_loss = total_gen_loss / min(max_train_batches, len(train_loader))
    avg_sim_loss = total_sim_loss / min(max_train_batches, len(train_loader))
    avg_reverse_mse_loss = total_reverse_mse_loss / min(max_train_batches, len(train_loader))  # 新增
    avg_total_loss = total_loss / min(max_train_batches, len(train_loader))

    return avg_total_loss, avg_gen_loss, avg_sim_loss, avg_reverse_mse_loss  # 新增返回值


def validate_one_epoch(epoch):
    """验证单个epoch"""
    image_embed.eval()
    motor_embed.eval()
    candidate_generator.eval()
    img_sim_model.eval()
    driver_sim_model.eval()

    total_gen_loss = 0.0
    total_sim_loss = 0.0
    total_reverse_mse_loss = 0.0  # 新增
    total_loss = 0.0

    max_val_batches = 20  # 限制验证批次
    batch_count = 0

    pbar = tqdm(
        enumerate(val_loader),
        desc=f"验证 Epoch {epoch + 1}/{config['epochs']}",
        total=min(max_val_batches, len(val_loader))
    )

    with torch.no_grad():  # 关闭梯度计算
        for batch_idx, batch in pbar:
            if batch_count >= max_val_batches:
                print(f"\n已验证{max_val_batches}个batch，提前终止当前epoch")
                break
            batch_count += 1

            # 解包数据
            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
            images = torch.stack([imgs1, imgs2], dim=2).to(device)
            future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
            driver = driver.to(device)
            next_driver = future_driver[:, 0, :].to(device)  # (batch, 2)

            # 特征嵌入
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)

            # 动作生成损失
            num_candidates = 5
            outputs = candidate_generator(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=num_candidates,
                temperature=1.0
            )

            # 提取均值和标准差（修正后单组参数）
            mean = outputs['mean']
            std = outputs['std']
            gen_loss = nll_loss(mean, std, next_driver)

            # 相似度损失
            future_image_embedded = image_embed(future_images)
            # 【修改5】验证阶段同样接收双输出
            img_proj, img_motor_pred = img_sim_model(future_image_embedded)

            # 【修改6】验证阶段计算反向MSE损失
            reverse_mse_loss_val = reverse_mse_loss(
                pred=img_motor_pred,
                target=next_driver,
                weight=config["reverse_mse_weight"]
            )

            # 均值动作作为正样本
            mean_action = mean.unsqueeze(1)
            mean_embedded = motor_embed(mean_action)
            mean_proj = driver_sim_model(mean_embedded[:, -1, :])
            candidate_projections = [mean_proj]

            # 候选动作作为负样本
            for candidate in outputs['candidates']:  # 修正点：遍历单组候选
                candidate_embedded = motor_embed(candidate)
                candidate_proj = driver_sim_model(candidate_embedded[:, -1, :])
                candidate_projections.append(candidate_proj)

            sim_loss = info_ce_loss(img_proj, candidate_projections)

            # 【修改7】验证总损失同样包含反向MSE损失
            total_batch_loss = gen_loss + sim_loss + reverse_mse_loss_val

            # 累计损失
            total_gen_loss += gen_loss.item()
            total_sim_loss += sim_loss.item()
            total_reverse_mse_loss += reverse_mse_loss_val.item()  # 新增
            total_loss += total_batch_loss.item()

            # 进度条显示（新增反向MSE损失）
            pbar.set_postfix({
                "验证总损失": f"{total_batch_loss.item():.4f}",
                "验证生成损失": f"{gen_loss.item():.4f}",
                "验证相似度损失": f"{sim_loss.item():.4f}",
                "验证反向MSE损失": f"{reverse_mse_loss_val.item():.4f}"  # 新增
            })

    # 计算平均损失
    avg_gen_loss = total_gen_loss / min(max_val_batches, len(val_loader))
    avg_sim_loss = total_sim_loss / min(max_val_batches, len(val_loader))
    avg_reverse_mse_loss = total_reverse_mse_loss / min(max_val_batches, len(val_loader))  # 新增
    avg_total_loss = total_loss / min(max_val_batches, len(val_loader))

    return avg_total_loss, avg_gen_loss, avg_sim_loss, avg_reverse_mse_loss  # 新增返回值


def main():
    best_val_loss = float('inf')
    print("=" * 50)
    print("开始训练（含验证集评估）")
    print(f"总epoch数：{config['epochs']} | 批量大小：{config['batch_size']} | 设备：{device}")
    print(f"反向MSE损失权重：{config['reverse_mse_weight']}")
    print(f"loss数据保存目录：{config['loss_data_path']}")
    print("=" * 50)

    # 【修改8】更新损失记录字典，新增反向MSE损失项
    loss_records = {
        "train_total": [], "train_gen": [], "train_sim": [], "train_reverse_mse": [],
        "val_total": [], "val_gen": [], "val_sim": [], "val_reverse_mse": []
    }

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()

        # 训练（接收新增的反向MSE损失返回值）
        train_total, train_gen, train_sim, train_reverse_mse = train_one_epoch(epoch)
        # 学习率调度
        sch.step()
        # 验证（接收新增的反向MSE损失返回值）
        val_total, val_gen, val_sim, val_reverse_mse = validate_one_epoch(epoch)

        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time

        # 【修改9】记录反向MSE损失
        loss_records["train_total"].append(train_total)
        loss_records["train_gen"].append(train_gen)
        loss_records["train_sim"].append(train_sim)
        loss_records["train_reverse_mse"].append(train_reverse_mse)  # 新增
        loss_records["val_total"].append(val_total)
        loss_records["val_gen"].append(val_gen)
        loss_records["val_sim"].append(val_sim)
        loss_records["val_reverse_mse"].append(val_reverse_mse)  # 新增

        # 打印epoch信息（新增反向MSE损失显示）
        print("\n" + "=" * 30)
        print(f"Epoch {epoch + 1}/{config['epochs']} | 耗时：{epoch_time:.2f}秒")
        print(
            f"【训练集】总损失：{train_total:.4f} | 生成损失：{train_gen:.4f} | 相似度损失：{train_sim:.4f} | 反向MSE损失：{train_reverse_mse:.4f}")
        print(
            f"【验证集】总损失：{val_total:.4f} | 生成损失：{val_gen:.4f} | 相似度损失：{val_sim:.4f} | 反向MSE损失：{val_reverse_mse:.4f}")
        print("=" * 30 + "\n")

        # 保存最佳模型（逻辑不变，总损失已包含反向MSE）
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_model_path = os.path.join(config["save_path"], "best_model")
            torch.save({
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "model_states": {
                    "image_embed": image_embed.state_dict(),
                    "motor_embed": motor_embed.state_dict(),
                    "candidate_generator": candidate_generator.state_dict(),
                    "img_sim_model": img_sim_model.state_dict(),
                    "driver_sim_model": driver_sim_model.state_dict()
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": sch.state_dict(),
                "config": config
            }, best_model_path)

            print(f"✅ 保存最佳模型（验证总损失：{best_val_loss:.4f}）至：{best_model_path}")

    # 保存损失记录（包含反向MSE损失）
    loss_save_path = os.path.join(config["loss_data_path"], "loss_records.npy")
    np.save(loss_save_path, loss_records)

    # 训练结束总结
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳验证总损失：{best_val_loss:.4f}")
    print(f"最佳模型路径：{os.path.join(config['save_path'], 'best_model')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
