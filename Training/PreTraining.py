import os, sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataModule.DataModule import CombinedDataset
from Model.Models import ImageEmbedding, MotorEmbedding, EncoderOnlyCandidateGenerator, SimilarityModelImage, \
    SimilarityModelDriver
from tqdm import tqdm
from Model.Models import calculate_model_size

# 设置设备为第三张显卡 (cuda:2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置参数
config = {
    # training parameters
    "batch_size": 4,
    "epochs": 100,
    "lr": 1e-5,
    "sampling_workers": 15,

    # generator model parameters
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,  # 每个动作包含2个电机信号
    "gen_seq_len": 30,  # 观测长度

    "sim_seq_len": 30,  # 预测长度
    "embed_dim_sim": 128,
    "num_layers_sim": 3,
    "nhead_sim": 4,
    "similarity_dim": 32,

    # data/model storage paths
    "data_root_dirs": '/data/cyzhao/collector_cydpo',  # 根据实际情况修改
    "save_path": "./saved_models",
    "loss_data_path": "./loss_records"  # 仅保存loss数据的目录
}

os.makedirs(config["save_path"], exist_ok=True)
os.makedirs(config["loss_data_path"], exist_ok=True)

# initialize models
image_embed = ImageEmbedding(embed_dim=config["embed_dim_gen"], num_layers=3, is_resnet=False).to(device)
motor_embed = MotorEmbedding(motor_dim=config["motor_dim"], embed_dim=config["embed_dim_gen"]).to(device)
candidate_generator = EncoderOnlyCandidateGenerator(
    embed_dim=config["embed_dim_gen"],
    nhead=config["nhead_gen"],
    num_layers=config["num_layers_gen"],
    motor_dim=config["motor_dim"],
    max_seq_length=config["gen_seq_len"]).to(device)

img_sim_model = SimilarityModelImage(
    embed_dim=config['embed_dim_sim'],
    num_frames=config['sim_seq_len'],
    num_layers=config['num_layers_sim'],
    nhead=config['nhead_sim'],
    similarity_dim=config['similarity_dim']
).to(device)

driver_sim_model = SimilarityModelDriver(
    embed_dim=config['embed_dim_sim'],
    similarity_dim=config['similarity_dim'],
).to(device)

data_root = config["data_root_dirs"]

data_dir_list = []
for file in os.listdir(data_root):
    if os.path.isdir(os.path.join(data_root, file)):
        if "2025" in file:
            data_dir_list.append(os.path.join(data_root, file))

all_dataset = CombinedDataset(dir_list=data_dir_list,
                              frame_len=config["gen_seq_len"],
                              predict_len=config['sim_seq_len'],
                              show=True)
train_dataset = all_dataset.training_dataset
val_dataset = all_dataset.val_dataset
print(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config['batch_size'],
                          shuffle=True,
                          num_workers=config['sampling_workers'])

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=config['batch_size'],
                        shuffle=False,
                        num_workers=config['sampling_workers'])

optimizer = torch.optim.Adam(params=[{'params': image_embed.parameters()},
                                     {'params': motor_embed.parameters()},
                                     {'params': candidate_generator.parameters()},
                                     {'params': img_sim_model.parameters()},
                                     {'params': driver_sim_model.parameters()}],
                             lr=config['lr'])
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def nll_loss(mean, std, target):
    """
    计算高斯分布的负对数似然损失
    :param mean: 模型预测的均值 (batch, motor_dim)
    :param std: 模型预测的标准差 (batch, motor_dim)
    :param target: 真实动作标签 (batch, motor_dim)
    :return: 每个样本的NLL损失 (batch, motor_dim)
    """
    # 避免标准差为0导致log(0)
    eps = 1e-6
    std = std + eps
    # 计算NLL损失（简化版，忽略常数项）
    nll = torch.log(std) + (target - mean) ** 2 / (2 * std ** 2)
    return nll.mean()  # 对两个电机信号的损失取平均


def info_ce_loss(img_proj, candidate_projections, temperature=0.1):
    """
    计算Noise Contrastive Estimation Loss (InfoCE Loss)
    :param img_proj: 未来图像序列的投影 (batch, similarity_dim)
    :param candidate_projections: 候选动作的投影列表，第一个元素是正样本
    :param temperature: 温度参数，控制分布的陡峭程度
    :return: InfoCE损失值
    """
    batch_size = img_proj.shape[0]

    # 将所有候选投影拼接成 (batch, num_candidates, similarity_dim)
    candidates = torch.stack(candidate_projections, dim=1)

    # 计算图像投影与每个候选动作投影的相似度 (batch, num_candidates)
    similarities = F.cosine_similarity(
        img_proj.unsqueeze(1),  # (batch, 1, similarity_dim)
        candidates,  # (batch, num_candidates, similarity_dim)
        dim=2
    )

    # 应用温度缩放
    similarities = similarities / temperature

    # InfoCE Loss：第一个候选是正样本，其余是负样本
    loss = F.cross_entropy(
        similarities,
        torch.zeros(batch_size, dtype=torch.long, device=img_proj.device)
    )

    return loss


start_time = time.time()


# 训练函数
def train_one_epoch(epoch):
    image_embed.train()
    motor_embed.train()
    candidate_generator.train()
    img_sim_model.train()
    driver_sim_model.train()

    total_gen_loss = 0.0
    total_sim_loss = 0.0
    total_loss = 0.0

    max_train_batches = 10  # 只训练前10个batch
    batch_count = 0  # 计数当前epoch已训练的batch数

    pbar = tqdm(enumerate(train_loader), desc=f"训练 Epoch {epoch + 1}/{config['epochs']}",
                total=min(max_train_batches, len(train_loader)))

    for batch_idx, batch in pbar:
        if batch_count >= max_train_batches:
            print(f"\n已训练{max_train_batches}个batch，提前终止当前epoch")
            break
        batch_count += 1

        # 解包数据并移动到设备
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        images = torch.stack([imgs1, imgs2], dim=2).to(device)  # (batch, seq, 2, 3, H, W)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        driver = driver.to(device)  # (batch, frame_len, motor_dim)
        next_driver = future_driver[:, 0, :].to(device)  # (batch, motor_dim)，下一帧完整动作

        # 清零梯度
        optimizer.zero_grad()

        # 1. 特征嵌入
        image_embedded = image_embed(images)  # (batch, seq, 2*embed_dim)
        motor_embedded = motor_embed(driver)  # (batch, seq, embed_dim)

        # 2. 动作生成任务 - 使用NLL损失
        # 获取生成器输出的均值、方差以及候选动作
        num_candidates = 5  # 生成5个候选动作
        outputs = candidate_generator(image_embedded, motor_embedded, num_candidates=num_candidates, temperature=1.0)

        # 构建完整动作的均值和标准差 (batch, motor_dim)
        mean = torch.stack([outputs['mean1'], outputs['mean2']], dim=1)  # 合并两个电机的均值
        std = torch.stack([outputs['std1'], outputs['std2']], dim=1)  # 合并两个电机的标准差

        # 计算生成损失（同时考虑两个电机）
        gen_loss = nll_loss(mean, std, next_driver)

        # 3. 相似度学习任务 - 使用InfoCE Loss
        # 未来图像嵌入
        future_images_emb = image_embed(future_images)  # (batch, predict_len, 2*embed_dim)
        img_proj = img_sim_model(future_images_emb)  # (batch, similarity_dim)

        # 准备候选动作：将均值动作作为正样本，其他候选作为负样本
        # 构建完整的均值动作 (batch, 1, motor_dim)
        mean_action = mean.unsqueeze(1)  # 增加时间维度
        mean_embedded = motor_embed(mean_action)  # 嵌入均值动作
        mean_proj = driver_sim_model(mean_embedded[:, -1, :])  # 投影到相似度空间

        # 构建候选动作投影列表
        candidate_projections = [mean_proj]  # 第一个是正样本（均值动作）

        # 处理所有候选动作（每个候选包含两个电机的值）
        for i in range(num_candidates):
            # 合并两个电机的候选值，构建完整动作 (batch, 1, motor_dim)
            candidate_action = torch.stack([
                outputs['candidates1'][i].squeeze(1),  # 第一个电机的候选值
                outputs['candidates2'][i].squeeze(1)  # 第二个电机的候选值
            ], dim=1).unsqueeze(1)

            # 嵌入并投影候选动作
            candidate_embedded = motor_embed(candidate_action)
            candidate_proj = driver_sim_model(candidate_embedded[:, -1, :])
            candidate_projections.append(candidate_proj)

        # 计算InfoCE损失
        sim_loss = info_ce_loss(img_proj, candidate_projections)

        # 总损失：加权求和两个任务的损失
        loss = gen_loss + sim_loss

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 累计损失
        total_gen_loss += gen_loss.item()
        total_sim_loss += sim_loss.item()
        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({
            "总损失": f"{loss.item():.4f}",
            "生成损失": f"{gen_loss.item():.4f}",
            "相似度损失": f"{sim_loss.item():.4f}"
        })

    # 计算平均损失
    avg_gen_loss = total_gen_loss / min(max_train_batches, len(train_loader))
    avg_sim_loss = total_sim_loss / min(max_train_batches, len(train_loader))
    avg_total_loss = total_loss / min(max_train_batches, len(train_loader))

    return avg_total_loss, avg_gen_loss, avg_sim_loss


def validate_one_epoch(epoch):
    # 设为评估模式
    image_embed.eval()
    motor_embed.eval()
    candidate_generator.eval()
    img_sim_model.eval()
    driver_sim_model.eval()

    total_gen_loss = 0.0
    total_sim_loss = 0.0
    total_loss = 0.0

    max_val_batches = 10  # 只验证前10个batch
    batch_count = 0  # 计数当前epoch已验证的batch数

    pbar = tqdm(enumerate(val_loader), desc=f"验证 Epoch {epoch + 1}/{config['epochs']}",
                total=min(max_val_batches, len(val_loader)))

    with torch.no_grad():
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
            next_driver = future_driver[:, 0, :].to(device)  # 下一帧完整动作

            # 特征嵌入
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)

            # 动作生成损失计算
            num_candidates = 5
            outputs = candidate_generator(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=num_candidates,
                temperature=1.0
            )

            # 构建完整动作的均值和标准差
            mean = torch.stack([outputs['mean1'], outputs['mean2']], dim=1)
            std = torch.stack([outputs['std1'], outputs['std2']], dim=1)
            gen_loss = nll_loss(mean, std, next_driver)

            # 相似度损失计算（使用InfoCE Loss）
            future_image_embedded = image_embed(future_images)
            img_proj = img_sim_model(future_image_embedded)

            # 处理均值动作作为正样本
            mean_action = mean.unsqueeze(1)
            mean_embedded = motor_embed(mean_action)
            mean_proj = driver_sim_model(mean_embedded[:, -1, :])
            candidate_projections = [mean_proj]

            # 处理所有候选动作
            for i in range(num_candidates):
                candidate_action = torch.stack([
                    outputs['candidates1'][i].squeeze(1),
                    outputs['candidates2'][i].squeeze(1)
                ], dim=1).unsqueeze(1)

                candidate_embedded = motor_embed(candidate_action)
                candidate_proj = driver_sim_model(candidate_embedded[:, -1, :])
                candidate_projections.append(candidate_proj)

            # 计算InfoCE损失
            sim_loss = info_ce_loss(img_proj, candidate_projections)

            # 累计验证损失
            total_batch_loss = gen_loss + sim_loss
            total_gen_loss += gen_loss.item()
            total_sim_loss += sim_loss.item()
            total_loss += total_batch_loss.item()

            # 进度条显示
            pbar.set_postfix({
                "验证总损失": f"{total_batch_loss.item():.4f}",
                "验证生成损失": f"{gen_loss.item():.4f}",
                "验证相似度损失": f"{sim_loss.item():.4f}"
            })

    # 计算验证集平均损失
    avg_gen_loss = total_gen_loss / min(max_val_batches, len(val_loader))
    avg_sim_loss = total_sim_loss / min(max_val_batches, len(val_loader))
    avg_total_loss = total_loss / min(max_val_batches, len(val_loader))
    return avg_total_loss, avg_gen_loss, avg_sim_loss


def main():
    best_val_loss = float('inf')  # 初始化最佳验证损失
    print("=" * 50)
    print("开始训练（含验证集评估）")
    print(f"总epoch数：{config['epochs']} | 批量大小：{config['batch_size']} | 设备：{device}")
    print(f"总Epoch数：{config['epochs']} | loss数据保存目录：{config['loss_data_path']}")
    print("=" * 50)

    # 用list记录loss
    loss_records = {
        "train_total": [],
        "train_gen": [],
        "train_sim": [],
        "val_total": [],
        "val_gen": [],
        "val_sim": []
    }

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()

        # 1. 训练一个epoch
        train_total, train_gen, train_sim = train_one_epoch(epoch)

        # 2. 学习率调度
        sch.step()

        # 3. 验证一个epoch
        val_total, val_gen, val_sim = validate_one_epoch(epoch)

        # 4. 计算epoch耗时
        epoch_time = time.time() - epoch_start_time

        # 记录loss
        loss_records["train_total"].append(train_total)
        loss_records["train_gen"].append(train_gen)
        loss_records["train_sim"].append(train_sim)
        loss_records["val_total"].append(val_total)
        loss_records["val_gen"].append(val_gen)
        loss_records["val_sim"].append(val_sim)

        # 5. 打印epoch统计信息
        print("\n" + "=" * 30)
        print(f"Epoch {epoch + 1}/{config['epochs']} | 耗时：{epoch_time:.2f}秒")
        print(f"【训练集】总损失：{train_total:.4f} | 生成损失：{train_gen:.4f} | 相似度损失：{train_sim:.4f}")
        print(f"【验证集】总损失：{val_total:.4f} | 生成损失：{val_gen:.4f} | 相似度损失：{val_sim:.4f}")
        print("=" * 30 + "\n")

        # 6. 保存最佳模型
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

    loss_save_path = os.path.join(config["loss_data_path"], "loss_records.npy")
    np.save(loss_save_path, loss_records)

    # 训练结束后打印总结
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳验证总损失：{best_val_loss:.4f}")
    print(f"最佳模型路径：{os.path.join(config['save_path'], 'best_model')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
