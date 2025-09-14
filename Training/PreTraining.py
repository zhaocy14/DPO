import os,sys
pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import NLLLoss
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from DataModule.DataModule import CombinedDataset
from Model.Models import ImageEmbedding, MotorEmbedding, EncoderOnlyCandidateGenerator, SimilarityModelImage, SimilarityModelDriver
from tqdm import tqdm

# from Training.DPOTraining import optimizer

# 设置设备为第三张显卡 (cuda:2)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置参数
config = {
    # training parameters
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-4,
    # "weight_decay": 1e-5,
    "sampling_workers": 10,

    # generator model parameters
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,
    "gen_seq_len": 30,  # 观测长度


    "sim_seq_len": 15,  # 预测长度
    "embed_dim_sim": 128,
    "num_layers_sim": 3,
    "nhead_sim": 4,
    "similarity_dim": 32,

    # data/model storage paths
    "data_root_dirs": '/data/cyzhao/collector_cydpo',  # 根据实际情况修改
    "save_path": "./saved_models"

}

os.makedirs(config["save_path"], exist_ok=True)

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
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


def nll_loss(mean, std, target):
    """
    计算高斯分布的负对数似然损失
    :param mean: 模型预测的均值 (batch, motor_dim)
    :param std: 模型预测的标准差 (batch, motor_dim)
    :param target: 真实动作标签 (batch, motor_dim)
    :return: 每个样本的NLL损失 (batch, motor_dim)
    """
    # 避免标准差为0导致log(0)（虽然模型中已限制logvar范围，但加小epsilon更安全）
    eps = 1e-6
    std = std + eps
    # 计算NLL损失（简化版，忽略常数项）
    nll = torch.log(std) + (target - mean) ** 2 / (2 * std ** 2)
    return nll

def cos_loss(img_proj, driver_proj):
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(img_proj, driver_proj, dim=1)  # 形状: (batch_size,)

    # 损失 = 1 - 余弦相似度（均值），目标是让损失接近0
    sim_loss = (1 - cos_sim).mean()
    return sim_loss

start_time = time.time()


# 训练函数
def train_one_epoch(epoch):
    # 设为训练模式（启用dropout/batchnorm更新）
    image_embed.train()
    motor_embed.train()
    candidate_generator.train()
    img_sim_model.train()
    driver_sim_model.train()

    total_gen_loss = 0.0
    total_sim_loss = 0.0
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{config['epochs']}")
    for batch in pbar:
        # 1. 解包数据并移动到设备
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        # 图像拼接：(batch, seq, 3, H, W) → (batch, seq, 2, 3, H, W)（2个相机）
        images = torch.stack([imgs1, imgs2], dim=2).to(device)
        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
        driver = driver.to(device)  # (batch, gen_seq_len, motor_dim)
        future_driver = future_driver.to(device)  # (batch, sim_seq_len, motor_dim)

        # 2. 清零梯度
        optimizer.zero_grad()

        # 3. 特征嵌入（共享嵌入模型）
        image_embedded = image_embed(images)  # (batch, gen_seq_len, 2*embed_dim_gen)
        motor_embedded = motor_embed(driver)  # (batch, gen_seq_len, embed_dim_gen)

        # 4. 动作生成任务：预测下一帧动作（future_driver第0帧）
        next_driver = future_driver[:, 0, :]  # (batch, motor_dim)
        # 生成器输出：mean1/std1（特征1）、mean2/std2（特征2）
        outputs = candidate_generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=1,
            temperature=1.0
        )
        gen_loss1 = nll_loss(outputs['mean1'], outputs['std1'], next_driver[:, 0])
        gen_loss2 = nll_loss(outputs['mean2'], outputs['std2'], next_driver[:, 1])
        gen_loss = (gen_loss1 + gen_loss2) / 2  # 平均两个特征的生成损失

        # 5. 相似度任务：未来图像与当前动作的对齐
        # 未来图像嵌入：(batch, sim_seq_len, 2*embed_dim_gen)
        future_image_embedded = image_embed(future_images)
        # 图像投射：输入未来图像序列，输出相似度向量
        img_proj = img_sim_model(future_image_embedded)  # (batch, similarity_dim)
        # 动作投射：取当前动作序列最后一帧的嵌入
        last_motor_embedded = motor_embedded[:, -1, :]  # (batch, embed_dim_gen)
        # 注意：若motor_embed输出维度与img_sim_model输入不匹配，需拼接/调整（此处按原逻辑）
        driver_proj = driver_sim_model(last_motor_embedded)  # (batch, similarity_dim)
        # 计算相似度损失
        sim_loss = cos_loss(img_proj, driver_proj)

        # 6. 总损失与反向传播
        total_batch_loss = gen_loss + sim_loss
        total_batch_loss.backward()  # 梯度计算
        optimizer.step()  # 参数更新

        # 7. 累计损失（用于计算epoch平均）
        total_gen_loss += gen_loss.item()
        total_sim_loss += sim_loss.item()
        total_loss += total_batch_loss.item()

        # 8. 更新进度条显示
        pbar.set_postfix({
            "总损失": f"{total_batch_loss.item():.4f}",
            "生成损失": f"{gen_loss.item():.4f}",
            "相似度损失": f"{sim_loss.item():.4f}"
        })

    # 计算epoch平均损失（除以batch数，而非样本数，因每个batch已取mean）
    avg_gen_loss = total_gen_loss / len(train_loader)
    avg_sim_loss = total_sim_loss / len(train_loader)
    avg_total_loss = total_loss / len(train_loader)
    return avg_total_loss, avg_gen_loss, avg_sim_loss

def validate_one_epoch(epoch):
    # 设为评估模式（关闭dropout，冻结batchnorm统计）
    image_embed.eval()
    motor_embed.eval()
    candidate_generator.eval()
    img_sim_model.eval()
    driver_sim_model.eval()

    total_gen_loss = 0.0
    total_sim_loss = 0.0
    total_loss = 0.0

    # 关闭梯度计算（加速验证，避免内存占用）
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}/{config['epochs']}")
        for batch in pbar:
            # 1. 解包数据（与训练逻辑完全一致）
            imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
            images = torch.stack([imgs1, imgs2], dim=2).to(device)
            future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)
            driver = driver.to(device)
            future_driver = future_driver.to(device)

            # 2. 特征嵌入（无梯度计算）
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)

            # 3. 动作生成损失计算（与训练一致）
            next_driver = future_driver[:, 0, :]
            outputs = candidate_generator(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=1,
                temperature=1.0
            )
            gen_loss1 = nll_loss(outputs['mean1'], outputs['std1'], next_driver[:, 0])
            gen_loss2 = nll_loss(outputs['mean2'], outputs['std2'], next_driver[:, 1])
            gen_loss = (gen_loss1 + gen_loss2) / 2

            # 4. 相似度损失计算（与训练一致）
            future_image_embedded = image_embed(future_images)
            img_proj = img_sim_model(future_image_embedded)
            last_motor_embedded = motor_embedded[:, -1, :]
            driver_proj = driver_sim_model(last_motor_embedded)
            sim_loss = cos_loss(img_proj, driver_proj)

            # 5. 累计验证损失
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
    avg_gen_loss = total_gen_loss / len(val_loader)
    avg_sim_loss = total_sim_loss / len(val_loader)
    avg_total_loss = total_loss / len(val_loader)
    return avg_total_loss, avg_gen_loss, avg_sim_loss


def main():
    best_val_loss = float('inf')  # 初始化最佳验证损失（无穷大）
    print("="*50)
    print("开始训练（含验证集评估）")
    print(f"总epoch数：{config['epochs']} | 批量大小：{config['batch_size']} | 设备：{device}")
    print("="*50)

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()

        # 1. 训练一个epoch
        train_total, train_gen, train_sim = train_one_epoch(epoch)

        # 2. 验证一个epoch
        val_total, val_gen, val_sim = validate_one_epoch(epoch)

        # 3. 学习率调度（每个epoch后更新）
        sch.step()

        # 4. 计算epoch耗时
        epoch_time = time.time() - epoch_start_time

        # 5. 打印epoch统计信息
        print("\n" + "="*30)
        print(f"Epoch {epoch + 1}/{config['epochs']} | 耗时：{epoch_time:.2f}秒")
        print(f"【训练集】总损失：{train_total:.4f} | 生成损失：{train_gen:.4f} | 相似度损失：{train_sim:.4f}")
        print(f"【验证集】总损失：{val_total:.4f} | 生成损失：{val_gen:.4f} | 相似度损失：{val_sim:.4f}")
        print("="*30 + "\n")

        # 6. 保存最佳模型（仅当当前验证损失优于历史最佳时）
        if val_total < best_val_loss:
            best_val_loss = val_total  # 更新最佳损失
            # 构建模型保存路径
            best_model_path = os.path.join(config["save_path"], config["best_model_name"])
            # 保存模型权重与训练状态（便于后续加载继续训练）
            torch.save({
                "epoch": epoch + 1,  # 当前epoch（已完成）
                "best_val_loss": best_val_loss,  # 最佳验证损失
                "model_states": {
                    "image_embed": image_embed.state_dict(),
                    "motor_embed": motor_embed.state_dict(),
                    "candidate_generator": candidate_generator.state_dict(),
                    "img_sim_model": img_sim_model.state_dict(),
                    "driver_sim_model": driver_sim_model.state_dict()
                },
                "optimizer_state": optimizer.state_dict(),  # 优化器状态
                "scheduler_state": sch.state_dict(),  # 学习率调度器状态
                "config": config  # 训练配置（便于复现）
            }, best_model_path)

            print(f"✅ 保存最佳模型（验证总损失：{best_val_loss:.4f}）至：{best_model_path}")

    # 训练结束后打印总结
    print("\n" + "="*50)
    print("训练完成！")
    print(f"最佳验证总损失：{best_val_loss:.4f}")
    print(f"最佳模型路径：{os.path.join(config['save_path'], config['best_model_name'])}")
    print("="*50)


if __name__ == "__main__":
    main()