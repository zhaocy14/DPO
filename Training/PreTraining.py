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
    "batch_size": 16,
    "epochs": 100,
    "lr": 1e-4,
    # "weight_decay": 1e-5,
    "sampling_workers": 15,

    # generator model parameters
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,
    "gen_seq_len": 30,  # 观测长度


    "sim_seq_len": 30,  # 预测长度
    "embed_dim_sim": 32,
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
)

driver_sim_model = SimilarityModelDriver(
    embed_dim=config['embed_dim_sim'],
    similarity_dim=config['similarity_dim'],
)

data_root = config["data_root_dirs"]

data_dir_list = []
for file in os.listdir(data_root):
    if os.path.isdir(os.path.join(data_root, file)):
        if "2025" in file:
            data_dir_list.append(os.path.join(data_root, file))

all_dataset = CombinedDataset(dir_list=data_dir_list,
                                frame_len=config["gen_seq_len"],
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
        # 解包数据并移动到设备
        imgs1, imgs2, driver, future_imgs1, future_imgs2, future_driver = batch
        print(imgs1.shape, imgs2.shape, driver.shape, future_imgs1.shape, future_imgs2.shape, future_driver.shape)
        # 调整图像数据形状 (batch, seq, 3, H, W) -> (batch, seq, num_cameras, 3, H, W)
        images = torch.stack([imgs1, imgs2], dim=2).to(device)  # (batch, seq, 2, 3, H, W)

        future_images = torch.stack([future_imgs1, future_imgs2], dim=2).to(device)

        driver = driver.to(device)  # (batch, frame_len, motor_dim)

        # 清零梯度
        optimizer.zero_grad()

        # 1. 特征嵌入
        image_embedded = image_embed(images)  # (batch, seq, 2*embed_dim)
        motor_embedded = motor_embed(driver)  # (batch, seq, embed_dim)

        # 2. 动作生成任务 - 使用NLL损失
        # 为了简化，我们只预测下一帧动作
        next_driver = future_driver[:, 0, :].to(device)  # (batch, motor_dim)

        # 获取生成器输出的均值和方差
        outputs = candidate_generator(image_embedded, motor_embedded, num_candidates=1, temperature=1.0)
        mean1 = outputs['mean1']
        std1 = outputs['std1']
        mean2 = outputs['mean2']
        std2 = outputs['std2']

        gen_loss1 = nll_loss(mean1, std1, next_driver[0]).mean()
        gen_loss2 = nll_loss(mean2, std2, next_driver[1]).mean()
        gen_loss = (gen_loss1 + gen_loss2) / 2

        # 3. 相似度学习任务
        # 未来图像嵌入
        future_img1_emb = image_embed(future_imgs1.to(device))  # (batch, predict_len, embed_dim)
        future_img2_emb = image_embed(future_imgs2.to(device))  # (batch, predict_len, embed_dim)

        # 对未来图像序列和当前动作进行投射
        img_proj = img_sim_model(future_img1_emb, future_img2_emb)  # (batch, similarity_dim)

        # 对当前动作的最后一帧进行投射
        last_motor_embedded = motor_embedded[:, -1, :]  # (batch, embed_dim)
        driver_proj = driver_sim_model(last_motor_embedded) # (batch, similarity_dim)

        # 相似度损失：希望余弦相似度接近1
        sim_loss = cos_loss(img_proj=img_proj, driver_proj=driver_proj)

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
    avg_gen_loss = total_gen_loss / len(train_dataset)
    avg_sim_loss = total_sim_loss / len(train_dataset)
    avg_total_loss = total_loss / len(train_dataset)

    return avg_total_loss, avg_gen_loss, avg_sim_loss


# 主训练循环
def main():
    best_val_loss = float('inf')

    print("开始训练...")
    for epoch in range(config["epochs"]):
        start_time = time.time()

        # 训练一个epoch
        train_total, train_gen, train_sim = train_one_epoch(epoch)
        sch.step()



if __name__ == "__main__":
    main()