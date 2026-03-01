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
    # 保留导入但注释（方便后续恢复）
    # SimilarityModelImage,
    # SimilarityModelDriver,
    calculate_model_size,
    # 新增：导入Judge相关模块
    JudgeModelImage,
    JudgeModelDriver,
    JudgeModel,
    ActionExtract
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

    # 生成器模型参数
    "embed_dim_gen": 128,
    "nhead_gen": 8,
    "num_layers_gen": 16,
    "motor_dim": 2,  # 两个电机，动作维度为2
    "gen_seq_len": 30,  # 观测序列长度

    # 相似度模型参数（保留配置但注释，方便后续恢复）
    # "sim_seq_len": 30,  # 预测序列长度
    # "embed_dim_sim": 128,
    # "num_layers_sim": 3,
    # "nhead_sim": 4,
    # "similarity_dim": 32,

    # 新增：Judge模型核心参数（匹配初始化模板）
    "embed_dim_judge": 128,  # 对应初始化的embed_dim
    "seq_length_judge": 30,  # 对应初始化的num_frames/seq_length
    "judge_dim": 32,  # 对应初始化的judge_dim
    "action_extract_hidden": 64,  # 对应初始化的hidden_dim
    "dropout_rate_judge": 0.2,  # ActionExtract的dropout_rate

    # 损失权重（可根据训练效果调整）
    "judge_loss_weight": 1.0,
    "action_extract_loss_weight": 1.0,

    # 数据和模型路径
    "data_root_dirs": '/data/cyzhao/collector_cydpo',
    "save_path": "./saved_models",
    "loss_data_path": "./loss_records"
}

# 创建保存目录
os.makedirs(config["save_path"], exist_ok=True)
os.makedirs(config["loss_data_path"], exist_ok=True)

# 初始化原有模型
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

# 注释掉相似度模型初始化（保留代码结构，方便后续恢复）
# img_sim_model = SimilarityModelImage(
#     embed_dim=config['embed_dim_sim'],
#     num_frames=config['sim_seq_len'],
#     num_layers=config['num_layers_sim'],
#     nhead=config['nhead_sim'],
#     similarity_dim=config['similarity_dim']
# ).to(device)

# driver_sim_model = SimilarityModelDriver(
#     embed_dim=config['embed_dim_sim'],
#     similarity_dim=config['similarity_dim'],
# ).to(device)

# ===================== 核心修正：Judge相关模型初始化 =====================
# 严格按照指定模板初始化，参数名/变量名完全匹配
judge_image_model = JudgeModelImage(
    embed_dim=config["embed_dim_judge"],
    num_frames=config["seq_length_judge"],
    num_layers=2,
    nhead=4,
    judge_dim=config["judge_dim"]
).to(device)

judge_driver_model = JudgeModelDriver(
    embed_dim=config["embed_dim_judge"],
    judge_dim=config["judge_dim"]
).to(device)

judge_total_model = JudgeModel(
    embed_dim=config["embed_dim_judge"],
    num_frames=config["seq_length_judge"],
    num_layers=2,
    nhead=4,
    judge_dim=config["judge_dim"]
).to(device)

# 初始化新增的ActionExtract模型
action_extract_model = ActionExtract(
    in_dim=config["judge_dim"],  # 输入维度=JudgeModelImage输出维度
    hidden_dim=config["action_extract_hidden"],
    out_dim=config["motor_dim"],  # 输出维度=电机信号维度
    dropout_rate=config["dropout_rate_judge"]
).to(device)
# =========================================================================

# 加载数据集
data_root = config["data_root_dirs"]
data_dir_list = [
    os.path.join(data_root, file)
    for file in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, file)) and "2025" in file
]

# predict_len改用gen_seq_len（因sim_seq_len已注释）
all_dataset = CombinedDataset(
    dir_list=data_dir_list,
    frame_len=config["gen_seq_len"],
    predict_len=config['gen_seq_len'],  # 替换原sim_seq_len
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

# 优化器和学习率调度器（注释掉相似度模型参数）
optimizer = torch.optim.Adam(
    params=[
        {'params': image_embed.parameters()},
        {'params': motor_embed.parameters()},
        {'params': candidate_generator.parameters()},
        # 注释掉相似度模型优化器参数
        # {'params': img_sim_model.parameters()},
        # {'params': driver_sim_model.parameters()},
        # 修正：使用新的Judge模型变量名
        {'params': judge_image_model.parameters()},
        {'params': judge_driver_model.parameters()},
        {'params': judge_total_model.parameters()},
        {'params': action_extract_model.parameters()}
    ],
    lr=config['lr']
)
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


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


# 注释掉相似度损失函数（保留代码结构，方便后续恢复）
# def info_ce_loss(img_proj, candidate_projections, temperature=0.1):
#     """
#     计算Info Noise Contrastive Estimation Loss
#     :param img_proj: 未来图像序列投影 (batch, similarity_dim)
#     :param candidate_projections: 候选动作投影列表，第一个为正样本
#     :return: InfoCE损失
#     """
#     batch_size = img_proj.shape[0]
#     candidates = torch.stack(candidate_projections, dim=1)  # (batch, num_candidates, similarity_dim)

#     # 计算图像投影与候选动作投影的余弦相似度
#     similarities = F.cosine_similarity(
#         img_proj.unsqueeze(1),  # (batch, 1, similarity_dim)
#         candidates,  # (batch, num_candidates, similarity_dim)
#         dim=2
#     )  # (batch, num_candidates)

#     # 温度缩放 + 交叉熵损失（第一个候选为正样本）
#     similarities = similarities / temperature
#     loss = F.cross_entropy(
#         similarities,
#         torch.zeros(batch_size, dtype=torch.long, device=img_proj.device)
#     )
#     return loss


def train_one_epoch(epoch):
    """训练单个epoch"""
    # 所有模型设为训练模式（注释掉相似度模型）
    image_embed.train()
    motor_embed.train()
    candidate_generator.train()
    # img_sim_model.train()  # 注释掉相似度模型训练模式
    # driver_sim_model.train()
    judge_image_model.train()
    judge_driver_model.train()
    judge_total_model.train()
    action_extract_model.train()

    total_gen_loss = 0.0
    # total_sim_loss = 0.0  # 注释掉相似度损失累计
    total_judge_loss = 0.0
    total_action_extract_loss = 0.0
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

        # 1. 原有逻辑：特征嵌入 + 动作生成
        # 特征嵌入
        image_embedded = image_embed(images)  # (batch, seq, 2*embed_dim_gen)
        motor_embedded = motor_embed(driver)  # (batch, seq, embed_dim_gen)

        # 动作生成
        num_candidates = 5
        outputs = candidate_generator(
            image_embedded=image_embedded,
            motor_embedded=motor_embedded,
            num_candidates=num_candidates,
            temperature=1.0
        )

        # 提取均值和标准差
        mean = outputs['mean']  # (batch, 2)
        std = outputs['std']  # (batch, 2)

        # 生成损失
        gen_loss = nll_loss(mean, std, next_driver)

        # 注释掉相似度损失计算逻辑
        # # 相似度学习
        # future_images_emb = image_embed(future_images)  # (batch, sim_seq_len, 2*embed_dim_gen)
        # img_proj = img_sim_model(future_images_emb)  # (batch, similarity_dim)

        # # 均值动作作为正样本
        # mean_action = mean.unsqueeze(1)  # (batch, 1, 2)
        # mean_embedded = motor_embed(mean_action)  # (batch, 1, embed_dim_gen)
        # mean_proj = driver_sim_model(mean_embedded[:, -1, :])  # (batch, similarity_dim)
        # candidate_projections = [mean_proj]

        # # 候选动作作为负样本
        # for candidate in outputs['candidates']:
        #     candidate_embedded = motor_embed(candidate)  # (batch, 1, embed_dim_gen)
        #     candidate_proj = driver_sim_model(candidate_embedded[:, -1, :])  # (batch, similarity_dim)
        #     candidate_projections.append(candidate_proj)

        # # 相似度损失
        # sim_loss = info_ce_loss(img_proj, candidate_projections)

        # 2. 新增：Judge模型训练逻辑（同步修正变量名）
        # 2.1 JudgeModelImage输入（复用future_images，无需相似度模型）
        future_images_emb = image_embed(future_images)  # 仅为Judge模型提供输入
        judge_img_feat = judge_image_model(future_images_emb)  # (batch, judge_dim)

        # 2.2 每个候选action的JudgeModelDriver输入
        # 构建候选动作列表（均值动作+生成的候选动作）
        candidate_actions = [mean.unsqueeze(1)] + outputs['candidates']  # 第一个是最高概率动作
        judge_driver_feats = []
        for action in candidate_actions:
            # action shape: (batch, 1, 2)
            action_emb = motor_embed(action)  # (batch, 1, embed_dim_gen)
            judge_driver_feat = judge_driver_model(action_emb[:, -1, :])  # (batch, judge_dim)
            judge_driver_feats.append(judge_driver_feat)

        # 2.3 JudgeModel计算匹配分数（使用judge_total_model）
        match_scores = judge_total_model(judge_img_feat, judge_driver_feats)  # (batch, num_candidates+1)

        # 2.4 构建标签：最高概率动作（第一个）为1，其他为0
        batch_size = match_scores.shape[0]
        judge_labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # 标签为第一个候选的索引

        # 2.5 Judge交叉熵损失
        judge_ce_loss = F.cross_entropy(match_scores, judge_labels) * config["judge_loss_weight"]

        # 3. 新增：ActionExtract训练逻辑（去除图像中的运动信息）
        # 3.1 ActionExtract预测动作（使用action_extract_model）
        pred_action = action_extract_model(judge_img_feat)  # (batch, motor_dim=2)

        # 3.2 MSE损失：让预测动作与真实动作不相似（注：若需强化该效果，可乘以权重或取反）
        # 核心逻辑：最小化该损失会让预测接近真实，因此这里取反实现“不相似”目标
        action_extract_loss = -F.mse_loss(pred_action, next_driver) * config["action_extract_loss_weight"]

        # 4. 总损失：生成损失 + Judge损失 + ActionExtract损失（去掉sim_loss）
        loss = gen_loss + judge_ce_loss + action_extract_loss

        # 反向传播 + 优化
        loss.backward()
        optimizer.step()

        # 累计损失（注释掉相似度损失）
        total_gen_loss += gen_loss.item()
        # total_sim_loss += sim_loss.item()
        total_judge_loss += judge_ce_loss.item()
        total_action_extract_loss += action_extract_loss.item()
        total_loss += loss.item()

        # 更新进度条（去掉相似度损失显示）
        pbar.set_postfix({
            "总损失": f"{loss.item():.4f}",
            "生成损失": f"{gen_loss.item():.4f}",
            # "相似度损失": f"{sim_loss.item():.4f}",
            "Judge损失": f"{judge_ce_loss.item():.4f}",
            "ActionExtract损失": f"{action_extract_loss.item():.4f}"
        })

    # 计算平均损失（注释掉相似度损失）
    avg_gen_loss = total_gen_loss / min(max_train_batches, len(train_loader))
    # avg_sim_loss = total_sim_loss / min(max_train_batches, len(train_loader))
    avg_judge_loss = total_judge_loss / min(max_train_batches, len(train_loader))
    avg_action_extract_loss = total_action_extract_loss / min(max_train_batches, len(train_loader))
    avg_total_loss = total_loss / min(max_train_batches, len(train_loader))

    # 返回值去掉相似度损失
    return avg_total_loss, avg_gen_loss, avg_judge_loss, avg_action_extract_loss


def validate_one_epoch(epoch):
    """验证单个epoch"""
    # 所有模型设为评估模式（注释掉相似度模型）
    image_embed.eval()
    motor_embed.eval()
    candidate_generator.eval()
    # img_sim_model.eval()
    # driver_sim_model.eval()
    judge_image_model.eval()
    judge_driver_model.eval()
    judge_total_model.eval()
    action_extract_model.eval()

    total_gen_loss = 0.0
    # total_sim_loss = 0.0
    total_judge_loss = 0.0
    total_action_extract_loss = 0.0
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

            # 1. 原有逻辑：特征嵌入 + 动作生成
            image_embedded = image_embed(images)
            motor_embedded = motor_embed(driver)

            # 动作生成
            num_candidates = 5
            outputs = candidate_generator(
                image_embedded=image_embedded,
                motor_embedded=motor_embedded,
                num_candidates=num_candidates,
                temperature=1.0
            )

            mean = outputs['mean']
            std = outputs['std']
            gen_loss = nll_loss(mean, std, next_driver)

            # 注释掉相似度损失计算
            # # 相似度损失
            # future_image_embedded = image_embed(future_images)
            # img_proj = img_sim_model(future_image_embedded)

            # mean_action = mean.unsqueeze(1)
            # mean_embedded = motor_embed(mean_action)
            # mean_proj = driver_sim_model(mean_embedded[:, -1, :])
            # candidate_projections = [mean_proj]

            # for candidate in outputs['candidates']:
            #     candidate_embedded = motor_embed(candidate)
            #     candidate_proj = driver_sim_model(candidate_embedded[:, -1, :])
            #     candidate_projections.append(candidate_proj)

            # sim_loss = info_ce_loss(img_proj, candidate_projections)

            # 2. 新增：Judge模型验证逻辑（同步修正变量名）
            future_image_embedded = image_embed(future_images)
            judge_img_feat = judge_image_model(future_image_embedded)
            candidate_actions = [mean.unsqueeze(1)] + outputs['candidates']
            judge_driver_feats = []
            for action in candidate_actions:
                action_emb = motor_embed(action)
                judge_driver_feat = judge_driver_model(action_emb[:, -1, :])
                judge_driver_feats.append(judge_driver_feat)

            match_scores = judge_total_model(judge_img_feat, judge_driver_feats)
            judge_labels = torch.zeros(match_scores.shape[0], dtype=torch.long, device=device)
            judge_ce_loss = F.cross_entropy(match_scores, judge_labels) * config["judge_loss_weight"]

            # 3. 新增：ActionExtract验证逻辑（同步修正变量名）
            pred_action = action_extract_model(judge_img_feat)
            action_extract_loss = -F.mse_loss(pred_action, next_driver) * config["action_extract_loss_weight"]

            # 4. 总损失（去掉sim_loss）
            total_batch_loss = gen_loss + judge_ce_loss + action_extract_loss

            # 累计损失（注释掉相似度损失）
            total_gen_loss += gen_loss.item()
            # total_sim_loss += sim_loss.item()
            total_judge_loss += judge_ce_loss.item()
            total_action_extract_loss += action_extract_loss.item()
            total_loss += total_batch_loss.item()

            # 进度条显示（去掉相似度损失）
            pbar.set_postfix({
                "验证总损失": f"{total_batch_loss.item():.4f}",
                "验证生成损失": f"{gen_loss.item():.4f}",
                # "验证相似度损失": f"{sim_loss.item():.4f}",
                "验证Judge损失": f"{judge_ce_loss.item():.4f}",
                "验证ActionExtract损失": f"{action_extract_loss.item():.4f}"
            })

    # 计算平均损失（注释掉相似度损失）
    avg_gen_loss = total_gen_loss / min(max_val_batches, len(val_loader))
    # avg_sim_loss = total_sim_loss / min(max_val_batches, len(val_loader))
    avg_judge_loss = total_judge_loss / min(max_val_batches, len(val_loader))
    avg_action_extract_loss = total_action_extract_loss / min(max_val_batches, len(val_loader))
    avg_total_loss = total_loss / min(max_val_batches, len(val_loader))

    # 返回值去掉相似度损失
    return avg_total_loss, avg_gen_loss, avg_judge_loss, avg_action_extract_loss


def main():
    best_val_loss = float('inf')
    print("=" * 50)
    print("开始训练（含验证集评估）")
    print(f"总epoch数：{config['epochs']} | 批量大小：{config['batch_size']} | 设备：{device}")
    print(f"loss数据保存目录：{config['loss_data_path']}")
    print("⚠️  相似度模型已注释，仅训练生成器+Judge+ActionExtract模型")
    print("=" * 50)

    # 记录损失（注释掉相似度损失字段）
    loss_records = {
        "train_total": [], "train_gen": [],  # "train_sim": [],
        "val_total": [], "val_gen": [],  # "val_sim": [],
        "train_judge": [], "train_action_extract": [],
        "val_judge": [], "val_action_extract": []
    }

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()

        # 训练（接收值去掉相似度损失）
        train_total, train_gen, train_judge, train_action_extract = train_one_epoch(epoch)
        # 学习率调度
        sch.step()
        # 验证（接收值去掉相似度损失）
        val_total, val_gen, val_judge, val_action_extract = validate_one_epoch(epoch)

        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time

        # 记录损失（注释掉相似度损失）
        loss_records["train_total"].append(train_total)
        loss_records["train_gen"].append(train_gen)
        # loss_records["train_sim"].append(train_sim)
        loss_records["train_judge"].append(train_judge)
        loss_records["train_action_extract"].append(train_action_extract)

        loss_records["val_total"].append(val_total)
        loss_records["val_gen"].append(val_gen)
        # loss_records["val_sim"].append(val_sim)
        loss_records["val_judge"].append(val_judge)
        loss_records["val_action_extract"].append(val_action_extract)

        # 打印epoch信息（去掉相似度损失）
        print("\n" + "=" * 30)
        print(f"Epoch {epoch + 1}/{config['epochs']} | 耗时：{epoch_time:.2f}秒")
        print(
            f"【训练集】总损失：{train_total:.4f} | 生成损失：{train_gen:.4f} | Judge损失：{train_judge:.4f} | ActionExtract损失：{train_action_extract:.4f}")
        print(
            f"【验证集】总损失：{val_total:.4f} | 生成损失：{val_gen:.4f} | Judge损失：{val_judge:.4f} | ActionExtract损失：{val_action_extract:.4f}")
        print("=" * 30 + "\n")

        # 保存最佳模型（注释掉相似度模型的state_dict）
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
                    # 注释掉相似度模型保存
                    # "img_sim_model": img_sim_model.state_dict(),
                    # "driver_sim_model": driver_sim_model.state_dict(),
                    # Judge和ActionExtract模型状态
                    "judge_image_model": judge_image_model.state_dict(),
                    "judge_driver_model": judge_driver_model.state_dict(),
                    "judge_total_model": judge_total_model.state_dict(),
                    "action_extract_model": action_extract_model.state_dict()
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": sch.state_dict(),
                "config": config
            }, best_model_path)

            print(f"✅ 保存最佳模型（验证总损失：{best_val_loss:.4f}）至：{best_model_path}")

    # 保存损失记录
    loss_save_path = os.path.join(config["loss_data_path"], "loss_records.npy")
    np.save(loss_save_path, loss_records)

    # 训练结束总结
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳验证总损失：{best_val_loss:.4f}")
    print(f"最佳模型路径：{os.path.join(config['save_path'], 'best_model')}")
    print("⚠️  相似度模型未参与训练，仅训练生成器+Judge+ActionExtract")
    print("=" * 50)


if __name__ == "__main__":
    main()
