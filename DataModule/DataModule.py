import os
import sys
import time
import torch
import torchvision.io
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms

# 设置路径
pwd = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)


class WalkerDataset(Dataset):
    def __init__(self, data_dir: str, frame_len: int = 10, pred_len: int = 5, step: int = 3):
        """
        dataset reader for training with step parameter to control frame间隔
        :param data_dir: 数据目录
        :param frame_len: 输入的连续帧数量
        :param pred_len: 预测的未来帧数量
        :param step: 相邻样本之间的间隔帧数
        """
        super().__init__()
        self.data_dir = os.path.join(data_dir, "dataset")  # 数据集目录
        self.frame_len = frame_len  # 输入的观测长度
        self.predict_len = pred_len  # 预测的未来长度
        self.step = step  # 相邻样本的间隔帧数
        self.csv_dir = os.path.join(self.data_dir, "meta.csv")
        self.csv_data = pd.read_csv(self.csv_dir)

        # 图像变换预处理 - 提前定义以提高效率
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.34582995, 0.27254773, 0.17211399],
                                 [0.39859927, 0.36244579, 0.29638241])
        ])

        # 图像缓存 - 减少重复读取
        self.image_cache = {}
        self.cache_size = 1000  # 缓存大小限制

    def __len__(self):
        # 根据步长调整有效样本数量
        max_start_idx = len(self.csv_data) - self.frame_len * self.step - self.predict_len * self.step
        if max_start_idx <= 0:
            return 0
        return max(0, (max_start_idx // self.step) + 1)

    def __getitem__(self, index: int):
        # 将索引映射到实际的起始帧，考虑步长
        actual_index = index * self.step

        # 确保不会超出数据范围
        max_actual_index = len(self.csv_data) - self.frame_len * self.step - self.predict_len * self.step
        if actual_index > max_actual_index:
            actual_index = max_actual_index

        # 输入数据
        driver_tensors = self.get_driver_tensors(start_idx=actual_index, lens=self.frame_len)
        imgs1_tensor, imgs2_tensor = self.get_dual_images(start_idx=actual_index, lens=self.frame_len)

        # 未来数据
        future_start_idx = actual_index + self.frame_len * self.step
        future_imgs1, future_imgs2 = self.get_dual_images(start_idx=future_start_idx, lens=self.predict_len)
        future_driver = self.get_driver_tensors(start_idx=future_start_idx, lens=self.predict_len)

        return imgs1_tensor, imgs2_tensor, driver_tensors, future_imgs1, future_imgs2, future_driver

    def get_driver_tensors(self, start_idx: int, lens: int) -> torch.Tensor:
        """获取驾驶员数据张量，考虑步长"""
        driver_data = []
        for i in range(lens):
            # 按照步长获取数据
            current_idx = start_idx + i * self.step
            driver_tuple = eval(self.csv_data.iloc[current_idx, 3])
            driver_vals = torch.tensor(driver_tuple[-2:], dtype=torch.float32)

            # 数据归一化
            driver_vals = torch.clamp(driver_vals, min=-150.0, max=50.0)
            driver_vals = (driver_vals + 50) / 100  # 映射到[-1, 1]范围
            driver_data.append(driver_vals)

        return torch.stack(driver_data)

    def get_dual_images(self, start_idx: int, lens: int) -> (torch.Tensor, torch.Tensor):
        """获取双摄像头图像张量，使用缓存优化"""
        imgs1 = []
        imgs2 = []

        for i in range(lens):
            current_idx = start_idx + i * self.step
            img1_path = os.path.join(self.data_dir, str(current_idx), "RGB1.jpg")
            img2_path = os.path.join(self.data_dir, str(current_idx), "RGB2.jpg")

            # 尝试从缓存获取图像
            if img1_path in self.image_cache:
                img1_tensor = self.image_cache[img1_path]
            else:
                img1_tensor = torchvision.io.read_image(img1_path)
                img1_tensor = self.transform(img1_tensor)
                # 缓存图像，超出缓存大小时清理
                if len(self.image_cache) >= self.cache_size:
                    self.image_cache.pop(next(iter(self.image_cache)))
                self.image_cache[img1_path] = img1_tensor

            if img2_path in self.image_cache:
                img2_tensor = self.image_cache[img2_path]
            else:
                img2_tensor = torchvision.io.read_image(img2_path)
                img2_tensor = self.transform(img2_tensor)
                if len(self.image_cache) >= self.cache_size:
                    self.image_cache.pop(next(iter(self.image_cache)))
                self.image_cache[img2_path] = img2_tensor

            imgs1.append(img1_tensor)
            imgs2.append(img2_tensor)

        return torch.stack(imgs1), torch.stack(imgs2)


class CombinedDataset(object):
    def __init__(self, dir_list: list, frame_len: int = 10, predict_len: int = 5,
                 step: int = 3, show: bool = False):
        """组合多个数据集"""
        self.datasets_dir = dir_list
        self.frame_len = frame_len
        self.predict_len = predict_len
        self.step = step  # 添加步长参数
        self.datasets_list = []
        self.total_sample_num = 0

        self.load_dataset()
        self.concatenated_dataset = self.concat_dataset()

        # 划分训练集和验证集
        self.train_len = int(len(self.concatenated_dataset) * 0.8)
        self.val_len = len(self.concatenated_dataset) - self.train_len
        self.training_dataset, self.val_dataset = random_split(
            self.concatenated_dataset,
            [self.train_len, self.val_len],
            torch.Generator().manual_seed(42)
        )

        if show:
            self.summary()

    def load_dataset(self):
        """加载数据集列表"""
        for directory in self.datasets_dir:
            dataset = WalkerDataset(
                data_dir=directory,
                frame_len=self.frame_len,
                pred_len=self.predict_len,
                step=self.step  # 传递步长参数
            )
            if len(dataset) > 0:  # 只添加有有效样本的数据集
                self.datasets_list.append(dataset)
                self.total_sample_num += len(dataset)

    def concat_dataset(self):
        """拼接数据集"""
        return ConcatDataset(self.datasets_list)

    def summary(self):
        """打印数据集摘要信息"""
        print("*" * 40)
        print(f"数据集数量: {len(self.datasets_list)}")
        print(f"总样本数量: {self.total_sample_num}")
        print(f"步长设置: {self.step} 帧")
        print(f"训练集样本数: {self.train_len}")
        print(f"验证集样本数: {self.val_len}")
        print("*" * 40)


if __name__ == "__main__":
    import platform
    import socket
    from tqdm import tqdm

    # 根据操作系统设置数据根目录
    if platform.system().lower() == 'windows':
        print("运行在Windows系统")
        dir_root = '../data'
    elif platform.system().lower() == 'linux':
        print("运行在Linux系统")
        hostname = socket.gethostname()
        if hostname == "net-g10":
            dir_root = '/home/ychong/data/cyzhao'
        else:
            dir_root = '/data/cyzhao/collector_cydpo'

    # 参数设置
    batch_size = 16
    summary = True
    num_workers = 8  # 适当调整工作进程数，过多可能导致内存问题
    frame_len = 15
    predict_len = 5
    step = 3  # 相邻样本间隔3帧，可以根据需要调整

    # 收集数据目录
    dir_list = []
    for file in os.listdir(dir_root):
        dir_path = os.path.join(dir_root, file)
        if os.path.isdir(dir_path) and "2025" in file:
            dir_list.append(dir_path)

    # 创建组合数据集
    con_dataset = CombinedDataset(
        dir_list=dir_list,
        frame_len=frame_len,
        predict_len=predict_len,
        step=step,  # 设置步长
        show=summary
    )

    # 创建数据加载器
    walker_dataloader = DataLoader(
        dataset=con_dataset.val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # 提高GPU加载速度
    )

    print(f"验证集长度: {len(con_dataset.val_dataset)}")
    print(f"训练集长度: {len(con_dataset.training_dataset)}")

    # 测试数据加载速度并计算driver数据的统计信息
    time_start = time.time()
    global_driver_min = torch.tensor([float('inf')] * 2)
    global_driver_max = torch.tensor([-float('inf')] * 2)

    for img1, img2, driver, img1_fut, img2_fu, driver_fu in tqdm(walker_dataloader):
        # 计算driver特征的全局最大最小值
        driver_flatten = driver.view(-1, 2)
        batch_min = driver_flatten.min(dim=0)[0]
        batch_max = driver_flatten.max(dim=0)[0]

        global_driver_min = torch.min(global_driver_min, batch_min)
        global_driver_max = torch.max(global_driver_max, batch_max)

    print(f"数据加载时间: {time.time() - time_start:.2f}秒")
    print("\n=== Driver 数据全局最大/最小值 ===")
    print(f"特征1: 最小值 = {global_driver_min[0].item():.6f}, 最大值 = {global_driver_max[0].item():.6f}")
    print(f"特征2: 最小值 = {global_driver_min[1].item():.6f}, 最大值 = {global_driver_max[1].item():.6f}")
