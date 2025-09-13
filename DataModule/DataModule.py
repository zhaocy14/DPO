import torch
import torchvision.io
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, ConcatDataset
from torch.utils.data import random_split
from torchvision import transforms
import pandas as pd
import matplotlib as plt
import os
from PIL import Image
import numpy as np
import time

class WalkerDataset(Dataset):
    def __init__(self, data_dir: str, frame_len: int = 0, pred_len: int = 5):
        """
        dataset reader for training.
        :param data_dir: the directory of data file, format:"./data/collector_id/date/dataset
                         the collector_id and date should be input by the user
        :param frame_len: number of the consecutive frames
        :param summary: print the summary
        """
        super().__init__()
        self.data_dir = data_dir + os.path.sep + "dataset"  # directory of the dataset
        self.frame_len = frame_len  # observation length for input
        self.predict_len = pred_len  # future length for prediction
        self.csv_dir = self.data_dir + os.path.sep + "meta.csv"
        self.csv_data = pd.read_csv(self.csv_dir)

        # if summary:
        #     self.summary()

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index: int):
        if index >= len(self.csv_data) - self.frame_len - self.predict_len:
            index = len(self.csv_data) - self.frame_len - self.predict_len

        # INPUT
        # Driver input: driver_frames, driver_frames: 12 * 1
        driver_tensors = self.get_driver_tensors(index=index, lens=self.frame_len)
        # Dual images input: img1 sequence, img2 sequence: 2 * 3 * 256 * 256
        imgs1_tensor, imgs2_tensor = self.get_dual_images(index=index, lens=self.frame_len)

        # Leg location from lidar
        # leg_frames = list(map(lambda x: eval(x), self.csv_data.iloc[index:index + self.frame_len, 2]))
        # leg_tensors = self.get_leg_tensors(leg_list=leg_frames, lens=self.frame_len)

        # LABEL
        # label_int = self.get_label(index=index)

        # FUTURE
        future_imgs1, future_imgs2 = self.get_dual_images(index=index + self.frame_len, lens=self.predict_len)
        future_driver = self.get_driver_tensors(index=index + self.frame_len, lens=self.predict_len)
        # future_leg = list(map(lambda x: eval(x),
        #                       self.csv_data.iloc[index + self.frame_len: index + self.frame_len + self.predict_len, 2]))

        return imgs1_tensor, imgs2_tensor, driver_tensors, future_imgs1, future_imgs2, future_driver

    def get_leg_tensors(self, leg_list: list, lens: int) -> torch.Tensor:
        """
        concatenate the tensors of left leg and right leg coordinate tensor.
        Leg coordinate theoretically could vary from [-50, 59] in height and [-19~19] in width.
        Also, to define no user situation, the coordinate will be [-180, -180]
        For formalization, we first see whether the coordinate is [-180, -180]
        Then do formalization
        :param leg_list: self.frame_len * (left_leg_coordinate, right_leg_coordinate)
        :param lens: the len of the time window
        :return: the concatenated leg tensor
        """
        leg_con_tensor = torch.zeros((0,))
        leg_tensor_list = [0, 0, 0, 0]
        for i in range(lens):
            if leg_list[i] == [-180, -180, -180, -180]:
                # TODO: check whether we need to specially assign this to distinguish from user-in-control data
                leg_tensor_list = [-1, -1, -1, -1]
            else:
                leg_tensor_list[0] = leg_list[i][0] / 59  # left leg
                leg_tensor_list[1] = leg_list[i][1] / 19
                leg_tensor_list[2] = leg_list[i][2] / 59  # right leg
                leg_tensor_list[3] = leg_list[i][3] / 19
            one_leg_tensor = torch.tensor(leg_tensor_list)
            leg_con_tensor = torch.cat(tensors=(leg_con_tensor, one_leg_tensor), dim=0)
        return leg_con_tensor

    def get_driver_tensors(self, index: int, lens: int) -> torch.Tensor:
        """
        concatenate the tensors of driver data
        :param index: the starting index
        :return: the concatenated driver tensor
        """
        # if index >= len(self.csv_data) - self.frame_len:
        #     index = len(self.csv_data) - self.frame_len
        driver_con_tensor = torch.zeros((0,))
        for i in range(index, lens + index):
            driver_tuple = eval(self.csv_data.iloc[i, 3])
            driver_tensor = torch.tensor(driver_tuple[-2:])
            driver_con_tensor = torch.cat(tensors=(driver_con_tensor, driver_tensor), dim=0)
        driver_con_tensor = driver_con_tensor.view(lens, 2)
        return driver_con_tensor

    def get_dual_images(self, index: int, lens: int) -> (torch.Tensor, torch.Tensor):
        """
        read dual images from the files
        :param index: index of target frame
        :return: two images tensors
        """
        img1_tensor = torch.zeros((0,))
        img2_tensor = torch.zeros((0,))
        for i in range(index, lens + index):
            # read images 1 from jpg file
            one_img1_tensor = torchvision.io.read_image(self.data_dir + os.path.sep + str(i) + os.path.sep + "RGB1.jpg")
            one_img1_tensor = transforms.Resize([256, 256])(one_img1_tensor)
            one_img1_tensor = one_img1_tensor / 255
            one_img1_tensor = transforms.Normalize([0.34582995, 0.27254773, 0.17211399],
                                                   [0.39859927, 0.36244579, 0.29638241])(one_img1_tensor)
            # one_img1_tensor = transforms.Normalize([0.485, 0.456, 0.406],
            #                                             [0.229, 0.224, 0.225])(one_img1_tensor)
            img1_tensor = torch.cat(tensors=(img1_tensor, one_img1_tensor), dim=0)

            # read images 2 from jpg file
            one_img2_tensor = torchvision.io.read_image(self.data_dir + os.path.sep + str(i) + os.path.sep +
                                                        "RGB2.jpg")
            one_img2_tensor = transforms.Resize([256, 256])(one_img2_tensor)
            one_img2_tensor = one_img2_tensor / 255
            one_img2_tensor = transforms.Normalize([0.34582995, 0.27254773, 0.17211399],
                                                   [0.39859927, 0.36244579, 0.29638241])(one_img2_tensor)
            # one_img2_tensor = transforms.Normalize([0.485, 0.456, 0.406],
            #                                             [0.229, 0.224, 0.225])(one_img2_tensor)
            img2_tensor = torch.cat(tensors=(img2_tensor, one_img2_tensor), dim=0)

        return img1_tensor, img2_tensor


    def summary(self):
        print("*" * 5, "Dataset:", self.data_dir, "*" * 5)
        print("Total number:", self.__len__(), '\n')



class CombinedDataset(object):
    def __init__(self, dir_list: list, frame_len: int = 10, predict_len:int=5, show: bool = False):
        """
        Create a concatenated dataset. Achieve that by calling
        :param dir_list: the list of dataset direction
        :param frame_len: the win-width of the
        :param show: bool value for printing summary of datasets in the list
        """
        # dataset source
        self.datasets_dir = dir_list
        self.frame_len = frame_len
        self.predict_len = predict_len
        self.datasets_list = []

        # for sampler
        self.total_sample_num = 0

        # loading dataset
        self.load_dataset()
        self.concatenated_dataset = self.concat_dataset()
        self.train_len = int(self.concatenated_dataset.__len__() * 0.8)
        self.val_len = self.concatenated_dataset.__len__() - self.train_len
        self.training_dataset, self.val_dataset = random_split(self.concatenated_dataset,
                                                               [self.train_len, self.val_len],
                                                               torch.Generator().manual_seed(42))

        # print out sub dataset and the combined dataset information
        if show:
            self.summary()

    def load_dataset(self):
        for directory in self.datasets_dir:
            self.datasets_list.append(WalkerDataset(data_dir=directory, frame_len=self.frame_len, pred_len=self.predict_len))
        for i in range(len(self.datasets_list)):
            self.total_sample_num += self.datasets_list[i].__len__()

    def concat_dataset(self):
        return ConcatDataset(self.datasets_list)

    def summary(self):
        print("*" * 40)
        print("Number of dataset:", len(self.datasets_list))
        print("Total number:", self.total_sample_num, '\n\n\n\n')


if __name__ == "__main__":
    import platform
    import socket
    from tqdm import tqdm

    if platform.system().lower() == 'windows':
        print("windows")
        dir_root = '../data'
    elif platform.system().lower() == 'linux':
        print("linux")
        hostname = socket.gethostname()
        if hostname == "net-g10":
            dir_root = '/home/ychong/data/cyzhao'
        else:
            dir_root = '/data/cyzhao/collector_cydpo'

    # # parameter:
    batch_size = 1
    summary = True
    num_workers_sampler = 10
    dir_list = []
    for file in os.listdir(dir_root):
        if os.path.isdir(os.path.join(dir_root, file)):
            if "2025" in file:
                dir_list.append(os.path.join(dir_root, file))

    con_dataset = CombinedDataset(dir_list=dir_list, frame_len=15, predict_len=5, show=summary)
    walker_dataloader = DataLoader(dataset=con_dataset.val_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers_sampler)
    print("done!")
    time_start = time.time()
    global_driver_min = torch.tensor([float('inf')] * 2)  # [特征1_min, 特征2_min]
    global_driver_max = torch.tensor([-float('inf')] * 2) # [特征1_max, 特征2_max]
    for img1, img2, driver, img1_fut, img2_fu, driver_fu in tqdm(walker_dataloader):
        driver_flatten = driver.view(-1, 2)

        # 2. 计算当前 batch 内的最大/最小值（按特征维度计算，即 dim=0）
        batch_min = driver_flatten.min(dim=0)[0]  # 每个特征的 batch 最小值
        batch_max = driver_flatten.max(dim=0)[0]  # 每个特征的 batch 最大值

        # 3. 更新全局最大/最小值（取「全局最值」和「当前 batch 最值」的极值）
        global_driver_min = torch.min(global_driver_min, batch_min)
        global_driver_max = torch.max(global_driver_max, batch_max)
        # print(img1.shape)
        pass
    print("Loading data time:", time.time() - time_start)
    print("\n=== Driver 数据全局最大/最小值 ===")
    print(f"特征1（第一个维度）: 最小值 = {global_driver_min[0].item():.6f}, 最大值 = {global_driver_max[0].item():.6f}")
    print(f"特征2（第二个维度）: 最小值 = {global_driver_min[1].item():.6f}, 最大值 = {global_driver_max[1].item():.6f}")
