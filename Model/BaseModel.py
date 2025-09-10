import torch
import torch.nn as nn
import torch.func as F

class ImageEmbedding(nn.Module):
    def __init__(self, layer_num):
        super(ImageEmbedding, self).__init__()
        self.layer_num = layer_num

    def _cnn_construct(self):
        """
        Construct the CNN layers
        :return:
        """
        pass


        