import numpy as np

import imageio

import torch
import torchvision.transforms as v_transforms
import torchvision.utils as v_utils
import torchvision.datasets as v_datasets

from torch.utils.data import DataLoader, TensorDataset, Dataset


class CelebADataLoader:
    def __init__(self, config):
        """

        :param config:
        """
        self.config = config

    def plot_samples_per_epoch(self, fake_batch, epoch):
        """
        Plotting the fake batch
        :param fake_batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        pass

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        pass

    def finalize(self):
        """

        :return:
        """
        pass
