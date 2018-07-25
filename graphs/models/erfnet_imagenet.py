# ERFNet encoder model definition used for pretraining in ImageNet
# Sept 2017
# Eduardo Romera
# A
#######################
"""
ERFNet encoder model definition used for pretraining in ImageNet
Sept 2017
Eduardo Romera
Taken from: https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet_imagenet.py
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from graphs.models.custom_blocks.erf_blocks import DownsamplerBlock, non_bottleneck_1d


class ERFNet(nn.Module):
    def __init__(self, num_classes):  # use encoder to pass pretrained encoder
        super().__init__()

        self.features = Features()
        self.classifier = Classifier(num_classes)

    def forward(self, input):
        output = self.features(input)
        output = self.classifier(output)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.extralayer1 = nn.MaxPool2d(2, stride=2)
        self.extralayer2 = nn.AvgPool2d(14, 1, 0)

    def forward(self, input):
        # print("Feat input: ", input.size())
        output = self.encoder(input)
        output = self.extralayer1(output)
        output = self.extralayer2(output)
        # print("Feat output: ", output.size())
        return output


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(128, num_classes)

    def forward(self, input):
        output = input.view(input.size(0), 128)  # first is batch_size
        output = self.linear(output)
        return output

