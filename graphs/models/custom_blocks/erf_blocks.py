"""
Custom ERFNet building blocks
Adapted from: https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class non_bottleneck_1d(nn.Module):
    def __init__(self, n_channel, drop_rate, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(n_channel, n_channel, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(n_channel, n_channel, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.conv3x1_2 = nn.Conv2d(n_channel, n_channel, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(n_channel, n_channel, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn1 = nn.BatchNorm2d(n_channel, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(n_channel, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return self.relu(output + input)  # +input = identity (residual connection)


class DownsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel - in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return self.relu(output)


class UpsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return self.relu(output)