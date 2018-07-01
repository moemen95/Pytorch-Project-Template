"""
CondenseNet Model
name: condensenet.py
date: May 2018
"""
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np

from graphs.weights_initializer import init_model_weights
from graphs.models.custom_layers.denseblock import DenseBlock
from graphs.models.custom_layers.learnedgroupconv import LearnedGroupConv

class CondenseNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.stages = self.config.stages
        self.growth_rate = self.config.growth_rate
        assert len(self.stages) == len(self.growth_rate)

        self.init_stride = self.config.init_stride
        self.pool_size = self.config.pool_size
        self.num_classes = self.config.num_classes

        self.progress = 0.0
        self.num_filters = 2 * self.growth_rate[0]
        """
        Initializing layers
        """
        self.transition_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.pool = nn.AvgPool2d(self.pool_size)
        self.relu = nn.ReLU(inplace=True)

        self.init_conv = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.num_filters, kernel_size=3, stride=self.init_stride, padding=1, bias=False)

        self.denseblock_one = DenseBlock(num_layers=self.stages[0], in_channels= self.num_filters, growth_rate=self.growth_rate[0], config=self.config)

        self.num_filters += self.stages[0] * self.growth_rate[0]

        self.denseblock_two = DenseBlock(num_layers=self.stages[1], in_channels= self.num_filters, growth_rate=self.growth_rate[1], config=self.config)

        self.num_filters += self.stages[1] * self.growth_rate[1]

        self.denseblock_three = DenseBlock(num_layers=self.stages[2], in_channels= self.num_filters, growth_rate=self.growth_rate[2], config=self.config)

        self.num_filters += self.stages[2] * self.growth_rate[2]
        self.batch_norm = nn.BatchNorm2d(self.num_filters)

        self.classifier = nn.Linear(self.num_filters, self.num_classes)

        self.apply(init_model_weights)

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress

        x = self.init_conv(x)

        x = self.denseblock_one(x)
        x = self.transition_pool(x)

        x = self.denseblock_two(x)
        x = self.transition_pool(x)

        x = self.denseblock_three(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        out = self.classifier(x)

        return out

"""
#########################
Model Architecture:
#########################

Input: (N, 32, 32, 3)

- Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
DenseBlock(num_layers=14, in_channels=16, growth_rate=8)
- AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
DenseBlock(num_layers=14, in_channels=128, growth_rate=16)
- AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
DenseBlock(num_layers=14, in_channels=352, growth_rate=32)
- BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True)
- ReLU(inplace)
- AvgPool2d(kernel_size=8, stride=8, padding=0, ceil_mode=False, count_include_pad=True)
- Linear(in_features=800, out_features=10, bias=True)
"""
