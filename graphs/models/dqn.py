"""
Deep Q Network model
based on the paper: https://www.nature.com/articles/nature14236
date: 1st of June 2018
"""
import torch
import torch.nn as nn

from graphs.weights_initializer import weights_init


class DQN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.config.conv_filters[0], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.config.conv_filters[0])

        self.conv2 = nn.Conv2d(in_channels=self.config.conv_filters[0], out_channels=self.config.conv_filters[1], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(self.config.conv_filters[1])

        self.conv3 = nn.Conv2d(in_channels=self.config.conv_filters[1], out_channels=self.config.conv_filters[2], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(self.config.conv_filters[2])

        self.linear = nn.Linear(448, self.config.num_classes)

        self.apply(weights_init)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        out = self.linear(x.view(x.size(0), -1))

        return out


"""
#########################
Architecture:
#########################
Input: (N, 3, 40, 80)

Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2))
Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2))
Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
Linear(in_features=448, out_features=2, bias=True)
----
torch.Size([128, 3, 40, 80])
torch.Size([128, 16, 18, 38])
torch.Size([128, 32, 7, 17])
torch.Size([128, 32, 2, 7]) --> 32x2x7 = 448
torch.Size([128, 2])

"""
