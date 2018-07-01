"""
Definitions for custom blocks for condensenet model
"""
import torch
import torch.nn as nn
from graphs.models.custom_layers.learnedgroupconv import LearnedGroupConv

class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, config):
        super().__init__()

        for layer_id in range(num_layers):
            layer = DenseLayer(in_channels=in_channels + (layer_id * growth_rate), growth_rate=growth_rate, config=config)
            self.add_module('dense_layer_%d' % (layer_id + 1), layer)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, config):
        super().__init__()
        self.config = config
        self.conv_bottleneck = self.config.conv_bottleneck
        self.group1x1 = self.config.group1x1
        self.group3x3 = self.config.group3x3
        self.condense_factor = self.config.condense_factor
        self.dropout_rate = self.config.dropout_rate

        # 1x1 conv in_channels --> bottleneck*growth_rate
        self.conv_1 = LearnedGroupConv(in_channels=in_channels, out_channels=self.conv_bottleneck * growth_rate, kernel_size=1,
                                       groups=self.group1x1, condense_factor=self.condense_factor, dropout_rate=self.dropout_rate)

        self.batch_norm = nn.BatchNorm2d(self.conv_bottleneck * growth_rate)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 conv bottleneck*growth_rate --> growth_rate
        self.conv_2 = nn.Conv2d(in_channels=self.conv_bottleneck * growth_rate, out_channels=growth_rate, kernel_size=3, padding=1, stride=1, groups=self.group3x3, bias=False)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv_2(out)

        return torch.cat([x, out], 1)
