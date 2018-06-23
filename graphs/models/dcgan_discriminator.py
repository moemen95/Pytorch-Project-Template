"""
DCGAN discriminator model
based on the paper: https://arxiv.org/pdf/1511.06434.pdf
date: 30 April 2018
"""
import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict
from graphs.weights_initializer import weights_init


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        pass
