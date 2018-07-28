"""
Cross Entropy 2D for CondenseNet
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self, config=None):
        super(CrossEntropyLoss, self).__init__()
        if config == None:
            self.loss = nn.CrossEntropyLoss()
        else:
            class_weights = np.load(config.class_weights)
            self.loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index,
                                      weight=torch.from_numpy(class_weights.astype(np.float32)),
                                      size_average=True, reduce=True)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)