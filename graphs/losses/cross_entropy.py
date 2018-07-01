"""
Cross Entropy 2D for CondenseNet
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss
