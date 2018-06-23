import torch
import torch.nn as nn


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, logits, labels):
        """

        :param logits:
        :param labels:
        :return:
        """
        pass
