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

        self.relu = nn.LeakyReLU(self.config.relu_slope, inplace=True)

        self.conv1 = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.config.num_filt_d, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=self.config.num_filt_d, out_channels=self.config.num_filt_d * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(self.config.num_filt_d*2)

        self.conv3 = nn.Conv2d(in_channels=self.config.num_filt_d*2, out_channels=self.config.num_filt_d * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(self.config.num_filt_d*4)

        self.conv4 = nn.Conv2d(in_channels=self.config.num_filt_d*4, out_channels=self.config.num_filt_d*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(self.config.num_filt_d*8)

        self.conv5 = nn.Conv2d(in_channels=self.config.num_filt_d*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)

        self.out = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm1(out)
        out =  self.relu(out)

        out = self.conv3(out)
        out = self.batch_norm2(out)
        out =  self.relu(out)

        out = self.conv4(out)
        out = self.batch_norm3(out)
        out =  self.relu(out)

        out = self.conv5(out)
        out = self.out(out)

        return out.view(-1, 1).squeeze(1)


"""
netD testing
"""
def main():
    config = json.load(open('../../configs/dcgan_exp_0.json'))
    config = edict(config)
    inp  = torch.autograd.Variable(torch.randn(config.batch_size, config.input_channels, config.image_size, config.image_size))
    print (inp.shape)
    netD = Discriminator(config)
    out = netD(inp)
    print (out)

if __name__ == '__main__':
    main()
