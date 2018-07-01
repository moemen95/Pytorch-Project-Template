"""
DCGAN generator model
based on the paper: https://arxiv.org/pdf/1511.06434.pdf
date: 30 April 2018
"""
import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict
from graphs.weights_initializer import weights_init

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.config.g_input_size, out_channels=self.config.num_filt_g * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(self.config.num_filt_g*8)

        self.deconv2 = nn.ConvTranspose2d(in_channels=self.config.num_filt_g * 8, out_channels=self.config.num_filt_g * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(self.config.num_filt_g*4)

        self.deconv3 = nn.ConvTranspose2d(in_channels=self.config.num_filt_g * 4, out_channels=self.config.num_filt_g * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(self.config.num_filt_g*2)

        self.deconv4 = nn.ConvTranspose2d(in_channels=self.config.num_filt_g * 2, out_channels=self.config.num_filt_g , kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(self.config.num_filt_g)

        self.deconv5 = nn.ConvTranspose2d(in_channels=self.config.num_filt_g, out_channels=self.config.input_channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.out = nn.Tanh()

        self.apply(weights_init)

    def forward(self, x):
        out = self.deconv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.batch_norm2(out)
        out =  self.relu(out)

        out = self.deconv3(out)
        out = self.batch_norm3(out)
        out =  self.relu(out)

        out = self.deconv4(out)
        out = self.batch_norm4(out)
        out =  self.relu(out)

        out = self.deconv5(out)

        out = self.out(out)

        return out


"""
netG testing
"""
def main():
    config = json.load(open('../../configs/dcgan_exp_0.json'))
    config = edict(config)
    inp  = torch.autograd.Variable(torch.randn(config.batch_size, config.g_input_size, 1, 1))
    print (inp.shape)
    netD = Generator(config)
    out = netD(inp)
    print (out.shape)

if __name__ == '__main__':
    main()
