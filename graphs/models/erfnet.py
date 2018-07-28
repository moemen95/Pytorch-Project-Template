"""
ERFNet Model
July 2017
Adapted from: https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
"""
import torch.nn as nn

from graphs.models.custom_layers.erf_blocks import DownsamplerBlock, UpsamplerBlock, non_bottleneck_1d


class ERF(nn.Module):
    def __init__(self, config, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        self.config = config
        self.num_classes = self.config.num_classes

        if encoder == None:
            self.encoder_flag = True
            self.encoder_layers = nn.ModuleList()

            # layer 1, downsampling
            self.initial_block = DownsamplerBlock(self.config.input_channels, 16)

            # layer 2, downsampling
            self.encoder_layers.append(DownsamplerBlock(in_channel=16, out_channel=64))

            # non-bottleneck 1d - layers 3 to 7
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))

            # layer 8, downsampling
            self.encoder_layers.append(DownsamplerBlock(in_channel=64, out_channel=128))

            # non-bottleneck 1d - layers 9 to 16
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16))

        else:
            self.encoder_flag = False
            self.encoder = encoder

        self.decoder_layers = nn.ModuleList()

        self.decoder_layers.append(UpsamplerBlock(in_channel=128, out_channel=64))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1))

        self.decoder_layers.append(UpsamplerBlock(in_channel=64, out_channel=16))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1))

        self.output_conv = nn.ConvTranspose2d(in_channels=16, out_channels=self.num_classes,kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        # self.apply(weights_init_normal)

    def forward(self, x):
        if self.encoder_flag:
            output = self.initial_block(x)

            for layer in self.encoder_layers:
                output = layer(output)
        else:
            output = self.encoder(x)

        for layer in self.decoder_layers:
            output = layer(output)

        output = self.output_conv(output)

        return output
