import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
        Originally introduced in (Microsoft Research Asia, He et al.): https://arxiv.org/abs/1512.03385
        Based on implementation from https://github.com/gordicaleksa/pytorch-nst-feedforward
    """

    def __init__(self, channels, norm='bn'):
        super(ResidualBlock, self).__init__()
        stride = 1
        kernel = 3
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=1, bias=False)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(channels, affine=True)
            self.n2 = nn.BatchNorm2d(channels, affine=True)
        elif norm == 'in':
            self.n1 = nn.InstanceNorm2d(channels, affine=True)
            self.n2 = nn.InstanceNorm2d(channels, affine=True)
        else:
            raise Exception('Normalization not implemented!')
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        out = out + residual  # Johnson et al use no ReLU after addition
        return out


class PerceptualNet(nn.Module):
    """
    Super resolution network used in 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'
    by Johnson et al, https://arxiv.org/abs/1603.08155
    """
    def __init__(self, magnification, activation='relu', resize_convolution=False, norm='bn'):
        super(PerceptualNet, self).__init__()

        # Variables
        self.magnification = magnification
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise Exception('Not implemented activation function!')
        # Normalization
        if norm == 'bn':
            self.norm = nn.BatchNorm2d
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d
        else:
            raise Exception('Not implemented normalization!')
        # Kernel
        kernel = 3
        pad = kernel // 2

        # Block types
        f_maps = [3, 64, 1]
        first_block = [
            nn.Conv2d(f_maps[0], f_maps[1], kernel_size=kernel, stride=1, padding=pad),  # Changed bias to true
            self.norm(f_maps[1], affine=True),
            self.activation
        ]
        mid_block = [
            ResidualBlock(f_maps[1], norm=norm),
            ResidualBlock(f_maps[1], norm=norm),
            ResidualBlock(f_maps[1], norm=norm),
            ResidualBlock(f_maps[1], norm=norm)
        ]
        if resize_convolution:
            upscale_block = [
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(f_maps[1], f_maps[1], kernel_size=3, stride=1, padding=0),
                self.norm(f_maps[1], affine=True),
                self.activation
            ]
        else:
            upscale_block = [
                nn.ConvTranspose2d(f_maps[1], f_maps[1], kernel_size=4, stride=2, padding=1, output_padding=0),
                self.norm(f_maps[1], affine=True),
                self.activation
            ]

        final_block = [
            nn.Conv2d(f_maps[1], f_maps[2], kernel_size=kernel, stride=1, padding=pad),
            self.norm(f_maps[2], affine=True),
            self.activation
        ]

        # Construct the layers
        layers = []
        layers.extend(first_block)
        layers.extend(mid_block)
        for _ in range(int(np.log2(magnification))):
            layers.extend(upscale_block)
        layers.extend(final_block)

        # Compile
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        # Pass through the model
        x = self.net(x)

        # Duplicate 1-channel image to represent RGB
        x = x.repeat(1, 3, 1, 1)

        # Scaled Tanh activation
        x = x.tanh()
        x = x.add(1.)
        x = x.mul(0.5)
        return x
