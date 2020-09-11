import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from BoneEnhance.components.models import make_layers


class ConvNet(nn.Module):
    def __init__(self, magnification, n_blocks=18, upscale_input=False, activation='relu', normalization=None):
        super(ConvNet, self).__init__()

        # Variables
        self.upscale_input = upscale_input
        self.magnification = magnification
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise Exception('Not implemented activation function!')

        # Block types
        f_maps = [3, 128, 1]
        first_block = [
            nn.Conv2d(f_maps[0], f_maps[1], kernel_size=3, stride=1, padding=1, bias=False),
            self.activation
        ]
        if normalization == 'bn':
            mid_block = [
                nn.Conv2d(f_maps[1], f_maps[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(f_maps[1]),
                self.activation
            ]
            upscale_block = [
                nn.ConvTranspose2d(f_maps[1], f_maps[1], kernel_size=4, stride=2, padding=1, output_padding=0,
                                   bias=False),
                nn.BatchNorm2d(f_maps[1]),
                self.activation,
                nn.Conv2d(f_maps[1], f_maps[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(f_maps[1]),
                self.activation
            ]
        else:
            mid_block = [
                nn.Conv2d(f_maps[1], f_maps[1], kernel_size=3, stride=1, padding=1, bias=False),
                self.activation
            ]
            upscale_block = [
                nn.ConvTranspose2d(f_maps[1], f_maps[1], kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
                self.activation,
                nn.Conv2d(f_maps[1], f_maps[1], kernel_size=3, stride=1, padding=1, bias=False),
                self.activation
            ]
        final_block = [
            nn.Conv2d(f_maps[1], f_maps[2], kernel_size=3, stride=1, padding=1, bias=False),
        ]

        # Construct the layers
        layers = []
        layers.extend(first_block)
        for _ in range(n_blocks):
            layers.extend(mid_block)
        if not upscale_input:
            for _ in range(int(np.log2(magnification))):
                layers.extend(upscale_block)
        layers.extend(final_block)

        # Compile
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Upscale to target size
        if self.upscale_input:
            x = F.interpolate(x, scale_factor=self.magnification)
            input_im = x.detach().clone()

        # Pass through the model
        x = self.net(x)

        # Duplicate 1-channel image to represent RGB
        x = x.repeat(1, 3, 1, 1)

        if self.upscale_input:
            #x = x.tanh()
            x = input_im + x
        else:
            x = x.sigmoid()
        return x
