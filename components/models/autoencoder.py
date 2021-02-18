import torch
import torch.nn as nn
import numpy as np
from glob import glob
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    """
    Autoencoder, as used in the paper: http://doi.org/10.1109/TMI.2020.2968472
    """
    def __init__(self, crop_size, vol=False,
                 final_activation=False, rgb=True):
        super(AutoEncoder, self).__init__()

        # Variables
        self.rgb = rgb
        self.final_activation = final_activation

        # Kernel
        kernel = 3

        # Input size
        if rgb:
            f_maps = 3  # RGB
        else:
            f_maps = 1  # One-channel
        self.input_size = (f_maps,) + crop_size

        # Construct the layers

        # Cube/square size 64 to 32
        self.encoder1 = nn.Sequential(*self._layer([f_maps, 64], kernel=kernel, use_3d=vol, encoder=True))
        self.encoder2 = nn.Sequential(*self._layer([64, 32], kernel=kernel, use_3d=vol, encoder=True))  # 32 to 16
        self.encoder3 = nn.Sequential(*self._layer([32, 16], kernel=kernel, use_3d=vol, encoder=True))  # 16 to 8
        self.encoder4 = nn.Sequential(*self._layer([16, 8], kernel=kernel, use_3d=vol, encoder=True))  # 8 to 4

        # 4x4x4 cube at the bottleneck. Calculate the total feature size
        self.output_size = self.encoder(Variable(torch.ones(1, *self.input_size)))
        self.bottleneck_fts = int(np.prod(self.output_size.size()[1:]))

        # Linear layer at bottleneck
        linear = [
            nn.Linear(self.bottleneck_fts, self.bottleneck_fts // 2),
            nn.BatchNorm1d(self.bottleneck_fts // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_fts // 2, self.bottleneck_fts),
            nn.BatchNorm1d(self.bottleneck_fts),
            nn.ReLU(inplace=True)
        ]
        self.linear = nn.Sequential(*linear)
        
        # Mirror the encoder
        decoder = list()
        decoder.extend(self._layer([8, 16], kernel=kernel, use_3d=vol, encoder=False))  
        decoder.extend(self._layer([16, 32], kernel=kernel, use_3d=vol, encoder=False)) 
        decoder.extend(self._layer([32, 64], kernel=kernel, use_3d=vol, encoder=False)) 
        # Last upscaling layer
        if vol:
            decoder.extend([nn.ConvTranspose3d(64, f_maps, kernel_size=2, stride=2)])
        else:
            decoder.extend([nn.ConvTranspose2d(64, f_maps, kernel_size=2, stride=2)])
        decoder.extend([nn.Sigmoid()])

        # Compile
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        # Pass through the model
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x = self.linear(x.view(-1, self.bottleneck_fts))
        x = self.decoder(x.view((-1,) + self.output_size.size()[1:]))

        # Duplicate 1-channel image to represent RGB
        if self.rgb:
            if len(x.size()) == 5:
                x = x.repeat(1, 3, 1, 1, 1)
            else:
                x = x.repeat(1, 3, 1, 1)

        # Scaled Tanh activation
        if self.final_activation:
            x = x.tanh()
        return x

    @staticmethod
    def _layer(f_maps, kernel=3, use_3d=True, encoder=True):

        pad = kernel // 2

        if use_3d:
            layers = [
                nn.Conv3d(f_maps[0], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.BatchNorm3d(f_maps[1], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[1], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.BatchNorm3d(f_maps[1], affine=True),
                nn.ReLU(inplace=True),
            ]
            # Downscale or upscale
            if encoder:
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            else:
                layers.append(nn.ConvTranspose3d(f_maps[1], f_maps[1], kernel_size=2, stride=2))
        else:
            layers = [
                nn.Conv2d(f_maps[0], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.BatchNorm2d(f_maps[1], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[1], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.BatchNorm2d(f_maps[1], affine=True),
                nn.ReLU(inplace=True),
            ]
            # Downscale or upscale
            if encoder:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.ConvTranspose2d(f_maps[1], f_maps[1], kernel_size=2, stride=2))

        return layers


class AutoEncoderLayers(AutoEncoder):
    def __init__(self, vol=False, final_activation=False, rgb=True):
        super(AutoEncoderLayers, self).__init__(vol=vol, final_activation=final_activation, rgb=rgb)

    def forward(self, x):
        # Pass through the model
        layer_1 = self.encoder1(x)
        layer_2 = self.encoder2(layer_1)
        layer_3 = self.encoder3(layer_2)
        layer_4 = self.encoder4(layer_3)

        # Duplicate 1-channel image to represent RGB
        if self.rgb:
            if len(x.size()) == 5:
                layer_1 = layer_1.repeat(1, 3, 1, 1, 1)
                layer_2 = layer_2.repeat(1, 3, 1, 1, 1)
                layer_3 = layer_3.repeat(1, 3, 1, 1, 1)
                layer_4 = layer_4.repeat(1, 3, 1, 1, 1)
            else:
                layer_1 = layer_1.repeat(1, 3, 1, 1)
                layer_2 = layer_2.repeat(1, 3, 1, 1)
                layer_3 = layer_3.repeat(1, 3, 1, 1)
                layer_4 = layer_4.repeat(1, 3, 1, 1)

        # Scaled Tanh activation
        if self.final_activation:
            layer_1 = layer_1.tanh()
            layer_2 = layer_2.tanh()
            layer_3 = layer_3.tanh()
            layer_4 = layer_4.tanh()
        return {'layer_1': layer_1, 'layer_2': layer_2, 'layer_3': layer_3, 'layer_4': layer_4}


def load_models(model_path, vol=False, rgb=False, gpus=1, fold=None):
    # Load models
    if fold is not None:
        models = glob(model_path + f'/*fold_{fold}*.pth')
    else:
        models = glob(model_path + '/*fold_*.pth')
    models.sort()

    # List the models
    model_list = []
    for fold in range(len(models)):
        if gpus > 1:
            model = nn.DataParallel(AutoEncoder(vol=vol, rgb=rgb))
        else:
            model = AutoEncoder(vol=vol, rgb=rgb)
        model.load_state_dict(torch.load(models[fold]))
        model_list.append(model)

    return model_list[0]
