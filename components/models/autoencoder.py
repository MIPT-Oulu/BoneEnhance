import torch
import torch.nn as nn
from glob import glob


class AutoEncoder(nn.Module):
    """
    Autoencoder, as used in the paper: http://doi.org/10.1109/TMI.2020.2968472
    """
    def __init__(self, vol=False,
                 final_activation=False, rgb=True):
        super(AutoEncoder, self).__init__()

        # Variables
        self.rgb = rgb
        self.final_activation = final_activation

        # Kernel
        if rgb:
            f_maps = [3, 64, 128, 256]  # RGB
        else:
            f_maps = [1, 64, 128, 256]  # One-channel
        kernel = 3
        pad = kernel // 2

        # Construct the layers
        if vol:
            encoder_1 = [
                nn.Conv3d(f_maps[0], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[1], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                
                nn.MaxPool3d(kernel_size=2, stride=2),
                
                nn.Conv3d(f_maps[1], f_maps[2], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[2], f_maps[2], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                ]

            encoder_2 = [
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(f_maps[2], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
            ]
            decoder = [
                nn.Conv3d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose3d(f_maps[3], f_maps[2], kernel_size=2, stride=2),
                nn.ReLU(inplace=True),

                nn.Conv3d(f_maps[2], f_maps[2], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[2], f_maps[2], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),

                nn.ConvTranspose3d(f_maps[2], f_maps[1], kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                
                nn.Conv3d(f_maps[1], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_maps[1], 1, kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
            ]
        else:
            encoder_1 = [
                nn.Conv2d(f_maps[0], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[1], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(f_maps[1], f_maps[2], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[2], f_maps[2], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),]
            encoder_2 = [
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(f_maps[2], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
            ]
            decoder = [
                nn.Conv2d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[3], f_maps[3], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(f_maps[3], f_maps[2], kernel_size=2, stride=2),
                nn.ReLU(inplace=True),

                nn.Conv2d(f_maps[2], f_maps[2], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[2], f_maps[2], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(f_maps[2], f_maps[1], kernel_size=2, stride=2),
                nn.ReLU(inplace=True),

                nn.Conv2d(f_maps[1], f_maps[1], kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(f_maps[1], 1, kernel_size=kernel, stride=1, padding=pad, bias=True),
                nn.ReLU(inplace=True),
            ]
            
        # Compile
        self.encoder_1 = nn.Sequential(*encoder_1)
        self.encoder_2 = nn.Sequential(*encoder_2)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        # Pass through the model
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.decoder(x)

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


class AutoEncoderLayers(AutoEncoder):
    def __init__(self, vol=False, final_activation=False, rgb=True):
        super(AutoEncoderLayers, self).__init__(vol=vol, final_activation=final_activation, rgb=rgb)

    def forward(self, x):
        # Pass through the model
        layer_1 = self.encoder_1(x)
        layer_2 = self.encoder_2(layer_1)

        # Duplicate 1-channel image to represent RGB
        if self.rgb:
            if len(x.size()) == 5:
                layer_1 = layer_1.repeat(1, 3, 1, 1, 1)
                layer_2 = layer_2.repeat(1, 3, 1, 1, 1)
            else:
                layer_1 = layer_1.repeat(1, 3, 1, 1)
                layer_2 = layer_2.repeat(1, 3, 1, 1)

        # Scaled Tanh activation
        if self.final_activation:
            layer_1 = layer_1.tanh()
            layer_2 = layer_2.tanh()
        return {'layer_1': layer_1, 'layer_2': layer_2}


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
