import torch.nn as nn
import torch
from torch.nn import Conv3d, ReLU, MaxPool3d, Parameter
from torchvision import models


class Vgg16(nn.Module):
    """
    Based on implementation from https://github.com/gordicaleksa/pytorch-nst-feedforward,
    Original paper: https://arxiv.org/pdf/1603.08155.pdf
    """
    def __init__(self, vol=False, zeros=False):
        super().__init__()
        # Freeze coefficients
        vgg16 = models.vgg16(pretrained=True).eval()
        vgg_pretrained_features = vgg16.features
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.vol = vol

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # Parameters
        ks = 3
        f_maps = [3, 64, 128, 256, 512]

        if vol:

            layers_3d = [
                Conv3d(f_maps[0], f_maps[1], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                Conv3d(f_maps[1], f_maps[1], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                Conv3d(f_maps[1], f_maps[2], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                Conv3d(f_maps[2], f_maps[2], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                Conv3d(f_maps[2], f_maps[3], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                Conv3d(f_maps[3], f_maps[3], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                Conv3d(f_maps[3], f_maps[3], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                Conv3d(f_maps[3], f_maps[4], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                Conv3d(f_maps[4], f_maps[4], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                Conv3d(f_maps[4], f_maps[4], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                Conv3d(f_maps[4], f_maps[4], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                Conv3d(f_maps[4], f_maps[4], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                Conv3d(f_maps[4], f_maps[4], kernel_size=ks, stride=1, padding=1),
                ReLU(inplace=True),
                MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            ]

            # TODO Obtained features should look similar to 2D

            # Use torch.no_grad() to prevent Autograd from tracking the weight changes
            with torch.no_grad():
                for x in range(4):
                    self.slice1.add_module(str(x), layers_3d[x])
                    if isinstance(layers_3d[x], nn.Conv3d):
                        if zeros:
                            weights = torch.zeros(vgg_pretrained_features[x].weight.size() + (3,))
                            weights[:, :, :, :, 1] = vgg_pretrained_features[x].weight
                        else:
                            weights = vgg_pretrained_features[x].weight.unsqueeze(4).repeat(1, 1, 1, 1, 3)#.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                        self.slice1[x].weight = Parameter(weights)
                for x in range(4, 9):
                    self.slice2.add_module(str(x), layers_3d[x])
                    if isinstance(layers_3d[x], nn.Conv3d):
                        if zeros:
                            weights = torch.zeros(vgg_pretrained_features[x].weight.size() + (3,))
                            weights[:, :, :, :, 1] = vgg_pretrained_features[x].weight
                        else:
                            weights = vgg_pretrained_features[x].weight.unsqueeze(4).repeat(1, 1, 1, 1, 3)#.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                        self.slice2[x - 4].weight = Parameter(weights)
                for x in range(9, 16):
                    self.slice3.add_module(str(x), layers_3d[x])
                    if isinstance(layers_3d[x], nn.Conv3d):
                        if zeros:
                            weights = torch.zeros(vgg_pretrained_features[x].weight.size() + (3,))
                            weights[:, :, :, :, 1] = vgg_pretrained_features[x].weight
                        else:
                            weights = vgg_pretrained_features[x].weight.unsqueeze(4).repeat(1, 1, 1, 1, 3)#.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                        self.slice3[x - 9].weight = Parameter(weights)
                for x in range(16, 23):
                    self.slice4.add_module(str(x), layers_3d[x])
                    if isinstance(layers_3d[x], nn.Conv3d):
                        if zeros:
                            weights = torch.zeros(vgg_pretrained_features[x].weight.size() + (3,))
                            weights[:, :, :, :, 1] = vgg_pretrained_features[x].weight
                        else:
                            weights = vgg_pretrained_features[x].weight.unsqueeze(4).repeat(1, 1, 1, 1, 3)#.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                        self.slice4[x - 16].weight = Parameter(weights)
        else:
            with torch.no_grad():
                for x in range(4):
                    self.slice1.add_module(str(x), vgg_pretrained_features[x])
                for x in range(4, 9):
                    self.slice2.add_module(str(x), vgg_pretrained_features[x])
                for x in range(9, 16):
                    self.slice3.add_module(str(x), vgg_pretrained_features[x])
                for x in range(16, 23):
                    self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        if x.size(1) != 3:
            if len(x.size()) == 5:
                x = x.repeat(1, 3, 1, 1, 1)
            else:
                x = x.repeat(1, 3, 1, 1)

        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        out = {
            self.layer_names[0]: relu1_2,
            self.layer_names[1]: relu2_2,
            self.layer_names[2]: relu3_3,
            self.layer_names[3]: relu4_3
        }

        return out
