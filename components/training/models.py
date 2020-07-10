import numpy as np
import torch.nn as nn
import torch
import math
from torch.nn import init


# 2D Conv
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, padding=0,
                     bias=False)


def conv2x2(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=2, stride=stride, padding=0,
                     bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv4x4(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=4, stride=stride, padding=1,
                     bias=False)


# 3D Conv
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=1, stride=stride, padding=0,
                     bias=False)


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv4x4x4(in_planes, out_planes, stride=2):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=4, stride=stride, padding=1,
                     bias=False)


# 2D Deconv
def deconv1x1(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=1, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv2x2(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=2, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=0,
                              bias=False)


def deconv4x4(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=4, stride=stride, padding=1, output_padding=0,
                              bias=False)


# 3D Deconv
def deconv1x1x1(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=1, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv3x3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=0,
                              bias=False)


def deconv4x4x4(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=4, stride=stride, padding=1, output_padding=0,
                              bias=False)


def _make_layers(in_channels, output_channels, layer_type, bn='', activation=None):
    layers = []

    # Layer type
    if layer_type == 'conv1_s1':
        layers.append(conv1x1(in_channels, output_channels, stride=1))
    elif layer_type == 'conv2_s2':
        layers.append(conv2x2(in_channels, output_channels, stride=2))
    elif layer_type == 'conv3_s1':
        layers.append(conv3x3(in_channels, output_channels, stride=1))
    elif layer_type == 'conv4_s2':
        layers.append(conv4x4(in_channels, output_channels, stride=2))
    elif layer_type == 'deconv1_s1':
        layers.append(deconv1x1(in_channels, output_channels, stride=1))
    elif layer_type == 'deconv2_s2':
        layers.append(deconv2x2(in_channels, output_channels, stride=2))
    elif layer_type == 'deconv3_s1':
        layers.append(deconv3x3(in_channels, output_channels, stride=1))
    elif layer_type == 'deconv4_s2':
        layers.append(deconv4x4(in_channels, output_channels, stride=2))
    elif layer_type == 'conv1x1_s1':
        layers.append(conv1x1x1(in_channels, output_channels, stride=1))
    elif layer_type == 'deconv1x1_s1':
        layers.append(deconv1x1x1(in_channels, output_channels, stride=1))
    elif layer_type == 'deconv3x3_s1':
        layers.append(deconv3x3x3(in_channels, output_channels, stride=1))
    elif layer_type == 'deconv4x4_s2':
        layers.append(deconv4x4x4(in_channels, output_channels, stride=2))
    else:
        raise NotImplementedError('layer type [{}] is not implemented'.format(layer_type))

    # Batch normalization
    if bn == '2d':
        layers.append(nn.BatchNorm2d(output_channels))
    elif bn == '3d':
        layers.append(nn.BatchNorm3d(output_channels))

    # Activation
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'sigm':
        layers.append(nn.Sigmoid())
    elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU(0.2, True))
    else:
        if activation is not None:
            raise NotImplementedError('activation function [{}] is not implemented'.format(activation))

    return nn.Sequential(*layers)


def _init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                print('Initializing Weights: {}...'.format(classname))
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Sequential') == -1 and classname.find('Conv5_Deconv5_Local') == -1:
            raise NotImplementedError('initialization of [{}] is not implemented'.format(classname))

    print('initialize network with {}'.format(init_type))
    net.apply(init_func)


def _initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class EnhanceNet(nn.Module):
    """Inspired by ReconNet https://doi.org/10.1038/s41551-019-0466-4"""


    def __init__(self, input_shape, magnification, gain=0.02, init_type='standard'):
        """

        :param input_shape: Size of the input image
        :param magnification: The factor for upscaling output image
        :param gain: Gain used in weight initialization. Not used in standard init.
        :param init_type: Option how to initialize model weights
        """
        super(EnhanceNet, self).__init__()

        # Variables
        output_shape = np.array(input_shape) * magnification
        # Feature map sizes
        f = [3, 128, 256, 512, 1024]
        self.__f = f

        ######### representation network - convolution layers
        self.conv_layer1 = _make_layers(3, f[0], 'conv3_s1')  # RGB input
        self.conv_layer2 = _make_layers(f[0], f[0], 'conv3_s1', bn='2d')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_layer3 = _make_layers(f[0], f[1], 'conv3_s1', bn='2d', activation='relu')
        self.conv_layer4 = _make_layers(f[1], f[1], 'conv3_s1', bn='2d')
        self.relu4 = nn.ReLU(inplace=True)
        self.conv_layer5 = _make_layers(f[1], f[2], 'conv3_s1', bn='2d', activation='relu')
        self.conv_layer6 = _make_layers(f[2], f[2], 'conv3_s1', bn='2d')
        self.relu6 = nn.ReLU(inplace=True)
        self.conv_layer7 = _make_layers(f[2], f[3], 'conv3_s1', bn='2d', activation='relu')
        self.conv_layer8 = _make_layers(f[3], f[3], 'conv3_s1', bn='2d')
        self.relu8 = nn.ReLU(inplace=True)
        self.conv_layer9 = _make_layers(f[3], f[4], 'conv3_s1', bn='2d', activation='relu')
        self.conv_layer10 = _make_layers(f[4], f[4], 'conv3_s1', bn='2d')
        self.relu10 = nn.ReLU(inplace=True)

        ######### transform module
        self.trans_layer1 = _make_layers(f[4], f[4], 'conv1_s1', 'relu')
        self.trans_layer2 = _make_layers(f[4], f[4], 'deconv1x1_s1', 'relu')

        ######### generation network - deconvolution layers
        self.deconv_layer10 = _make_layers(f[4], f[3], 'deconv4x4_s2', bn='3d', activation='relu')
        self.deconv_layer8 = _make_layers(f[3], f[3], 'deconv3x3_s1', bn='3d', activation='relu')
        self.deconv_layer7 = _make_layers(f[3], f[3], 'deconv3x3_s1', bn='3d', activation='relu')
        self.deconv_layer6 = _make_layers(f[3], f[2], 'deconv4x4_s2', bn='3d', activation='relu')
        self.deconv_layer5 = _make_layers(f[2], f[2], 'deconv3x3_s1', bn='3d', activation='relu')
        self.deconv_layer4 = _make_layers(f[3], f[1], 'deconv4x4_s2', bn='3d', activation='relu')
        self.deconv_layer3 = _make_layers(f[1], f[1], 'deconv3x3_s1', bn='3d', activation='relu')
        self.deconv_layer2 = _make_layers(f[1], 1, 'deconv4x4_s2', bn='3d', activation='relu')
        self.deconv_layer1 = _make_layers(1, 1, 'deconv3x3_s1', bn='3d', activation='relu')
        self.deconv_layer0 = _make_layers(1, 1, 'conv1x1_s1', activation='relu')
        self.output_layer = _make_layers(input_shape[0], f[0], 'conv1_s1')

        if init_type == 'standard':
            _initialize_weights(self)
        else:
            _init_weights(self, gain=gain, init_type=init_type)

    def forward(self, x):
        ### representation network
        x = self.conv_layer1(x)
        x2 = self.conv_layer2(x)
        x = self.relu2(x + x2)
        x = self.conv_layer3(x)
        x2 = self.conv_layer4(x)
        x = self.relu4(x + x2)
        x = self.conv_layer5(x)
        x2 = self.conv_layer6(x)
        x = self.relu6(x + x2)
        x = self.conv_layer7(x)
        x2 = self.conv_layer8(x)
        x = self.relu8(x + x2)
        x = self.conv_layer9(x)
        x2 = self.conv_layer10(x)
        x = self.relu10(x + x2)

        ### transform module
        x = self.trans_layer1(x)
        x = x.view(-1, self.__f[4], 4, 16, 16)
        x = self.trans_layer2(x)

        ### generation network
        x = self.deconv_layer10(x)
        x = self.deconv_layer8(x)
        x = self.deconv_layer7(x)
        #x = self.deconv_layer6(x)
        #x = self.deconv_layer5(x)
        x = self.deconv_layer4(x)
        x = self.deconv_layer3(x)
        x = self.deconv_layer2(x)
        x = self.deconv_layer1(x)

        ### output
        out = self.deconv_layer0(x)
        out = torch.squeeze(out, 1)
        out = self.output_layer(out)

        return out
