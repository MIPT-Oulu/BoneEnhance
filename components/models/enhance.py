from BoneEnhance.components.models.model_blocks import *
from BoneEnhance.components.models.model_initialization import *
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
        Originally introduced in (Microsoft Research Asia, He et al.): https://arxiv.org/abs/1512.03385
        Based on implementation from https://github.com/gordicaleksa/pytorch-nst-feedforward
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        stride_size = 1
        self.conv1 = conv3x3(channels, channels, stride=stride_size)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = conv3x3(channels, channels, stride=stride_size)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = self.relu(out + residual)
        return out


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


class EnhanceNet(nn.Module):
    """Inspired by ReconNet https://doi.org/10.1038/s41551-019-0466-4"""

    def __init__(self, input_shape, magnification, gain=0.02, init_type='standard', residual=False, upscale_input=False):
        """

        :param input_shape: Size of the input image
        :param magnification: The factor for upscaling output image
        :param gain: Gain used in weight initialization. Not used in standard init.
        :param init_type: Option how to initialize model weights
        """
        super(EnhanceNet, self).__init__()

        # Feature map sizes
        f = [3, 128, 256, 512, 1024]
        self.__magnification = magnification
        self.residual = residual
        self.upscale_input = upscale_input
        self.upscale_factor = magnification

        # Representation network - convolution layers
        self.conv_layer1 = _make_layers(3, f[0], 'conv3_s1')  # RGB input
        self.conv_layer2 = _make_layers(f[0], f[0], 'conv3_s1', bn='2d')
        self.conv_layer3 = _make_layers(f[0], f[1], 'conv3_s1', bn='2d', activation='relu')
        self.conv_layer4 = _make_layers(f[1], f[1], 'conv3_s1', bn='2d')
        self.conv_layer5 = _make_layers(f[1], f[2], 'conv3_s1', bn='2d', activation='relu')
        self.conv_layer6 = _make_layers(f[2], f[2], 'conv3_s1', bn='2d')
        self.conv_layer7 = _make_layers(f[2], f[3], 'conv3_s1', bn='2d', activation='relu')
        self.conv_layer8 = _make_layers(f[3], f[3], 'conv3_s1', bn='2d')
        self.conv_layer9 = _make_layers(f[3], f[4], 'conv3_s1', bn='2d', activation='relu')
        self.conv_layer10 = _make_layers(f[4], f[4], 'conv3_s1', bn='2d')

        # Rectified linear unit
        self.relu = nn.ReLU(inplace=True)

        # Transform module
        self.trans_layer1 = _make_layers(f[4], f[4], 'conv1_s1', 'relu')
        self.trans_layer2 = _make_layers(f[4], f[4], 'deconv1_s1', 'relu')

        # Residual blocks
        self.res1 = ResidualBlock(f[4])
        self.res2 = ResidualBlock(f[4])
        self.res3 = ResidualBlock(f[4])
        self.res4 = ResidualBlock(f[4])

        # Generation network - deconvolution layers
        self.deconv_layer8 = _make_layers(f[4], f[3], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer7 = _make_layers(f[3], f[3], 'deconv3_s1', bn='2d', activation='relu')
        self.upscale_layer1 = _make_layers(f[3], f[3], 'deconv4_s2', bn='2d', activation='relu')
        self.deconv_layer6 = _make_layers(f[3], f[3], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer5 = _make_layers(f[3], f[2], 'deconv3_s1', bn='2d', activation='relu')
        self.upscale_layer2 = _make_layers(f[2], f[2], 'deconv4_s2', bn='2d', activation='relu')
        self.deconv_layer4 = _make_layers(f[2], f[1], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer3 = _make_layers(f[1], f[1], 'deconv3_s1', bn='2d', activation='relu')
        self.upscale_layer3 = _make_layers(f[1], f[1], 'deconv4_s2', bn='2d', activation='relu')
        self.deconv_layer2 = _make_layers(f[1], f[0], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer1 = _make_layers(f[0], f[0], 'deconv3_s1', activation='relu')
        self.output_layer = _make_layers(f[0], 1, 'conv1_s1')

        if init_type == 'standard':
            initialize_weights(self)
        else:
            init_weights(self, gain=gain, init_type=init_type)

    def forward(self, x):

        if self.upscale_input:
            x = F.interpolate(x, scale_factor=self.upscale_factor)
            self.__magnification = 1

        # Representation network
        x = self.conv_layer1(x)
        x2 = self.conv_layer2(x)
        x = self.relu(x + x2)
        x = self.conv_layer3(x)
        x2 = self.conv_layer4(x)
        x = self.relu(x + x2)
        x = self.conv_layer5(x)
        x2 = self.conv_layer6(x)
        x = self.relu(x + x2)
        x = self.conv_layer7(x)
        x2 = self.conv_layer8(x)
        x = self.relu(x + x2)
        x = self.conv_layer9(x)
        x2 = self.conv_layer10(x)
        x = self.relu(x + x2)

        # Transform module
        if self.residual:
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
        else:
            x = self.trans_layer1(x)
            x = self.trans_layer2(x)
            #x = self.relu(x + x2)

        # Generation network
        x = self.deconv_layer8(x)
        x = self.deconv_layer7(x)
        # Upscale 2x
        if self.__magnification > 1:
            x = self.upscale_layer1(x)
        x = self.deconv_layer6(x)
        #x = self.relu(x + x2)
        x = self.deconv_layer5(x)
        # Upscale 2x
        if self.__magnification == 4:
            x = self.upscale_layer2(x)
        x = self.deconv_layer4(x)
        x = self.deconv_layer3(x)
        # Upscale 2x
        if self.__magnification == 8:
            x = self.upscale_layer2(x)  # Upscale 2x
            x = self.upscale_layer3(x)
        x = self.deconv_layer2(x)

        # Output
        x = self.deconv_layer1(x)
        x = self.output_layer(x)

        # Duplicate 1-channel image to represent RGB
        x = x.repeat(1, 3, 1, 1)
        return x
