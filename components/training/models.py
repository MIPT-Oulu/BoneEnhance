import torch
import torch.nn.functional as F

from collagen.core import Module
from collagen.modelzoo.segmentation.decoders._fpn import FPNBlock, SegmentationBlock
from collagen.modelzoo.modules import ConvBlock
from collagen.modelzoo.segmentation import backbones, constants, decoders

from BoneEnhance.components.training.model_blocks import *
from BoneEnhance.components.training.model_initialization import *


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

    def __init__(self, input_shape, magnification, gain=0.02, init_type='standard'):
        """

        :param input_shape: Size of the input image
        :param magnification: The factor for upscaling output image
        :param gain: Gain used in weight initialization. Not used in standard init.
        :param init_type: Option how to initialize model weights
        """
        super(EnhanceNet, self).__init__()

        # Feature map sizes
        f = [3, 128, 256, 512, 1024]
        self.__f = f

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

        # Generation network - deconvolution layers
        self.deconv_layer10 = _make_layers(f[4], f[3], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer8 = _make_layers(f[3], f[3], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer7 = _make_layers(f[3], f[3], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer6 = _make_layers(f[3], f[2], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer5 = _make_layers(f[2], f[2], 'deconv4_s2', bn='2d', activation='relu')
        self.deconv_layer4 = _make_layers(f[2], f[1], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer3 = _make_layers(f[1], f[1], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer2 = _make_layers(f[1], f[0], 'deconv3_s1', bn='2d', activation='relu')
        self.deconv_layer1 = _make_layers(f[0], f[0], 'deconv4_s2', bn='2d', activation='relu')
        self.deconv_layer0 = _make_layers(f[0], f[0], 'deconv3_s1', activation='relu')
        self.output_layer = _make_layers(f[0], f[0], 'conv1_s1')

        if init_type == 'standard':
            initialize_weights(self)
        else:
            init_weights(self, gain=gain, init_type=init_type)

    def forward(self, x):
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
        x = self.trans_layer1(x)
        # x = x.view(-1, self.__f[4], 16, 16)
        x = self.trans_layer2(x)
        x = self.relu(x + x2)

        # Generation network
        x = self.deconv_layer10(x)
        x = self.deconv_layer8(x)
        x = self.deconv_layer7(x)
        x = self.deconv_layer6(x)
        x = self.deconv_layer5(x)
        x = self.deconv_layer4(x)
        x = self.deconv_layer3(x)
        x = self.deconv_layer2(x)
        x = self.deconv_layer1(x)

        # Output
        out = self.deconv_layer0(x)
        # out = torch.squeeze(out, 1)
        out = self.output_layer(out)

        return out


class EncoderDecoder(Module):
    def __init__(self, n_outputs, backbone: str or Module, decoder: str or Module,
                 decoder_normalization='BN', spatial_dropout=None, bayesian_dropout=None,
                 magnification=4):
        super(EncoderDecoder, self).__init__()
        if isinstance(backbone, str):
            if backbone in constants.allowed_encoders:
                if 'resnet' in backbone:
                    backbone = backbones.ResNetBackbone(backbone, dropout=bayesian_dropout)
                else:
                    ValueError('Cannot find the implementation of the backbone!')
            else:
                raise ValueError('This backbone name is not in the list of allowed backbones!')

        if isinstance(decoder, str):
            if decoder == 'enhance':
                decoder = FPNDecoder(encoder_channels=backbone.output_shapes,
                                     pyramid_channels=256, segmentation_channels=128,
                                     final_channels=n_outputs, spatial_dropout=spatial_dropout,
                                     normalization=decoder_normalization,
                                     bayesian_dropout=bayesian_dropout)

        decoder.initialize()

        self.backbone = backbone
        self.decoder = decoder
        self.magnification = magnification

    def forward(self, x):
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        features = self.backbone(x)
        return self.decoder(features)

    def switch_dropout(self):
        """
        Has effect only if the model supports monte-carlo dropout inference.

        """
        self.backbone.switch_dropout()
        self.decoder.switch_dropout()

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass


class FPNDecoder(Module):
    """
    Extended implementation from https://github.com/qubvel/segmentation_models.pytorch
    """

    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
            segmentation_channels=128,
            final_channels=1,
            spatial_dropout=0.2,
            normalization='BN',
            bayesian_dropout=None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1], dropout=bayesian_dropout)
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2], dropout=bayesian_dropout)
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3], dropout=bayesian_dropout)

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3,
                                    normalization=normalization)

        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2,
                                    normalization=normalization)

        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1,
                                    normalization=normalization)

        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0,
                                    normalization=normalization)

        self.assembly = ConvBlock(ks=3, inp=segmentation_channels * 4,
                                  out=segmentation_channels, stride=1, pad=1,
                                  activation='relu', normalization=normalization)

        self.spatial_dropout = spatial_dropout

        self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)

        self.initialize()

    def switch_dropout(self):
        self.p4.switch_dropout()
        self.p3.switch_dropout()
        self.p2.switch_dropout()

    def forward(self, x, out_shape=None):
        _, c2, c3, c4, c5 = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        x = self.assembly(torch.cat((s5, s4, s3, s2), 1))

        if self.spatial_dropout is not None:
            x = F.dropout2d(x, self.spatial_dropout, training=self.training)

        x = self.final_conv(x)

        x = x.repeat(1, 3, 1, 1)

        if out_shape is None:
            return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            return F.interpolate(x, size=out_shape, mode='bilinear', align_corners=True)

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass
