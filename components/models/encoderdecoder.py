import torch
from collagen import Module
from collagen.modelzoo.modules import ConvBlock
from collagen.modelzoo.segmentation import constants, backbones
from collagen.modelzoo.segmentation.decoders._fpn import FPNBlock, SegmentationBlock
from torch import nn
from torch.nn import functional as F


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