from torch import nn


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
