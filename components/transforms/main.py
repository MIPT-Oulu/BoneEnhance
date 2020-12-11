import torch
import numpy as np
from functools import partial

from solt import DataContainer
import solt.transforms as slt
import solt.core as slc
from BoneEnhance.components.transforms.custom_transforms import Crop, Pad, Brightness, Contrast, Blur, Flip, Rotate90, \
    Noise
from BoneEnhance.components.transforms.spatial_transforms import Rotate, Translate


from collagen.data.utils import ApplyTransform, Compose


def normalize_channel_wise(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalizes given tensor channel-wise
    Parameters
    ----------
    tensor: torch.Tensor
        Tensor to be normalized
    mean: torch.tensor
        Mean to be subtracted
    std: torch.Tensor
        Std to be divided by
    Returns
    -------
    result: torch.Tensor
    """

    # 3D
    if len(tensor.size()) == 4:
        # Modified shape
        for channel in range(tensor.size(0)):
            tensor[:, :, :, channel] -= mean[channel]
            tensor[:, :, :, channel] /= std[channel]

        return tensor
    # Noncompatible
    elif len(tensor.size()) != 3:
        raise ValueError
    # 2D
    else:
        # Modified shape
        for channel in range(tensor.size(0)):
            tensor[:, :, channel] -= mean[channel]
            tensor[:, :, channel] /= std[channel]

        return tensor


def numpy2tens(x: np.ndarray, dtype='f') -> torch.Tensor:
    """Converts a numpy array into torch.Tensor
    Parameters
    ----------
    x: np.ndarray
        Array to be converted
    dtype: str
        Target data type of the tensor. Can be f - float and l - long
    Returns
    -------
    result: torch.Tensor
    """
    #x = x.squeeze()

    # CxHxW format
    if len(x.shape) == 2:
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
    else:
        x = np.rollaxis(x, -1)
        x = torch.from_numpy(x)

    if dtype == 'f':
        return x.float()
    elif dtype == 'l':
        return x.long()
    else:
        raise NotImplementedError


def wrap_solt_double(entry):
    return DataContainer(entry, 'II', allow_inconsistency=True, transform_settings={0: {'interpolation': 'bilinear'},
                                                                                    1: {'interpolation': 'bilinear'}})


def wrap_solt_single(entry):
    return DataContainer(entry, 'I', allow_inconsistency=False, transform_settings={0: {'interpolation': 'bilinear'}})


def unwrap_solt(dc):
    return dc.data


def train_test_transforms(conf, mean=None, std=None):
    trf = conf.transforms
    training = conf.training
    crop_small = tuple(training.crop_small)
    crop_large = tuple([crop * training.magnification for crop in crop_small])
    prob = trf.probability
    # Training transforms
    train_transforms = [slc.SelectiveStream([
        slc.Stream([
            # slt.Projection(
            #    slc.Stream([
            #        slt.Rotate(angle_range=tuple(trf['rotation']), p=prob),
            #        slt.Scale(range_x=tuple(trf['scale']),
            #                        range_y=tuple(trf['scale']), same=False, p=prob),
            # slt.Shear(range_x=tuple(trf['shear']),
            #                range_y=tuple(trf['shear']), p=prob),
            # slt.Translate(range_x=trf['translation'], range_y=trf['translation'], p=prob)
            #    ]),
            #    v_range=tuple(trf['v_range'])),

            # Spatial
            Rotate(angle_range=tuple(trf['rotation']), p=prob),
            Translate(range_x=trf['translation'], range_y=trf['translation'], range_z=trf['translation'], p=prob),
            Flip(axis=-1, p=prob),
            slc.SelectiveStream([Rotate90(k=1, p=prob), Rotate90(k=-1, p=prob), Rotate90(k=2, p=prob)]),

            # Make sure the batch is the correct size
            Crop(training.magnification, crop_mode='r', crop_to=(crop_small, crop_large)),
            Pad(pad_to=(crop_small, crop_large)),

            # 50% Chance for Brightness & contrast adjustment
            slc.Stream([
                Brightness(brightness_range=tuple(trf.brightness), p=prob),
                Contrast(contrast_range=trf.contrast, p=prob)]),

            # 50% Chance for smoothing/blurring
            slc.SelectiveStream([
                Blur(p=prob, blur_type='g', k_size=3, gaussian_sigma=tuple(trf.sigma)),
                Blur(p=prob, blur_type='m', k_size=3, gaussian_sigma=tuple(trf.sigma))
                ]),

            # 50% Chance for Added noise
            slc.SelectiveStream([
                Noise(p=prob, mode='gaussian', gain_range=trf['gain_gn']),
                Noise(p=prob, mode='poisson', gain_range=trf['gain_gn']),
                Noise(p=prob, mode='s&p', gain_range=trf['gain_sp']),
                Noise(p=prob, mode='speckle', gain_range=trf['gain_sp']),
            ])
        ]),

        # Empty stream
        slc.Stream([
            Crop(training.magnification, crop_mode='r', crop_to=(crop_small, crop_large)),
            Pad(pad_to=(crop_small, crop_large)),
        ])

    ])]

    # 2D or 3D?
    if len(crop_small) == 3:
        axis = (0, 1, 2, 3)
    else:
        axis = (0, 1, 2)

    # Training transforms
    random_trf = [
        wrap_solt_double,
        slc.Stream(train_transforms),
        unwrap_solt,
        ApplyTransform(numpy2tens, axis)
    ]

    # Validation transforms
    val_trf = [
        wrap_solt_double,
        slc.Stream([
            Pad(pad_to=(crop_small, crop_large)),
            Crop(training['magnification'], crop_mode='r', crop_to=(crop_small, crop_large))
        ]),
        unwrap_solt,
        ApplyTransform(numpy2tens, axis)
    ]

    # Use normalize_channel_wise if mean and std are calculated (training and evaluation)
    if mean is not None and std is not None:
        random_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))
        val_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))

    # Compose transforms
    train_trf_cmp = Compose(random_trf, return_torch=False)

    val_trf_cmp = Compose(val_trf, return_torch=False)

    return {'train': train_trf_cmp, 'eval': val_trf_cmp,
            'train_list': random_trf, 'eval_list': val_trf}


