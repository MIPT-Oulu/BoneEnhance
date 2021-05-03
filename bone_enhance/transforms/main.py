import torch
import numpy as np
from functools import partial

from solt import DataContainer
import solt.transforms as slt
import solt.core as slc
from bone_enhance.transforms.custom_transforms import Crop, Pad, Brightness, Contrast, Blur, Flip, Rotate90, \
    Noise
from bone_enhance.transforms.random_augmentations import return_transforms

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

    # Check that channel dimension is first
    if tensor.size(0) not in [1, 3]:
        raise Exception('Tensor in incorrect format!')

    # 3D
    if len(tensor.size()) == 4:
        # Modified shape
        for channel in range(tensor.size(0)):
            tensor[channel, :, :, :] -= mean[channel]
            tensor[channel, :, :, :] /= std[channel]

        return tensor
    # Noncompatible
    elif len(tensor.size()) != 3:
        raise ValueError
    # 2D
    else:
        # Modified shape
        for channel in range(tensor.size(0)):
            tensor[channel, :, :] -= mean[channel]
            tensor[channel, :, :] /= std[channel]

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


def wrap_solt_segmentation(entry):
    return DataContainer(entry, 'IM', allow_inconsistency=False, transform_settings={0: {'interpolation': 'bilinear'},
                                                                                     1: {'interpolation': 'nearest'}})

def wrap_solt_single(entry):
    return DataContainer(entry, 'I', allow_inconsistency=False, transform_settings={0: {'interpolation': 'bilinear'}})


def unwrap_solt(dc):
    return dc.data


def train_test_transforms(conf, args, mean=None, std=None):
    trf = conf.transforms
    training = conf.training
    crop_small = tuple(training.crop_small)
    crop_large = tuple([crop * training.magnification for crop in crop_small])
    prob = trf.probability
    vol = len(crop_small) == 3

    # Training transforms
    train_transforms, val_transforms = return_transforms(prob, trf, training.magnification, crop_small, conf, vol)

    # 2D or 3D?
    if vol:
        axis = (0, 1, 2, 3)
    else:
        axis = (0, 1, 2)

    # SR or segmentation?
    if conf.training.segmentation:
        wrap = wrap_solt_segmentation
    else:
        wrap = wrap_solt_double

    # Training transforms
    random_trf = [
        wrap,
        slc.Stream(train_transforms),
        unwrap_solt,
        ApplyTransform(numpy2tens, axis)
    ]

    # Validation transforms
    val_trf = [
        wrap,
        val_transforms,
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


