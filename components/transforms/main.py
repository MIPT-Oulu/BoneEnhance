import torch
import numpy as np
from functools import partial
from tqdm import tqdm

from solt import DataContainer
import solt.transforms as slt
import solt.core as slc

from collagen.data.utils import ApplyTransform, Compose
from collagen.data import ItemLoader


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
    if len(tensor.size()) != 3:
        raise ValueError
    # Original version
    """
    for channel in range(tensor.size(0)):
        tensor[channel, :, :] -= mean[channel]
        tensor[channel, :, :] /= std[channel]

    return tensor
    """
    # Modified shape
    for channel in range(tensor.size(2)):
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
    x = x.squeeze()
    x = torch.from_numpy(x)
    if x.dim() == 2:  # CxHxW format
        x = x.unsqueeze(0)

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
    trf = conf['transforms']
    training = conf['training']
    crop_small = tuple(training['crop_small'])
    crop_large = tuple(training['crop_large'])
    prob = trf['probability']
    # Training transforms
    train_transforms = [slc.SelectiveStream([
        slc.Stream([
            slt.Projection(
                slc.Stream([
                    slt.Rotate(angle_range=tuple(trf['rotation']), p=prob),
                    slt.Scale(range_x=tuple(trf['scale']),
                                    range_y=tuple(trf['scale']), same=False, p=prob),
                    #slt.Shear(range_x=tuple(trf['shear']),
                    #                range_y=tuple(trf['shear']), p=prob),
                    #slt.Translate(range_x=trf['translation'], range_y=trf['translation'], p=prob)
                ]),
                v_range=tuple(trf['v_range'])),
            # Spatial
            slt.Flip(p=prob, axis=-1),
            #slt.Pad(pad_to=crop_size),
            #slt.Crop(crop_mode='c', crop_to=crop_size),

            # Intensity
            # Brightness/contrast
            #slc.SelectiveStream([
            #    slt.Brightness(brightness=tuple(trf['brightness']), p=prob),
            #    slt.Contrast(contrast=trf['contrast'], p=prob)]),
            # Noise
            #slc.SelectiveStream([
            #    slt.SaltAndPepper(p=prob, gain=trf['gain_sp']),
            #    slt.Noise(p=prob, gain=trf['gain_gn']),
            #    slc.SelectiveStream([
            #        slt.Blur(p=prob, blur_type='g', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma'])),
            #       slt.Blur(p=prob, blur_type='m', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma']))])])

            ]),

        # Empty stream
        slc.Stream()

        ])]

    # Stream to crop a large and small image from the center
    small_transforms = [slc.Stream([
        slt.Pad(pad_to=crop_small),
        slt.Crop(crop_mode='c', crop_to=crop_small)])]

    large_transforms = [slc.Stream([
        slt.Pad(pad_to=crop_large),
        slt.Crop(crop_mode='c', crop_to=crop_large)])]

    random_trf = [
        wrap_solt_double,
        slc.Stream(train_transforms),
        unwrap_solt
    ]

    small_trf = [
        wrap_solt_single,
        slc.Stream(small_transforms),
        unwrap_solt,
        ApplyTransform(numpy2tens, (0, 1, 2))
    ]

    large_trf = [
        wrap_solt_single,
        slc.Stream(large_transforms),
        unwrap_solt,
        ApplyTransform(numpy2tens, (0, 1, 2))
    ]

    # Validation transforms
    val_trf = [
        wrap_solt_double,
        slc.Stream(),
        unwrap_solt
    ]

    # Use normalize_channel_wise if mean and std not calculated
    if mean is not None and std is not None:
        small_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))
        large_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))

    if mean is not None and std is not None:
        val_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))

    # Compose transforms
    train_trf_cmp = [
        Compose(random_trf, return_torch=False),
        Compose(small_trf, return_torch=False),
        Compose(large_trf, return_torch=False)
    ]

    val_trf_cmp = [
        Compose(val_trf, return_torch=False),
        Compose(small_trf, return_torch=False),
        Compose(large_trf, return_torch=False)
    ]

    return {'train': train_trf_cmp, 'val': val_trf_cmp,
            'train_list': random_trf, 'val_list': val_trf}


def estimate_mean_std(config, metadata, parse_item_cb, num_threads=8, bs=16):
    mean_std_loader = ItemLoader(meta_data=metadata,
                                 transform=train_test_transforms(config)['train'],
                                 parse_item_cb=parse_item_cb,
                                 batch_size=bs, num_workers=num_threads,
                                 shuffle=False)

    mean = None
    std = None
    for i in tqdm(range(len(mean_std_loader)), desc='Calculating mean and standard deviation'):
        for batch in mean_std_loader.sample():
            if mean is None:
                mean = torch.zeros(batch['data'].size(1))
                std = torch.zeros(batch['data'].size(1))
            # for channel in range(batch['data'].size(1)):
            #     mean[channel] += batch['data'][:, channel, :, :].mean().item()
            #     std[channel] += batch['data'][:, channel, :, :].std().item()
            mean += batch['data'].mean().item()
            std += batch['data'].std().item()

    mean /= len(mean_std_loader)
    std /= len(mean_std_loader)

    return mean, std
