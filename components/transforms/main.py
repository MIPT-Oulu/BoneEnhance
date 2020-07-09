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


def wrap_solt(entry):
    return DataContainer(entry, 'II', transform_settings={0: {'interpolation': 'bilinear'},
                                                          1: {'interpolation': 'bilinear'}})


def unwrap_solt(dc):
    #return dc['images']
    return dc.data


def train_test_transforms(conf, mean=None, std=None):
    trf = conf['training']
    crop_size = trf['crop_size']
    prob = trf['transform_probability']
    # Training transforms
    if trf['experiment'] == '3D':
        train_transforms = [slc.SelectiveStream([
            slc.Stream([
                #slt.Projection(
                #    slc.Stream([
                #        slt.Rotate(angle_range=tuple(trf['rotation_range']), p=prob),
                #        slt.Scale(range_x=tuple(trf['scale_range']),
                #                        range_y=tuple(trf['scale_range']), same=False, p=prob),
                        #slt.Shear(range_x=tuple(trf['shear_range']),
                        #                range_y=tuple(trf['shear_range']), p=prob),
                        #slt.Translate(range_x=trf['translation_range'], range_y=trf['translation_range'], p=prob)
                #    ]),
                #    v_range=tuple(trf['v_range'])),
                # Spatial
                slt.Flip(p=prob, axis=-1),
                #slt.Pad(pad_to=crop_size),
                slt.Crop(crop_mode='c', crop_to=crop_size),

                # Intensity
                # Brightness/contrast
                #slc.SelectiveStream([
                #    slt.Brightness(brightness_range=tuple(trf['brightness_range']), p=prob),
                #    slt.Contrast(contrast_range=trf['contrast_range'], p=prob)]),
                # Noise
                #slc.SelectiveStream([
                #    slt.SaltAndPepper(p=prob, gain_range=trf['gain_range_sp']),
                #    slt.Noise(p=prob, gain_range=trf['gain_range_gn']),
                #    slc.SelectiveStream([
                #        slt.Blur(p=prob, blur_type='g', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma'])),
                #       slt.Blur(p=prob, blur_type='m', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma']))])])

                ]),

            # Empty stream
            slc.Stream()
                #[slt.Pad(pad_to=crop_size),
                #slt.Crop(crop_mode='r', crop_to=crop_size)])
            ])

        ]
    """
    else:
        train_transforms = [slc.SelectiveStream([
            slc.Stream([
                # Projection
                slt.Projection(
                    slc.Stream([
                        slt.Rotate(angle_range=tuple(trf['rotation_range']), p=prob),
                        slt.Scale(range_x=tuple(trf['scale_range']),
                                        range_y=tuple(trf['scale_range']), same=False, p=prob),
                        #slt.RandomShear(range_x=tuple(trf['shear_range']),
                        #                range_y=tuple(trf['shear_range']), p=prob),
                        #slt.RandomTranslate(range_x=trf['translation_range'], range_y=trf['translation_range'], p=prob)
                    ]),
                    v_range=tuple(trf['v_range'])),
                # Spatial
                slt.Flip(p=prob),
                slt.Pad(pad_to=crop_size),
                slt.Crop(crop_mode='r', crop_to=crop_size),
                # Intensity
                slc.SelectiveStream([
                    slt.GammaCorrection(gamma_range=tuple(trf['gamma_range']), p=prob),
                    slt.HSV(h_range=tuple(trf['hsv_range']),
                                       s_range=tuple(trf['hsv_range']),
                                       v_range=tuple(trf['hsv_range']), p=prob)]),
                slc.SelectiveStream([
                    slt.Brightness(brightness_range=tuple(trf['brightness_range']), p=prob),
                    slt.Contrast(contrast_range=trf['contrast_range'], p=prob)]),
                slc.SelectiveStream([
                    slt.SaltAndPepper(p=prob, gain_range=trf['gain_range_sp']),
                    slt.Noise(p=prob, gain_range=trf['gain_range_gn']),
                    slc.SelectiveStream([
                        slt.Blur(p=prob, blur_type='g', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma'])),
                        slt.Blur(p=prob, blur_type='m', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma']))])])]),

            # Empty stream
            slc.Stream([
                slt.Pad(pad_to=crop_size),
                slt.Crop(crop_mode='r', crop_to=crop_size)])])
        ]
    """

    train_trf = [
        wrap_solt,
        slc.Stream(train_transforms),
        unwrap_solt,
        ApplyTransform(numpy2tens, (0, 1, 2))
    ]
    # Validation transforms
    val_trf = [
        wrap_solt,
        slc.Stream([
            slt.Pad(pad_to=crop_size[1]),
            slt.Crop(crop_mode='r', crop_to=crop_size)
        ]),
        unwrap_solt,
        ApplyTransform(numpy2tens, idx=(0, 1, 2))
    ]

    # Use normalize_channel_wise if mean and std not calculated
    if mean is not None and std is not None:
        train_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))

    if mean is not None and std is not None:
        val_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))

    # Compose transforms
    train_trf_cmp = Compose(train_trf)
    val_trf_cmp = Compose(val_trf)

    return {'train': train_trf_cmp, 'val': val_trf_cmp,
            'train_list': train_trf, 'val_list': val_trf}


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
