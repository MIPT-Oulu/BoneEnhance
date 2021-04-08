from solt import DataContainer
import solt.transforms as slt
import solt.core as slc
from BoneEnhance.components.transforms.custom_transforms import Crop, Pad, Brightness, Contrast, Blur, Flip, Rotate90, \
    Noise
from BoneEnhance.components.transforms.spatial_transforms import Rotate, Translate


from collagen.data.utils import ApplyTransform, Compose


def return_transforms(prob, trf, magnification, crop_small, config, vol=False):
    crop_large = tuple([crop * magnification for crop in crop_small])

    if vol:
        transforms = slc.SelectiveStream([
            slc.Stream([

                # Spatial
                Rotate(angle_range=tuple(trf['rotation']), p=prob, vol=True),
                Translate(range_x=trf['translation'], range_y=trf['translation'], range_z=trf['translation'], p=prob),
                Flip(axis=-1, p=prob),
                slc.SelectiveStream([Rotate90(k=1, p=prob), Rotate90(k=-1, p=prob), Rotate90(k=2, p=prob)]),

                # Make sure the batch is the correct size
                Crop(magnification, crop_mode='r', crop_to=(crop_small, crop_large)),
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
                #slc.SelectiveStream([
                #    Noise(p=prob, mode='gaussian', gain_range=trf['gain_gn']),
                #    Noise(p=prob, mode='poisson', gain_range=trf['gain_gn']),
                #    Noise(p=prob, mode='s&p', gain_range=trf['gain_sp']),
                #    Noise(p=prob, mode='speckle', gain_range=trf['gain_sp']),
                #])
            ]),

            # Empty stream
            slc.Stream([
                Crop(magnification, crop_mode='r', crop_to=(crop_small, crop_large)),
                Pad(pad_to=(crop_small, crop_large)),
            ])
        ])

        val_transfroms = slc.Stream([
            Pad(pad_to=(crop_small, crop_large)),
            Crop(magnification, crop_mode='r', crop_to=(crop_small, crop_large))
        ])

    elif config.training.segmentation:
        transforms = slc.SelectiveStream([
            slc.Stream([

                # Spatial
                slt.Rotate(angle_range=tuple(trf['rotation']), p=prob),
                slt.Translate(range_x=trf['translation'], range_y=trf['translation'], p=prob),
                slt.Flip(axis=-1, p=prob),
                slc.SelectiveStream([slt.Rotate90(k=1, p=prob), slt.Rotate90(k=-1, p=prob), slt.Rotate90(k=2, p=prob)]),

                # Make sure the batch is the correct size
                slt.Pad(pad_to=crop_large),
                slt.Crop(crop_mode='r', crop_to=crop_large),

                # 50% Chance for Brightness & contrast adjustment
                slc.SelectiveStream([
                    slt.Brightness(brightness_range=tuple(trf['brightness']), p=prob),
                    slt.Contrast(contrast_range=trf['contrast'], p=prob)]),
                # Noise
                slc.SelectiveStream([
                    # slt.SaltAndPepper(p=prob, gain_range=trf['gain_sp']),
                    # slt.Noise(p=prob, gain_range=trf['gain_gn']),
                    slc.SelectiveStream([
                        slt.Blur(p=prob, blur_type='g', k_size=(3, 5), gaussian_sigma=tuple(trf['sigma'])),
                        slt.Blur(p=prob, blur_type='m', k_size=(3, 5), gaussian_sigma=tuple(trf['sigma']))])
                ])
            ]),

            # Empty stream
            slc.Stream([
                slt.Pad(pad_to=crop_large),
                slt.Crop(crop_mode='r', crop_to=crop_large),
            ])
        ])

        val_transfroms = slc.Stream([
            slt.Pad(pad_to=crop_large),
            slt.Crop(crop_mode='r', crop_to=crop_large),
        ])

    else:
        transforms = slc.SelectiveStream([
            slc.Stream([

                # Spatial
                Rotate(angle_range=tuple(trf['rotation']), p=prob, vol=False),
                Translate(range_x=trf['translation'], range_y=trf['translation'], p=prob),
                slt.Flip(axis=-1, p=prob),
                slc.SelectiveStream([slt.Rotate90(k=1, p=prob), slt.Rotate90(k=-1, p=prob), slt.Rotate90(k=2, p=prob)]),

                # Make sure the batch is the correct size
                Crop(magnification, crop_mode='r', crop_to=(crop_small, crop_large)),
                Pad(pad_to=(crop_small, crop_large)),

                # 50% Chance for Brightness & contrast adjustment
                slc.SelectiveStream([
                    slt.Brightness(brightness_range=tuple(trf['brightness']), p=prob),
                    slt.Contrast(contrast_range=trf['contrast'], p=prob)]),
                # Noise
                slc.SelectiveStream([
                    # slt.SaltAndPepper(p=prob, gain_range=trf['gain_sp']),
                    # slt.Noise(p=prob, gain_range=trf['gain_gn']),
                    slc.SelectiveStream([
                        slt.Blur(p=prob, blur_type='g', k_size=(3, 5), gaussian_sigma=tuple(trf['sigma'])),
                        slt.Blur(p=prob, blur_type='m', k_size=(3, 5), gaussian_sigma=tuple(trf['sigma']))])
                ])
            ]),

            # Empty stream
            slc.Stream([
                Crop(magnification, crop_mode='r', crop_to=(crop_small, crop_large)),
                Pad(pad_to=(crop_small, crop_large)),
            ])
        ])

        val_transfroms = slc.Stream([
            Pad(pad_to=(crop_small, crop_large)),
            Crop(magnification, crop_mode='r', crop_to=(crop_small, crop_large))
        ])

    return [transforms], val_transfroms
