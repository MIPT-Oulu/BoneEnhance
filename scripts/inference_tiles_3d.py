import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import dill
import torch
import torch.nn as nn
import yaml
from time import time
from tqdm import tqdm
from glob import glob
from scipy.ndimage import zoom
from skimage.metrics import mean_squared_error
from omegaconf import OmegaConf
from skimage.transform import resize
import h5py

from BoneEnhance.components.utilities import load, save, print_orthogonal, render_volume
from BoneEnhance.components.inference import InferenceModel, inference, largest_object, load_models, inference_3d
from BoneEnhance.components.models import ConvNet, EnhanceNet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()

    # snap = 'dios-erc-gpu_2020_10_12_09_40_33_perceptualnet_newsplit'
    #snap = 'dios-erc-gpu_2020_11_04_15_58_20_3D_perceptualnet_ds'
    snap = 'dios-erc-gpu_2020_11_04_14_10_25_3D_perceptualnet_scratch'  # Perceptual scratch, abstract
    # snap = 'dios-erc-gpu_2020_09_30_14_14_42_perceptualnet_noscaling_3x3_cm_curated_trainloss'
    #snap = '2020_12_11_07_10_16_3D_perceptualnet_ds_16'  # Intensity augmentations applied
    snap = '2021_01_11_05_41_47_3D_perceptualnet_ds_autoencoder_16'  # Autoencoder, 4 folds, 2 layers
    # Segmentation models
    #snap = '2021_03_25_13_51_14_rn50_UNet_bcejci'
    #snap = '2021_03_25_13_51_14_rn50_fpn_bcejci'
    # 1176 models
    snap = '2021_03_03_11_52_07_1_3D_mse_tv_1176_HR'  # High resolution 1176 model (mse+tv)
    # snap = '2021_03_04_10_11_34_1_3D_mse_tv_1176'  # Low resolution 1176 model (mse+tv)
    snap = '2021_02_26_05_52_47_3D_perceptualnet_ds_mse_tv'  # Perceptualnet downscaled
    snap = '2021_02_25_07_51_17_1_3D_perceptualnet_cm_perceptual_pretrained'
    ds = False
    snaps = ['2021_03_03_11_52_07_1_3D_mse_tv_1176_HR',  # High resolution 1176 model (mse+tv)
             #'2021_02_25_07_51_17_1_3D_perceptualnet_cm_perceptual_pretrained',
             #'2021_02_26_05_52_47_3D_perceptualnet_ds_mse_tv',  # Perceptualnet downscaled
             #'2021_02_24_12_30_02_3D_perceptualnet_cm_mse_tv',  # Perceptualnet CBCT
             ]
    # Segmentation models (no magnification)
    #snaps = [
    #    '2021_03_25_13_51_14_rn50_UNet_bcejci',
    #    '2021_03_25_13_51_14_rn50_fpn_bcejci'
    #]
    for snap in snaps:
        print(f'Running inference for model {snap}')

        parser = argparse.ArgumentParser()
        if ds:
            parser.add_argument('--dataset_root', type=Path, default='../../Data/Test set (KP02)/target_3d')
        else:
            parser.add_argument('--dataset_root', type=Path, default='../../Data/Test set (KP02)/input_3d')
        parser.add_argument('--save_dir', type=Path, default=f'../../Data/Test set (KP02)/predictions_test/{snap}')
        parser.add_argument('--bs', type=int, default=12)
        parser.add_argument('--step', type=int, default=1)
        parser.add_argument('--plot', type=bool, default=False)
        parser.add_argument('--segmentation', type=bool, default=False)
        parser.add_argument('--calculate_mean_std', type=bool, default=False)
        parser.add_argument('--weight', type=str, choices=['gaussian', 'mean'], default='mean')
        parser.add_argument('--completed', type=int, default=0)
        parser.add_argument('--avg_planes', type=bool, default=False)
        parser.add_argument('--snapshot', type=Path,
                            default=f'../../Workdir/snapshots/{snap}')
        parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
        args = parser.parse_args()

        # Load snapshot configuration
        with open(args.snapshot / 'config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = OmegaConf.create(config)

        with open(args.snapshot / 'args.dill', 'rb') as f:
            args_experiment = dill.load(f)

        with open(args.snapshot / 'split_config.dill', 'rb') as f:
            split_config = dill.load(f)

        # Create directories
        args.save_dir.parent.mkdir(exist_ok=True)
        args.save_dir.mkdir(exist_ok=True)
        (args.save_dir / 'visualizations').mkdir(exist_ok=True)

        # Load models
        device = 'cuda'  # Use the second GPU for inference

        if config.training.crossmodality:
            cm = 'cm'
        else:
            cm = 'ds'
        crop = config.training.crop_small
        config.training.bs = args.bs
        mag = config.training.magnification
        mean_std_path = args.snapshot.parent / f"mean_std_{crop}_{cm}.pth"
        tmp = torch.load(mean_std_path)
        mean, std = tmp['mean'], tmp['std']

        # List the models
        model_list = load_models(str(args.snapshot), config, n_gpus=args_experiment.gpus)

        model = InferenceModel(model_list, sigmoid=args.segmentation).to(device)
        model.eval()
        print(f'Found {len(model_list)} models.')

        # Load samples
        # samples = [os.path.basename(x) for x in glob(str(args.dataset_root / '*XZ'))]  # Load with specific name
        samples = os.listdir(args.dataset_root)
        samples.sort()
        #samples = [samples[id] for id in [106]]  # Get intended samples from list

        # Skip the completed samples
        if args.completed > 0:
            samples = samples[args.completed:]

        # Main loop
        for idx, sample in enumerate(samples):
            print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample}')

            # Load image stacks
            #data_xy, files = load(str(args.dataset_root / sample), rgb=True, axis=(1, 2, 0))
            with h5py.File(str(args.dataset_root / sample), 'r') as f:
                data_original = f['data'][:]

            if ds:
                # Resize target with the given magnification to provide the input image
                factor = (data_original.shape[0] // mag, data_original.shape[1] // mag, data_original.shape[2] // mag)
                data = resize(data_original, factor, order=0, anti_aliasing=True, preserve_range=True)

                if args.segmentation:
                    factor = (data_original.shape[0], data_original.shape[1], data_original.shape[2])
                    data = resize(data, factor, order=3, anti_aliasing=True, preserve_range=True)
            else:
                data = data_original

            # 3-channel
            data = np.expand_dims(data, 3)
            data = np.repeat(data, 3, axis=3)

            x, y, z, ch = data.shape

            print_orthogonal(data[:, :, :, 0], invert=True, res=0.2, title='Input', cbar=True,
                             savepath=str(args.save_dir / 'visualizations' / (sample[:-3] + '_input.png')),
                             scale_factor=1000)

            # Calculate mean and std from the sample
            if args.calculate_mean_std:
                mean = torch.Tensor([np.mean(data) / 255])
                std = torch.Tensor([np.std(data) / 255])

            # Loop for image slices
            # 1st orientation
            with torch.no_grad():  # Do not update gradients
                if args.segmentation:
                    out = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
                    for slice_idx in tqdm(range(data.shape[2]), desc='Running inference, XY'):
                        out[:, :, slice_idx] = inference(model, args, config, data[:, :, slice_idx, :],
                                                         tile=args.step,
                                                         mean=mean, std=std)
                    data = out
                else:
                    data = inference_3d(model, args, config, data, step=args.step, mean=mean, std=std, plot=args.plot)


            if ds:
                print(f'Sample: {sample}, MSE: {mean_squared_error(data, data_original / 255.)}')

                print_orthogonal(data_original, invert=True, res=0.2 / 4, title='Target', cbar=True,
                                 savepath=str(args.save_dir / 'visualizations' / (sample[:-3] + '_target.png')),
                                 scale_factor=1000)
            # Scale the dynamic range
            #data -= np.min(data)
            #data /= np.max(data)

            # Convert to uint8
            data = (data * 255).astype('uint8')

            print_orthogonal(data, invert=True, res=0.2 / 4, title='Output', cbar=True,
                             savepath=str(args.save_dir / 'visualizations' / (sample[:-3] + '_prediction.png')),
                             scale_factor=1000)

            # Save predicted full mask
            save(str(args.save_dir / sample[:-3]), sample[:-3], data, dtype=args.dtype)

            """
            render_volume(data_yz[:, :, :, 0] * mask_final,
                          savepath=str(args.save_dir / 'visualizations' / (sample + '_render' + args.dtype)),
                          white=True, use_outline=False)
            """

    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
