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
from omegaconf import OmegaConf
from skimage.transform import resize
import h5py

from bone_enhance.utilities import load, save, print_orthogonal, render_volume, threshold
from bone_enhance.inference import InferenceModel, inference, largest_object, load_models, inference_3d
from bone_enhance.models import ConvNet, EnhanceNet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    #snap = 'dios-erc-gpu_2020_11_04_14_10_25_3D_perceptualnet_scratch'  # Perceptual scratch
    snap = '2020_12_07_09_36_17_3D_perceptualnet_ds_20'
    snap = '2020_12_10_09_16_07_3D_perceptualnet_ds_20'  # Brightness and contrast augmentations applied
    snap = '2020_12_11_07_10_16_3D_perceptualnet_ds_16'  # Intensity augmentations applied
    snap = '2020_12_14_07_26_07_3D_perceptualnet_ds_16'  # Intensity and spatial augs
    snap = '2020_12_21_12_58_39_3D_perceptualnet_ds_16'  # 2D perceptual loss, 3D model
    snap = '2021_01_05_09_21_06_3D_perceptualnet_ds_16'  # Autoencoder perceptual loss, 2 folds
    snap = '2021_01_11_05_41_47_3D_perceptualnet_ds_autoencoder_16'  # Autoencoder, 4 folds, 2 layers
    snap = '2021_02_21_11_12_11_3D_perceptualnet_ds_mse_tv'  # No perceptual loss
    #snap = '2021_03_02_14_55_25_1_3D_perceptualnet_ds_autoencoder_fullpass'  # Trained with 1176 data, 200µm resolution
    #snap = '2021_03_03_07_00_39_1_3D_perceptualnet_ds_autoencoder_fullpass'
    snap = '2021_03_03_11_52_07_1_3D_mse_tv_1176_HR'  # High resolution 1176 model (mse+tv)
    snap = '2021_03_04_10_11_34_1_3D_mse_tv_1176'  # Low resolution 1176 model (mse+tv)

    path = '../../Workdir/ankle_experiments'
    snaps = os.listdir(path)
    snaps.sort()
    snaps = [snap for snap in snaps if os.path.isdir(os.path.join(path, snap))]

    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default='../../Data/Clinical_data')
    parser.add_argument('--save_dir', type=Path, default=f'../../Data/predictions_clinical/{snap}')
    parser.add_argument('--visualizations', type=Path,
                        default=f'../../Data/predictions_clinical/visualization')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['gaussian', 'mean'], default='gaussian')
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--step', type=int, default=1, help='Factor for tile step size. 1=no overlap, 2=50% overlap...')
    parser.add_argument('--avg_planes', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=False, help='Whether to merge the inference tiles on GPU or CPU')
    parser.add_argument('--scale', type=bool, default=True, help='Whether to scale prediction to full dynamic range')
    parser.add_argument('--calculate_mean_std', type=bool, default=True, help='Whether to calculate individual mean and std')
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    args = parser.parse_args()

    args.snapshot = Path(os.path.join(path, snaps[args.exp_id - 1]))

    # Load snapshot configuration
    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = OmegaConf.create(config)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

    with open(args.snapshot / 'split_config.dill', 'rb') as f:
        split_config = dill.load(f)
    args.save_dir.mkdir(exist_ok=True)
    args.visualizations.mkdir(exist_ok=True)

    is_3d = len(config.training.crop_small) == 3

    # Load models
    device = 'cuda'  # Use the second GPU for inference

    crop = config.training.crop_small
    config.training.bs = args.bs
    mag = config.training.magnification
    if config.training.crossmodality:
        cm = 'cm'
    else:
        cm = 'ds'
    mean_std_path = args.snapshot.parent / f"mean_std_{crop}_{cm}.pth"
    tmp = torch.load(mean_std_path)
    mean, std = tmp['mean'], tmp['std']

    # List the models
    model_list = load_models(str(args.snapshot), config, n_gpus=args_experiment.gpus)#, fold=0)

    model = InferenceModel(model_list).to(device)
    model.eval()
    print(f'Found {len(model_list)} models.')

    # Load samples
    # samples = [os.path.basename(x) for x in glob(str(args.dataset_root / '*XZ'))]  # Load with specific name
    samples = os.listdir(args.dataset_root)
    samples.sort()
    samples = [samples[id] for id in [2]]  # Get intended samples from list

    # Skip the completed samples

    # Main loop
    if is_3d:
        for idx, sample in enumerate(samples):
            print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample}')

            # Load image stacks
            if sample.endswith('.h5'):
                with h5py.File(str(args.dataset_root / sample), 'r') as f:
                    data_xy = f['data'][:]
            else:
                data_xy, files = load(str(args.dataset_root / sample), rgb=True, axis=(1, 2, 0))

            # 3-channel
            if len(data_xy.shape) != 4 and config.training.rgb:
                data_xy = np.expand_dims(data_xy, 3)
                data_xy = np.repeat(data_xy, 3, axis=3)

            x, y, z, ch = data_xy.shape

            # Input image
            print_orthogonal(data_xy[:, :, :, 0], invert=True, res=0.2, title='Input', cbar=True,
                             savepath=str(args.visualizations / (sample[:-3] + f'_{snap}_input.png')), scale_factor=10)

            # Calculate mean and std from the sample
            if args.calculate_mean_std:
                mean = torch.Tensor([np.mean(data_xy) / 255])
                std = torch.Tensor([np.std(data_xy) / 255])

            # Loop for image slices
            with torch.no_grad():  # Do not update gradients
                prediction = inference_3d(model, args, config, data_xy, step=args.step, cuda=args.cuda, mean=mean, std=std,
                                          weight=args.weight)

            # Scale the dynamic range
            if args.scale:
                prediction -= np.min(prediction)
                prediction /= np.max(prediction)

            # Convert to uint8
            prediction = (prediction * 255).astype('uint8')

            # Save predicted full mask
            save(str(args.save_dir / (sample[:-3] + f'_{snap}')), sample, prediction, dtype=args.dtype)

            print_orthogonal(prediction, invert=True, res=50/1000, title='Output', cbar=True,
                             savepath=str(args.visualizations / (sample[:-3] + f'_{snap}_prediction.png')),
                             scale_factor=10)

        dur = time() - start
        print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
    else:
        # Main loop
        for idx, sample in enumerate(samples):
            print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample}')

            # Load image stacks
            if sample.endswith('.h5'):
                with h5py.File(str(args.dataset_root / sample), 'r') as f:
                    data_xy = f['data'][:]
            else:
                data_xy, files = load(str(args.dataset_root / sample), rgb=False, axis=(1, 2, 0))

            if len(data_xy.shape) != 4:
                data_xy = np.expand_dims(data_xy, -1)
            x, y, z, ch = data_xy.shape

            print_orthogonal(data_xy[:, :, :, 0], invert=True, res=0.2, title='Input', cbar=True,
                             savepath=str(args.visualizations / (sample + f'_{snap}_input.png')),
                             scale_factor=1000)

            # Calculate mean and std from the sample
            if args.calculate_mean_std:
                mean = torch.Tensor([np.mean(data_xy) / 255])
                std = torch.Tensor([np.std(data_xy) / 255])

            data_xz = np.transpose(data_xy, (0, 2, 1, 3))  # X-Z-Y-Ch
            data_yz = np.transpose(data_xy, (1, 2, 0, 3))  # Y-Z-X-Ch

            # Interpolate 3rd dimension
            data_xy = zoom(data_xy, zoom=(1, 1, config.training.magnification, 1))
            if args.avg_planes:
                data_xz = zoom(data_xz, zoom=(1, 1, config.training.magnification, 1))
                data_yz = zoom(data_yz, zoom=(1, 1, config.training.magnification, 1))

            # Output shape
            out_xy = np.zeros((x * mag, y * mag, z * mag))
            if args.avg_planes:
                out_xz = np.zeros((x * mag, z * mag, y * mag))
                out_yz = np.zeros((y * mag, z * mag, x * mag))

            # Loop for image slices
            # 1st orientation
            with torch.no_grad():  # Do not update gradients

                # 1st orientation
                for slice_idx in tqdm(range(data_xy.shape[2]), desc='Running inference, XY', disable=not args.verbose):
                    out_xy[:, :, slice_idx] = inference(model, args, config, data_xy[:, :, slice_idx, :],
                                                        weight=args.weight, tile=args.step)

                # 2nd and 3rd orientation
                if args.avg_planes:
                    for slice_idx in tqdm(range(data_xz.shape[2]), desc='Running inference, XZ', disable=not args.verbose):
                        out_xz[:, :, slice_idx] = inference(model, args, config, data_xz[:, :, slice_idx, :],
                                                            weight=args.weight, tile=args.step)
                    for slice_idx in tqdm(range(data_yz.shape[2]), desc='Running inference, YZ', disable=not args.verbose):
                        out_yz[:, :, slice_idx] = inference(model, args, config, data_yz[:, :, slice_idx, :],
                                                            weight=args.weight, tile=args.step)

            # Average probability maps
            if args.avg_planes:
                mask_avg = ((out_xy + np.transpose(out_xz, (0, 2, 1)) + np.transpose(out_yz, (2, 0, 1))) / 3)

                # Free memory
                del out_xz, out_yz
            else:
                mask_avg = out_xy

            # Scale the dynamic range
            # mask_avg -= np.min(mask_avg)
            # mask_avg /= np.max(mask_avg)

            mask_avg = (mask_avg * 255).astype('uint8')

            # Save predicted full mask
            save(str(args.save_dir / (sample[:-3] + f'_{snap}')), sample[:-3], mask_avg, dtype=args.dtype)

            print_orthogonal(mask_avg, invert=True, res=0.2 / 4, title='Output', cbar=True,
                             savepath=str(args.visualizations / (sample + f'_{snap}_prediction.png')),
                             scale_factor=1000)

        dur = time() - start
        print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')