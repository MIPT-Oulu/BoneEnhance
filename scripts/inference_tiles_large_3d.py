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

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main(args, config, args_experiment, sample_id=None, render=False):
    # Create save directory
    args.save_dir.mkdir(exist_ok=True)
    (args.save_dir / 'visualizations').mkdir(exist_ok=True)

    # Parameters
    crop = config.training.crop_small
    config.training.bs = args.bs
    device = 'cuda'  # Use the second GPU for inference
    if config.training.crossmodality:
        cm = 'cm'
    else:
        cm = 'ds'
    # Mean and std
    mean_std_path = args.snapshot.parent / f"mean_std_{crop}_{cm}.pth"
    tmp = torch.load(mean_std_path)
    mean, std = tmp['mean'], tmp['std']

    # Load models
    model_list = load_models(str(args.snapshot), config, n_gpus=args_experiment.gpus)  # , fold=0)
    model = InferenceModel(model_list).to(device)
    model.eval()
    print(f'Found {len(model_list)} models.')

    # Load samples
    samples = os.listdir(args.dataset_root)
    samples.sort()
    if sample_id is not None:
        samples = [samples[id] for id in [sample_id]]  # Get intended samples from list

    # Skip the completed samples
    if args.completed > 0:
        samples = samples[args.completed:]

    # Main loop
    for idx, sample in enumerate(samples):
        # Remove file extensions
        sample_stem = Path(sample).stem

        print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample_stem}')

        # Load image stacks
        if sample.endswith('.h5'):
            with h5py.File(str(args.dataset_root / sample), 'r') as f:
                data_xy = f['data'][:]
        else:
            data_xy, files = load(str(args.dataset_root / sample), rgb=True, axis=(1, 2, 0))

        # Channel dimension
        if len(data_xy.shape) != 4:
            data_xy = np.expand_dims(data_xy, 3)
        if config.training.rgb and data_xy.shape[3] != 3:
            data_xy = np.repeat(data_xy, 3, axis=3)

        # Visualize input stack
        print_orthogonal(data_xy[:, :, :, 0], invert=True, res=args.res, title='Input', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample_stem + '_input.png')), scale_factor=10)

        # Calculate mean and std from the sample
        if args.calculate_mean_std:
            print('Calculating mean and std from the input')
            mean = torch.Tensor([np.mean(data_xy) / 255])
            std = torch.Tensor([np.std(data_xy) / 255])

        # Calculate inference
        with torch.no_grad():  # Do not update gradients
            prediction = inference_3d(model, args, config, data_xy, step=args.step, cuda=args.cuda, mean=mean, std=std,
                                      weight=args.weight, plot=args.plot)

        # Scale the dynamic range
        pred_max = np.max(prediction)
        if args.scale:
            prediction -= np.min(prediction)
            prediction /= pred_max
        elif pred_max > 1:
            print(f'Maximum value {pred_max} will be scaled to one')
            prediction /= pred_max

        # Convert to uint8
        prediction = (prediction * 255).astype('uint8')

        # Background removal
        if args.mask:
            data_xy = zoom(data_xy[:, :, :, 0], (4, 4, 4), order=3)
            mask = largest_object(np.invert(data_xy > 120), area_limit=10000).astype('bool')
            # Set BG to 0
            prediction[mask] = 0

        # Save predicted full mask
        save(str(args.save_dir / sample_stem), sample_stem, prediction, dtype=args.dtype)
        if render:
            render_volume(prediction,
                          savepath=str(args.save_dir / 'visualizations' / (sample_stem + '_render' + args.dtype)),
                          white=True, use_outline=False)

        # Visualize output
        print_orthogonal(prediction, invert=True, res=50 / 1000, title='Output', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample_stem + '_prediction_final.png')),
                         scale_factor=10)

    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')


if __name__ == "__main__":
    start = time()

    # Single snapshot
    path = '../../Workdir/snapshots/'
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
    #snap = '2021_03_04_10_11_34_1_3D_mse_tv_1176'  # Low resolution 1176 model (mse+tv)

    # List all snapshots from a path
    path = '../../Workdir/wacv_experiments_new'
    snaps = os.listdir(path)
    snaps = [snap for snap in snaps if os.path.isdir(os.path.join(path, snap))]
    # List of specific snapshots
    #snaps = ['2021_03_04_10_11_34_1_3D_mse_tv_1176']

    for snap_id in range(len(snaps)):
        # Print snapshot info
        snap = snaps[snap_id]
        print(f'Calculating inference for snapshot: {snap} {snap_id + 1}/{len(snaps)}')

        # Input arguments
        parser = argparse.ArgumentParser()
        #parser.add_argument('--dataset_root', type=Path, default='/media/dios/kaappi/Santeri/BoneEnhance/Clinical data')
        parser.add_argument('--dataset_root', type=Path, default='../../Data/Test set (full)/input_3d')
        #parser.add_argument('--save_dir', type=Path, default=f'../../Data/predictions_3D_clinical/ankle_experiments2/{snap}')
        parser.add_argument('--save_dir', type=Path,
                            default=f'../../Data/Test set (full)/predictions_wacv_new/{snap}')
        parser.add_argument('--bs', type=int, default=16)
        parser.add_argument('--plot', type=bool, default=False)
        parser.add_argument('--weight', type=str, choices=['gaussian', 'mean'], default='gaussian')
        parser.add_argument('--completed', type=int, default=0)
        parser.add_argument('--step', type=int, default=3, help='Factor for tile step size. 1=no overlap, 2=50% overlap...')
        parser.add_argument('--cuda', type=bool, default=False, help='Whether to merge the inference tiles on GPU or CPU')
        parser.add_argument('--mask', type=bool, default=False, help='Whether to remove background with postprocessing')
        parser.add_argument('--scale', type=bool, default=True, help='Whether to scale prediction to full dynamic range')
        parser.add_argument('--res', type=float, default=0.2, help='Input image pixel size')
        parser.add_argument('--calculate_mean_std', type=bool, default=True, help='Whether to calculate individual mean and std')
        parser.add_argument('--snapshot', type=Path, default=os.path.join(path, snap))
        parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
        args = parser.parse_args()

        # Load snapshot configuration
        with open(args.snapshot / 'config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = OmegaConf.create(config)

        with open(args.snapshot / 'args.dill', 'rb') as f:
            args_experiment = dill.load(f)

        main(args, config, args_experiment, sample_id=None)
