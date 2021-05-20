import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import h5py
import dill
import torch
import torch.nn as nn
import yaml
from time import time
from tqdm import tqdm
from glob import glob
from scipy.ndimage import zoom
from skimage.transform import resize
from omegaconf import OmegaConf

from bone_enhance.utilities import load, save, print_orthogonal, render_volume
from bone_enhance.inference import InferenceModel, inference, largest_object, load_models
from bone_enhance.models import ConvNet, EnhanceNet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main(args, config, args_experiment, sample_id=None, render=False, res=0.2, ds=False):
    #Save path
    args.save_dir.mkdir(exist_ok=True)
    (args.save_dir / 'visualizations').mkdir(exist_ok=True)
    snapshot = args.snapshot.name

    # Load models
    models = glob(str(args.snapshot) + '/*fold_[0-9]_*.pth')
    # models = glob(str(args.snapshot) + '/*fold_3_*.pth')
    models.sort()
    # device = auto_detect_device()
    device = 'cuda'  # Use the second GPU for inference

    crop = config.training.crop_small
    config.training.bs = args.bs
    mag = config.training.magnification
    if config.training.crossmodality:
        cm = 'cm'
    else:
        cm = 'ds'
    mean_std_path = args.snapshot.parent / f'mean_std_{crop}_{cm}.pth'
    tmp = torch.load(mean_std_path)
    mean, std = tmp['mean'], tmp['std']

    # List the models
    model_list = load_models(str(args.snapshot), config, n_gpus=args_experiment.gpus)

    model = InferenceModel(model_list).to(device)
    model.eval()
    print(f'Found {len(model_list)} models.')

    # Load samples
    # samples = [os.path.basename(x) for x in glob(str(args.dataset_root / '*XZ'))]  # Load with specific name
    samples = os.listdir(args.dataset_root)
    samples.sort()
    if sample_id is not None:
        samples = [samples[id] for id in [sample_id]]  # Get intended samples from list

    # Skip the completed samples
    if args.completed > 0:
        samples = samples[args.completed:]

    # Main loop
    for idx, sample in enumerate(samples):
        print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample}')

        # Load image stacks
        if sample.endswith('.h5'):
            with h5py.File(str(args.dataset_root / sample), 'r') as f:
                data_xy = f['data'][:]
        else:
            data_xy, files = load(str(args.dataset_root / sample), rgb=False, axis=(1, 2, 0))

        if ds:
            factor = (data_xy.shape[0] // mag, data_xy.shape[1] // mag, data_xy.shape[2] // mag)
            data_xy = resize(data_xy, factor, order=0, anti_aliasing=True, preserve_range=True)

        if len(data_xy.shape) != 4:
            data_xy = np.expand_dims(data_xy, -1)
        x, y, z, ch = data_xy.shape


        print_orthogonal(data_xy[:, :, :, 0], invert=True, res=res, title='Input', cbar=True,
                         savepath=str(args.visualizations / (sample[:-3] + f'_{snapshot}_input.png')), scale_factor=100)

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
            for slice_idx in tqdm(range(data_xy.shape[2]), desc='Running inference, XY'):
                out_xy[:, :, slice_idx] = inference(model, args, config, data_xy[:, :, slice_idx, :],
                                                    weight=args.weight, step=args.step, mean=mean, std=std)

            # 2nd and 3rd orientation
            if args.avg_planes:
                for slice_idx in tqdm(range(data_xz.shape[2]), desc='Running inference, XZ'):
                    out_xz[:, :, slice_idx] = inference(model, args, config, data_xz[:, :, slice_idx, :],
                                                        weight=args.weight, step=args.step, mean=mean, std=std)
                for slice_idx in tqdm(range(data_yz.shape[2]), desc='Running inference, YZ'):
                    out_yz[:, :, slice_idx] = inference(model, args, config, data_yz[:, :, slice_idx, :],
                                                        weight=args.weight, step=args.step, mean=mean, std=std)

        # Average probability maps
        if args.avg_planes:
            prediction = ((out_xy + np.transpose(out_xz, (0, 2, 1)) + np.transpose(out_yz, (2, 0, 1))) / 3)

            # Free memory
            del out_xz, out_yz
        else:
            prediction = out_xy

        # Scale the dynamic range
        pred_max = np.max(prediction)
        if args.scale:
            prediction -= np.min(prediction)
            prediction /= pred_max
        elif pred_max > 1:
            print(f'Maximum value {pred_max} will be scaled to one')
            prediction /= pred_max

        prediction = (prediction * 255).astype('uint8')

        # Save predicted full mask
        save(str(args.save_dir / sample), sample, prediction, dtype=args.dtype)
        if render:
            render_volume(data_yz[:, :, :, 0] * prediction,
                          savepath=str(args.save_dir / 'visualizations' / (sample + '_render' + args.dtype)),
                          white=True, use_outline=False)

        print_orthogonal(prediction, invert=True, res=res / 4, title='Output', cbar=True,
                         savepath=str(args.visualizations / (sample[:-3] + f'_{snapshot}_prediction.png')),
                         scale_factor=100)

    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')


if __name__ == "__main__":
    start = time()

    snap = 'dios-erc-gpu_2020_10_12_09_40_33_perceptualnet_newsplit'
    #snap = 'dios-erc-gpu_2020_10_19_14_09_24_3D_perceptualnet'
    #snap = 'dios-erc-gpu_2020_09_30_14_14_42_perceptualnet_noscaling_3x3_cm_curated_trainloss'
    snap = '2020_12_15_10_28_57_2D_perceptualnet_ds_16'  # Latest 2D model with fixes, only 1 fold
    snap = '2021_01_08_09_49_45_2D_perceptualnet_ds_16'  # 2D model, 3 working folds

    #snap_path = '../../Workdir/ankle_experiments_2D'
    snap_path = '../../Workdir/snapshots'
    snaps = os.listdir(snap_path)
    snaps.sort()
    snaps = [snap for snap in snaps if os.path.isdir(os.path.join(snap_path, snap))]
    snaps = snaps[2:]
    snaps = ['2021_05_11_10_04_40_2D_perceptual_tv_1176_HR_seed42']
    #snaps = ['2021_05_05_10_55_23_2D_perceptual_tv_1176_seed10']

    for snap_id in range(len(snaps)):

        snap = snaps[snap_id]
        print(f'Calculating inference for snapshot: {snap} {snap_id+1}/{len(snaps)}')

        parser = argparse.ArgumentParser()
        #parser.add_argument('--dataset_root', type=Path, default='/media/dios/kaappi/Santeri/BoneEnhance/Clinical data')
        parser.add_argument('--dataset_root', type=Path, default='../../Data/Test set (KP02)/input_3d')
        #parser.add_argument('--save_dir', type=Path, default=f'../../Data/predictions_3D_clinical/ankle_experiments/{snap}')
        parser.add_argument('--save_dir', type=Path,
                            default=f'../../Data/Test set (KP02)/predictions_test/{snap}')
        #parser.add_argument('--visualizations', type=Path,
        #                    default=f'../../Data/predictions_3D_clinical/ankle_experiments/visualization')
        parser.add_argument('--visualizations', type=Path,
                            default=f'../../Data/Test set (KP02)/predictions_test/visualization')
        parser.add_argument('--subdir', type=Path, choices=['NN_prediction', ''], default='')
        parser.add_argument('--bs', type=int, default=64)
        parser.add_argument('--step', type=int, default=3)
        parser.add_argument('--plot', type=bool, default=False)
        parser.add_argument('--calculate_mean_std', type=bool, default=True)
        parser.add_argument('--scale', type=bool, default=False)
        parser.add_argument('--weight', type=str, choices=['gaussian', 'mean', 'pyramid'], default='gaussian')
        parser.add_argument('--completed', type=int, default=0)
        parser.add_argument('--avg_planes', type=bool, default=True)
        parser.add_argument('--snapshot', type=Path,
                            default=os.path.join(snap_path, snap))
        parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
        args = parser.parse_args()
        #subdir = 'NN_prediction'  # 'NN_prediction'

        # Load snapshot configuration
        with open(args.snapshot / 'config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = OmegaConf.create(config)

        with open(args.snapshot / 'args.dill', 'rb') as f:
            args_experiment = dill.load(f)

        args.save_dir.mkdir(exist_ok=True)
        args.visualizations.mkdir(exist_ok=True)

        main(args, config, args_experiment, sample_id=None, res=0.2)
