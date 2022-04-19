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

from bone_enhance.utilities import load, save, print_orthogonal, render_volume, calculate_mean_std
from bone_enhance.inference import InferenceModel, inference, largest_object, load_models
from bone_enhance.models import ConvNet, EnhanceNet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main(args, config, args_experiment, sample_id=None, render=False, ds=False):
    # Save path
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
    model = InferenceModel(model_list, sigmoid=config.training.segmentation).to(device)
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
        # Remove file extensions
        sample_stem = Path(sample).stem

        # Print sample info
        print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample_stem}')

        # Load image stacks
        if sample.endswith('.h5'):
            with h5py.File(str(args.dataset_root / sample), 'r') as f:
                data_xy = f['data'][:]
        else:
            data_xy, files = load(str(args.dataset_root / sample), rgb=False, axis=(1, 2, 0), dicom=args.dicom)

        # Downscale input image
        if ds:
            factor = (data_xy.shape[0] // mag, data_xy.shape[1] // mag, data_xy.shape[2] // mag)
            data_xy = resize(data_xy, factor, order=0, anti_aliasing=True, preserve_range=True)

        # Channel dimension
        if len(data_xy.shape) != 4:
            data_xy = np.expand_dims(data_xy, -1)
        if config.training.rgb:
            data_xy = np.repeat(data_xy, 3, axis=-1)

        # Visualize input stack
        print_orthogonal(data_xy[:, :, :, 0], invert=True, res=args.res, title='Input', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample_stem + f'_{snapshot}_input.png')), scale_factor=100)

        # Calculate mean and std from the sample
        if args.calculate_mean_std:
            mean, std = calculate_mean_std(data_xy, config.training.rgb)


        # In case of MRI, make the resolution isotropic
        if args.mri:
            slice_thickness = 1.0
            anisotropy_factor = slice_thickness / args.res
            data_xy = zoom(data_xy, zoom=(1, 1, anisotropy_factor, 1))
            print_orthogonal(data_xy[:, :, :, 0], invert=True, res=args.res, title='Input (interpolated)', cbar=True,
                             savepath=str(args.save_dir / 'visualizations' / (sample_stem + f'_{snapshot}_input_scaled.png')), scale_factor=100)

        # Copy the stack into other orthogonal planes
        if args.avg_planes:
            data_xz = np.transpose(data_xy, (0, 2, 1, 3))  # X-Z-Y-Ch
            data_yz = np.transpose(data_xy, (1, 2, 0, 3))  # Y-Z-X-Ch

        # Interpolate 3rd dimension
        x, y, z, ch = data_xy.shape
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
                #out_xy = ((out_xy + np.transpose(out_xz, (0, 2, 1)) + np.transpose(out_yz, (2, 0, 1))) / 3).astype('float32')
                out_xy += np.transpose(out_xz, (0, 2, 1))
                del out_xz
                out_xy += np.transpose(out_yz, (2, 0, 1))
                del out_yz
                out_xy = (out_xy / 3).astype('float32')

        # Scale the dynamic range
        pred_max = np.max(out_xy)
        if args.scale:
            out_xy -= np.min(out_xy)
            out_xy /= pred_max
        elif pred_max > 1:
            print(f'Maximum value {pred_max} will be scaled to one')
            out_xy /= pred_max

        out_xy = (out_xy * 255).astype('uint8')

        # Save predicted full mask
        save(str(args.save_dir / sample_stem), sample_stem, out_xy, dtype=args.dtype)
        if render:
            render_volume(data_yz[:, :, :, 0] * out_xy,
                          savepath=str(args.save_dir / 'visualizations' / (sample_stem + '_render' + args.dtype)),
                          white=True, use_outline=False)

        # Visualize output
        print_orthogonal(out_xy, invert=True, res=args.res / 4, title='Output', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample_stem + f'_{snapshot}_prediction.png')),
                         scale_factor=100)

    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')


if __name__ == "__main__":
    start = time()

    # Single snapshot
    snap = '2020_12_15_10_28_57_2D_perceptualnet_ds_16'  # Latest 2D model with fixes, only 1 fold
    snap = '2021_01_08_09_49_45_2D_perceptualnet_ds_16'  # 2D model, 3 working folds

    # List all snapshots from a path
    snap_path = '../../Workdir/wacv_experiments_new_2D'
    #snap_path = '../../Workdir/IVD_experiments_2D'
    #snap_path = '../../Workdir/snapshots'
    snaps = os.listdir(snap_path)
    snaps.sort()
    snaps = [snap for snap in snaps if os.path.isdir(os.path.join(snap_path, snap))]
    # Skip snapshots
    #snaps = [snaps[-1]]
    # List of specific snapshots
    #snaps = ['2021_05_27_08_56_20_2D_perceptual_tv_IVD_4x_pretrained_seed42']
    snaps = ['2021_06_11_11_59_53_2D_perceptual_tv_1176_seed10', '2021_06_10_23_57_51_2D_ssim_1176_seed10',
             '2021_06_10_23_24_54_2D_mse_tv_1176_seed10']

    for snap_id in range(len(snaps)):

        snap = snaps[snap_id]
        print(f'Calculating inference for snapshot: {snap} {snap_id+1}/{len(snaps)}')

        parser = argparse.ArgumentParser()
        #parser.add_argument('--dataset_root', type=Path, default='/media/dios/kaappi/Santeri/BoneEnhance/Clinical data')
        #parser.add_argument('--dataset_root', type=Path, default='../../Data/Fantomi/H5B-fantomi/Series1/Series1/')
        parser.add_argument('--dataset_root', type=Path, default='../../Data/dental/')
        #parser.add_argument('--dataset_root', type=Path, default='../../Data/Test_set_(full)/input_3d')
        #parser.add_argument('--dataset_root', type=Path, default='../../Data/MRI_IVD/Repeatability/')
        #parser.add_argument('--save_dir', type=Path, default=f'../../Data/predictions_3D_clinical/IVD_experiments/{snap}_avg')
        parser.add_argument('--save_dir', type=Path,
                            default=f'../../Data/predictions_3D_clinical/dental_experiments/{snap}_single')
        parser.add_argument('--bs', type=int, default=64)
        parser.add_argument('--step', type=int, default=3)
        parser.add_argument('--plot', type=bool, default=False)
        parser.add_argument('--calculate_mean_std', type=bool, default=True)
        parser.add_argument('--scale', type=bool, default=False)
        parser.add_argument('--dicom', type=bool, default=False, help='Is DICOM format used for loading?')
        parser.add_argument('--weight', type=str, choices=['gaussian', 'mean', 'pyramid'], default='gaussian')
        parser.add_argument('--completed', type=int, default=0)
        parser.add_argument('--res', type=float, default=0.200, help='Input image pixel size')
        parser.add_argument('--sample_id', type=list, default=11, help='Process specific samples unless None.')
        parser.add_argument('--avg_planes', type=bool, default=False)
        parser.add_argument('--mri', type=bool, default=False, help='Is anisotropic MRI data used?')
        parser.add_argument('--snapshot', type=Path,
                            default=os.path.join(snap_path, snap))
        parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
        args = parser.parse_args()

        # Load snapshot configuration
        with open(args.snapshot / 'config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = OmegaConf.create(config)

        with open(args.snapshot / 'args.dill', 'rb') as f:
            args_experiment = dill.load(f)

        args.save_dir.parent.mkdir(exist_ok=True)
        args.save_dir.mkdir(exist_ok=True)

        main(args, config, args_experiment, sample_id=args.sample_id)
