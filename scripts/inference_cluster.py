import numpy as np
import os
import sys
sys.path.append('/scratch/project_2002147/rytkysan/BoneEnhance/BoneEnhance')
sys.path.append('/projappl/project_2002147/miniconda3/lib/python3.7/site-packages')

from pathlib import Path
import argparse
import dill
import torch
import yaml
from time import time
from omegaconf import OmegaConf
from scipy.ndimage import zoom
import h5py

from bone_enhance.utilities import load, save, print_orthogonal, calculate_mean_std
from bone_enhance.inference import InferenceModel, load_models, inference_3d, inference

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default='/media/dios/kaappi/Santeri/BoneEnhance/Clinical data')
    parser.add_argument('--save_dir', type=Path, default=f'../../Data/predictions_3D_clinical')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['gaussian', 'mean'], default='gaussian')
    parser.add_argument('--completed', type=int, default=0)
    parser.add_argument('--res', type=float, default=0.200, help='Input image pixel size')
    parser.add_argument('--sample_id', type=int, default=None)
    parser.add_argument('--step', type=int, default=3, help='Factor for tile step size. 1=no overlap, 2=50% overlap...')
    parser.add_argument('--avg_planes', type=bool, default=False)
    parser.add_argument('--dicom', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=False, help='Whether to merge the inference tiles on GPU or CPU')
    parser.add_argument('--scale', type=bool, default=True, help='Whether to scale prediction to full dynamic range')
    parser.add_argument('--calculate_mean_std', type=bool, default=True, help='Whether to calculate individual mean and std')
    parser.add_argument('--snapshot', type=Path, default=f'../../Workdir/snapshots/')
    parser.add_argument('--snap_id', type=int, default=1)
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    args = parser.parse_args()

    snaps = os.listdir(args.snapshot)
    snaps = [snap for snap in snaps if os.path.isdir(os.path.join(args.snapshot, snap))]
    snaps.sort()
    args.snapshot = args.snapshot / snaps[args.snap_id - 1]
    args.save_dir = args.save_dir / args.snapshot.name

    # Load snapshot configuration
    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = OmegaConf.create(config)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

    # Create save directory
    args.save_dir.parent.mkdir(exist_ok=True)
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
    if args.sample_id is not None:
        samples = [samples[id] for id in [args.sample_id]]  # Get intended samples from list

    # Skip the completed samples
    if args.completed > 0:
        samples = samples[args.completed:]

    # Main loop
    for idx, sample in enumerate(samples):
        # Remove file extensions
        sample_stem = Path(sample).stem

        print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample_stem}')
        print(str(args.dataset_root / sample))

        # Load image stacks
        if sample.endswith('.h5'):
            with h5py.File(str(args.dataset_root / sample), 'r') as f:
                data_xy = f['data'][:]
        else:
            data_xy, files = load(str(args.dataset_root / sample), rgb=True, axis=(1, 2, 0), dicom=args.dicom)

        # Channel dimension
        if len(data_xy.shape) != 4:
            data_xy = np.expand_dims(data_xy, 3)
        if config.training.rgb and data_xy.shape[3] != 3:
            data_xy = np.repeat(data_xy, 3, axis=3)

            # Visualize input stack
            print_orthogonal(data_xy[:, :, :, 0], invert=True, res=args.res, title='Input', cbar=True,
                             savepath=str(args.save_dir / 'visualizations' / (sample_stem + '_input.png')), scale_factor=100)

        # Calculate mean and std from the sample
        if args.calculate_mean_std:
            print('Calculating mean and std from the input')
            mean, std = calculate_mean_std(data_xy, config.training.rgb)

        # Loop for image slices
        # 1st orientation
        if '3D' in args.snapshot.name:
            with torch.no_grad():  # Do not update gradients
                prediction = inference_3d(model, args, config, data_xy, step=args.step, cuda=args.cuda, mean=mean, std=std)
        else:
            # Copy the stack into other orthogonal planes
            mag = config.training.magnification
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
            prediction = np.zeros((x * mag, y * mag, z * mag))
            if args.avg_planes:
                out_xz = np.zeros((x * mag, z * mag, y * mag))
                out_yz = np.zeros((y * mag, z * mag, x * mag))

            # Loop for image slices
            # 1st orientation
            with torch.no_grad():  # Do not update gradients

                # 1st orientation
                print('Running inference, XY:')
                for slice_idx in range(data_xy.shape[2]):
                    prediction[:, :, slice_idx] = inference(model, args, config, data_xy[:, :, slice_idx, :],
                                                        weight=args.weight, step=args.step, mean=mean, std=std)

                # 2nd and 3rd orientation
                if args.avg_planes:
                    print('Running inference, XZ:')
                    for slice_idx in range(data_xz.shape[2]):
                        out_xz[:, :, slice_idx] = inference(model, args, config, data_xz[:, :, slice_idx, :],
                                                            weight=args.weight, step=args.step, mean=mean, std=std)
                    print('Running inference, YZ:')
                    for slice_idx in range(data_yz.shape[2]):
                        out_yz[:, :, slice_idx] = inference(model, args, config, data_yz[:, :, slice_idx, :],
                                                            weight=args.weight, step=args.step, mean=mean, std=std)

                    # Average probability maps
                    prediction += np.transpose(out_xz, (0, 2, 1))
                    del out_xz
                    prediction += np.transpose(out_yz, (2, 0, 1))
                    del out_yz
                    prediction = (prediction / 3).astype('float32')

        # Scale the dynamic range
        pred_max = np.max(prediction)
        if args.scale:
            prediction -= np.min(prediction)
            prediction /= pred_max
        elif pred_max > 1:
            print(f'Maximum value {pred_max} will be scaled to one')
            prediction[prediction > 1] = 1

        # Convert to uint8
        prediction = (prediction * 255).astype('uint8')

        # Save predicted full mask
        save(str(args.save_dir / sample_stem), sample, prediction, dtype=args.dtype)

        # Visualize output
        print_orthogonal(prediction, invert=True, res=args.res / config.training.magnification, title='Output', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample_stem + '_prediction_final.png')), scale_factor=100)

    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
