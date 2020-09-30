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

from BoneEnhance.components.utilities import load, save, print_orthogonal, render_volume
from BoneEnhance.components.inference import InferenceModel, inference, largest_object
from BoneEnhance.components.models import ConvNet, EnhanceNet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default='../../Data/target_original')
    parser.add_argument('--save_dir', type=Path, default='../../Data/predictions_3D')
    parser.add_argument('--subdir', type=Path, choices=['NN_prediction', ''], default='')
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--completed', type=int, default=0)
    parser.add_argument('--avg_planes', type=bool, default=False)
    parser.add_argument('--snapshot', type=Path,
                        default='../../Workdir/snapshots/dios-erc-gpu_2020_09_11_12_22_28_enhance_standard')
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    args = parser.parse_args()
    #subdir = 'NN_prediction'  # 'NN_prediction'

    # Load snapshot configuration
    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = OmegaConf.create(config)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

    with open(args.snapshot / 'split_config.dill', 'rb') as f:
        split_config = dill.load(f)
    args.save_dir.mkdir(exist_ok=True)
    (args.save_dir / 'visualizations').mkdir(exist_ok=True)

    # Load models
    models = glob(str(args.snapshot) + '/*fold_[0-9]_*.pth')
    #models = glob(str(args.snapshot) + '/*fold_3_*.pth')
    models.sort()
    #device = auto_detect_device()
    device = 'cuda'  # Use the second GPU for inference

    crop = config.training.crop_small
    config.training.bs = args.bs
    mag = config.training.magnification
    mean_std_path = args.snapshot.parent / f"mean_std_{crop[0]}x{crop[1]}.pth"
    tmp = torch.load(mean_std_path)
    mean, std = tmp['mean'], tmp['std']

    # List the models
    model_list = []
    for fold in range(len(models)):
        if config.training.architecture == 'enhance' and args_experiment.gpus > 1:
            model = nn.DataParallel(EnhanceNet(config.training.crop_small, mag,
                              activation=config.training.activation,
                              add_residual=config.training.add_residual,
                              upscale_input=config.training.upscale_input))
        elif config.training.architecture == 'enhance':
            model = EnhanceNet(config.training.crop_small, mag,
                              activation=config.training.activation,
                              add_residual=config.training.add_residual,
                              upscale_input=config.training.upscale_input)
        elif args_experiment.gpus > 1:
            model = nn.DataParallel(ConvNet(mag,
                           activation=config.training.activation,
                           upscale_input=config.training.upscale_input,
                           n_blocks=config.training.n_blocks,
                           normalization=config.training.normalization))
        else:
            model = ConvNet(mag,
                           activation=config.training.activation,
                           upscale_input=config.training.upscale_input,
                           n_blocks=config.training.n_blocks,
                           normalization=config.training.normalization)
        model.load_state_dict(torch.load(models[fold]))
        model_list.append(model)

    model = InferenceModel(model_list).to(device)
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
        data_xy, files = load(str(args.dataset_root / sample), rgb=True, axis=(1, 2, 0))
        x, y, z, ch = data_xy.shape
        data_xz = np.transpose(data_xy, (0, 2, 1, 3))  # X-Z-Y-Ch
        data_yz = np.transpose(data_xy, (1, 2, 0, 3))  # Y-Z-X-Ch

        # Interpolate 3rd dimension
        data_xy = zoom(data_xy, zoom=(1, 1, config.training.magnification, 1))
        data_xz = zoom(data_xz, zoom=(1, 1, config.training.magnification, 1))
        data_yz = zoom(data_yz, zoom=(1, 1, config.training.magnification, 1))

        # Output shape
        out_xy = np.zeros((x * mag, y * mag, z * mag))
        out_xz = np.zeros((x * mag, z * mag, y * mag))
        out_yz = np.zeros((y * mag, z * mag, x * mag))

        # Loop for image slices
        # 1st orientation
        with torch.no_grad():  # Do not update gradients

            for slice_idx in tqdm(range(data_xy.shape[2]), desc='Running inference, XY'):
                out_xy[:, :, slice_idx] = inference(model, args, config, data_xy[:, :, slice_idx, :])

            # 2nd and 3rd orientation
            if args.avg_planes:
                for slice_idx in tqdm(range(data_xz.shape[2]), desc='Running inference, XZ'):
                    out_xz[:, :, slice_idx] = inference(model, args, config, data_xz[:, :, slice_idx, :])
                for slice_idx in tqdm(range(data_yz.shape[2]), desc='Running inference, YZ'):
                    out_yz[:, :, slice_idx] = inference(model, args, config, data_yz[:, :, slice_idx, :])

        # Average probability maps
        if args.avg_planes:
            #mask_avg = ((mask_xz + np.transpose(mask_yz, (0, 2, 1))) / 2)
            mask_avg = ((out_xy + np.transpose(out_xz, (0, 2, 1)) + np.transpose(out_yz, (2, 0, 1))) / 3)
        else:
            mask_avg = out_xy

        # Free memory
        del out_xz, out_yz

        mask_avg = (mask_avg * 255).astype('uint8')

        # Save predicted full mask
        save(str(args.save_dir / sample), files, mask_avg, dtype=args.dtype)
        """
        render_volume(data_yz[:, :, :, 0] * mask_final,
                      savepath=str(args.save_dir / 'visualizations' / (sample + '_render' + args.dtype)),
                      white=True, use_outline=False)
        """
        print_orthogonal(data_xy[:, :, :, 0], invert=True, res=0.2, title='Input', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample + '_input.png')),
                         scale_factor=1000)
        print_orthogonal(mask_avg, invert=True, res=0.2/4, title='Output', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample + '_input.png')),
                         scale_factor=1000)

    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
