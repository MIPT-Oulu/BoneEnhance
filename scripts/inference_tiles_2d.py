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
from skimage.transform import resize
from omegaconf import OmegaConf

from bone_enhance.utilities import load, save, print_orthogonal, render_volume
from bone_enhance.inference import InferenceModel, inference, largest_object, load_models
from bone_enhance.models import ConvNet, EnhanceNet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()

    snap = 'dios-erc-gpu_2020_10_12_09_40_33_perceptualnet_newsplit'
    #snap = 'dios-erc-gpu_2020_10_19_14_09_24_3D_perceptualnet'
    #snap = 'dios-erc-gpu_2020_09_30_14_14_42_perceptualnet_noscaling_3x3_cm_curated_trainloss'
    snap = '2021_01_08_09_49_45_2D_perceptualnet_ds_16'  # 2D model, 3 working folds
    snap = '2021_02_04_13_02_05_rn18_fpn'  # Segmentation model
    snap = '2021_02_10_05_29_09_rn50_UNet'
    snap = '2021_02_10_05_29_09_rn50_fpn'
    #snap = '2021_02_04_13_02_05_rn34_fpn'  # Gives error


    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset_root', type=Path, default='/media/dios/kaappi/Santeri/BoneEnhance/Clinical data')
    parser.add_argument('--dataset_root', type=Path, default='../../Data/Test set (KP02)/input_original')
    parser.add_argument('--save_dir', type=Path, default=f'../../Data/predictions_2D/{snap}')
    parser.add_argument('--subdir', type=Path, choices=['NN_prediction', ''], default='')
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--tile', type=int, default=4)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
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
    args.save_dir.mkdir(exist_ok=True)
    (args.save_dir / 'visualizations').mkdir(exist_ok=True)

    segmentation = config.training.architecture == 'encoderdecoder'

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
    model_list = load_models(str(args.snapshot), config, n_gpus=args_experiment.gpus, fold=None)

    model = InferenceModel(model_list, sigmoid=segmentation).to(device)
    model.eval()
    print(f'Found {len(model_list)} models.')

    # Load samples
    # samples = [os.path.basename(x) for x in glob(str(args.dataset_root / '*XZ'))]  # Load with specific name
    samples = os.listdir(args.dataset_root)
    samples.sort()

    #samples = [samples[id] for id in [6]]  # Get intended samples from list

    # Skip the completed samples
    if args.completed > 0:
        samples = samples[args.completed:]

    # Main loop
    for idx, sample in enumerate(samples):
        print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample}')

        # Load image stacks
        data_xy, files = load(str(args.dataset_root / sample), rgb=config.training.rgb, axis=(1, 2, 0))
        x, y, z, ch = data_xy.shape

        # Interpolate image size for segmentation
        if segmentation:
            new_size = (data_xy.shape[0] * mag, data_xy.shape[1] * mag, data_xy.shape[2] * mag, 3)
            data_xy = resize(data_xy, new_size, order=3, preserve_range=True) / 255.

            # Output shape
            out_xy = np.zeros((x * mag, y * mag, z * mag))
        else:
            # Output shape
            out_xy = np.zeros((x * mag, y * mag, z))

        print_orthogonal(data_xy[:, :, :, 0], invert=True, res=0.2, title='Input', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample + '_input.png')),
                         scale_factor=1000)

        # Loop for image slices
        # 1st orientation
        with torch.no_grad():  # Do not update gradients

            for slice_idx in tqdm(range(data_xy.shape[2]), desc='Running inference, XY'):
                out_xy[:, :, slice_idx] = inference(model, args, config, data_xy[:, :, slice_idx, :], step=args.tile,
                                                    mean=mean, std=std)

        # Scale the dynamic range
        out_xy -= np.min(out_xy)
        out_xy /= np.max(out_xy)

        out_xy = (out_xy * 255).astype('uint8')

        # Save predicted full mask
        save(str(args.save_dir / sample), sample, out_xy, dtype=args.dtype)
        """
        render_volume(data_yz[:, :, :, 0] * mask_final,
                      savepath=str(args.save_dir / 'visualizations' / (sample + '_render' + args.dtype)),
                      white=True, use_outline=False)
        """

        print_orthogonal(out_xy, invert=True, res=0.2 / 4, title='Output', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample + '_prediction.png')),
                         scale_factor=1000)

    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
