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

from BoneEnhance.components.utilities import load, save, print_orthogonal, render_volume, threshold
from BoneEnhance.components.inference import InferenceModel, inference, largest_object, load_models, inference_3d
from BoneEnhance.components.models import ConvNet, EnhanceNet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()

    #snap = 'dios-erc-gpu_2020_11_04_14_10_25_3D_perceptualnet_scratch'  # Perceptual scratch
    #snap = '2020_11_27_12_06_12_3D_perceptualnet_ds'
    #snap = '2020_12_01_10_52_00_3D_perceptualnet_ds'
    #snap = '2020_12_01_13_39_52_3D_perceptualnet_ds'
    snap = '2020_12_07_09_36_17_3D_perceptualnet_ds_20'
    snap = '2020_12_10_09_16_07_3D_perceptualnet_ds_20'  # Brightness and contrast augmentations applied
    snap = '2020_12_11_07_10_16_3D_perceptualnet_ds_16'  # Intensity augmentations applied

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default='/media/dios/kaappi/Santeri/BoneEnhance/Clinical data')
    parser.add_argument('--save_dir', type=Path, default=f'../../Data/predictions_3D_clinical/{snap}')
    parser.add_argument('--subdir', type=Path, choices=['NN_prediction', ''], default='')
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--completed', type=int, default=0)
    parser.add_argument('--step', type=int, default=2, help='Factor for tile step size. 1=no overlap, 2=50% overlap...')
    parser.add_argument('--avg_planes', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=False, help='Whether to merge the inference tiles on GPU or CPU')
    parser.add_argument('--mask', type=bool, default=False, help='Whether to remove background with postprocessing')
    parser.add_argument('--scale', type=bool, default=True, help='Whether to scale prediction to full dynamic range')
    parser.add_argument('--snapshot', type=Path, default=f'../../Workdir/snapshots/{snap}')
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
    model_list = load_models(str(args.snapshot), config, n_gpus=args_experiment.gpus)#, fold=3)

    model = InferenceModel(model_list).to(device)
    model.eval()
    print(f'Found {len(model_list)} models.')

    # Load samples
    # samples = [os.path.basename(x) for x in glob(str(args.dataset_root / '*XZ'))]  # Load with specific name
    samples = os.listdir(args.dataset_root)
    samples.sort()
    samples = [samples[id] for id in [3]]  # Get intended samples from list

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
            data_xy, files = load(str(args.dataset_root / sample), rgb=True, axis=(1, 2, 0))

        # 3-channel
        if len(data_xy.shape) != 4:
            data_xy = np.expand_dims(data_xy, 3)
            data_xy = np.repeat(data_xy, 3, axis=3)

        x, y, z, ch = data_xy.shape

        print_orthogonal(data_xy[:, :, :, 0], invert=True, res=0.2, title='Input', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample[:-3] + '_input.png')), scale_factor=10)

        # Loop for image slices
        # 1st orientation
        with torch.no_grad():  # Do not update gradients
            prediction = inference_3d(model, args, config, data_xy, step=args.step, cuda=args.cuda)
            #prediction, _ = load(str(args.save_dir / sample[:-3]), axis=(1, 2, 0))
            print_orthogonal(prediction, invert=True, res=50 / 1000, title='Output', cbar=True,
                             savepath=str(args.save_dir / 'visualizations' / (sample[:-3] + '_prediction.png')),
                             scale_factor=10)

        # Scale the dynamic range
        if args.scale:
            prediction -= np.min(prediction)
            prediction /= np.max(prediction)

        # Convert to uint8
        prediction = (prediction * 255).astype('uint8')

        # Background removal
        if args.mask:
            data_xy = zoom(data_xy[:, :, :, 0], (4, 4, 4), order=3)
            #mask = np.invert(mask > 120)
            #mask, _ = threshold(mask, method='otsu', block=51)
            mask = largest_object(np.invert(data_xy > 120), area_limit=10000).astype('bool')
            #print_orthogonal(mask)
            #contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            #for cnt in contour:
            #    cv2.drawContours(mask, [cnt], 0, 255, -1)
            # Set BG to 0
            prediction[mask] = 0
            # Set BG = TCI
            #prediction += data_xy * mask

        # Save predicted full mask
        save(str(args.save_dir / sample[:-3]), sample, prediction, dtype=args.dtype)
        #render_volume(prediction,
        #              savepath=str(args.save_dir / 'visualizations' / (sample + '_render' + args.dtype)),
        #              white=True, use_outline=False)

        print_orthogonal(prediction, invert=True, res=50/1000, title='Output', cbar=True,
                         savepath=str(args.save_dir / 'visualizations' / (sample[:-3] + '_prediction_final.png')),
                         scale_factor=10)

    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')