import os
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
import pandas as pd
import h5py
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import cv2
from omegaconf import OmegaConf
from pathlib import Path
from scipy.stats import pearsonr
from time import time
import argparse
import dill
import yaml

from collagen.core.utils import auto_detect_device
from bone_enhance.inference.model_components import load_models
from bone_enhance.utilities import load, calculate_bvtv, threshold

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def evaluation_runner(args, save_dir, masks=True, suffix='_3d'):

    # Evaluation arguments
    args.image_path = args.dataset_root / 'input'
    args.target_path = args.dataset_root / f'target{suffix}'
    args.masks = Path('/media/dios/kaappi/Sakke/Saskatoon/Verity/Registration')
    args.save_dir.mkdir(exist_ok=True)

    # Snapshots to be evaluated
    if type(save_dir) != list:
        save_dir = [save_dir]

    # Iterate through snapshots
    for snap in save_dir:

        # Initialize results
        results = {'Sample': [], 'MSE': [], 'PSNR': [], 'SSIM': [], 'BVTV': []}

        # Sample list
        all_samples = os.listdir(snap)
        samples = []
        for i in range(len(all_samples)):
            if os.path.isdir(str(snap / all_samples[i])):
                samples.append(all_samples[i])
        samples.sort()
        if 'visualizations' in samples:
            samples.remove('visualizations')
        # List the µCT target
        samples_target = os.listdir(args.target_path)
        samples_target.sort()
        # List VOI
        samples_voi = os.listdir(args.image_path)
        samples_voi.sort()

        # Loop for samples
        for idx, sample in tqdm(enumerate(samples), total=len(samples), desc=f'Running evaluation for snap {snap.stem}'):
            #try:
            # Load image stacks
            with h5py.File(str(args.target_path / samples_target[idx]), 'r') as f:
                target = f['data'][:]

            pred, files_pred = load(str(args.pred_path / snap.name / sample / 'conventional_segmentation_gray'), axis=(1, 2, 0), rgb=False,
                                    n_jobs=args.num_threads)

            # Crop in case of inconsistency
            crop = np.min((pred.shape, target.shape), axis=0)
            target = target[:crop[0], :crop[1], :crop[2]]
            pred = pred[:crop[0], :crop[1], :crop[2]].squeeze()

            # Evaluate metrics
            mse = mean_squared_error(target / 255., pred / 255.)
            psnr = peak_signal_noise_ratio(target / 255., pred / 255.)
            ssim = structural_similarity(target / 255., pred / 255.)

            # Binarize and calculate BVTV

            # Otsu thresholding
            if len(np.unique(pred)) != 2:
                pred, _ = threshold(pred, method=args.threshold)

            if masks:
                # Apply VOI
                voi, _ = load(str(args.masks / samples_voi[idx] / 'ROI'), axis=(1, 2, 0))
                voi = zoom(voi.squeeze(), (4, 4, 4), order=0)
                # Fix size mismatch
                size = np.min((voi.shape, pred.shape), axis=0)
                pred = np.logical_and(pred[:size[0], :size[1], :size[2]],
                                      voi[:size[0], :size[1], :size[2]])

                # Calculate BVTV
                bvtv = calculate_bvtv(pred, voi)
            else:
                # Cannot calculate bvtv without reference VOI
                bvtv = 0

            #print(f'Sample {sample}: MSE = {mse}, PSNR = {psnr}, SSIM = {ssim}, BVTV: {bvtv}')

            # Update results
            results['Sample'].append(sample)
            results['MSE'].append(mse)
            results['PSNR'].append(psnr)
            results['SSIM'].append(ssim)
            results['BVTV'].append(bvtv)

            #except (AttributeError, ValueError):
            #    print(f'Sample {sample} failing. Skipping to next one.')
            #    continue

        # Add average value to
        results['Sample'].append('Average values')
        results['MSE'].append(np.average(results['MSE']))
        results['PSNR'].append(np.average(results['PSNR']))
        results['SSIM'].append(np.average(results['SSIM']))
        results['BVTV'].append(np.average(results['BVTV']))

        # Load ground truth BVTV
        bvtv_test = pd.read_csv(str(args.bvtv_path), header=None)
        pearson = pearsonr(results['BVTV'][:-1], bvtv_test.iloc[:, 1])
        results['Pearson'] = np.zeros(len(results['MSE']))
        results['Pearson'][:2] = pearson

        # Display on console
        res = (results['MSE'][-1], results['PSNR'][-1], results['SSIM'][-1])
        print(f'Average results:\n'
              f'MSE = {res[0]},\n'
              f'PSNR = {res[1]},\n'
              f'SSIM = {res[2]},\n')
        print(f'Pearson correlation to µCT: {pearson[0]}, p = {pearson[1]}')


        # Write to excel
        writer = pd.ExcelWriter(str(args.save_dir / ('metrics_' + str(snap.name))) + '.xlsx')
        df1 = pd.DataFrame(results)
        df1.to_excel(writer, sheet_name='Metrics')
        writer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default=f'../../Data/Test set (KP02)')
    parser.add_argument('--pred_path', type=Path, default=#f'../../Data/Test set (KP02)/predictions_cbct')
                        '/media/dios/kaappi/Santeri/BoneEnhance/upscaled_images')
    parser.add_argument('--save_dir', type=Path, default='../../Data/Test set (KP02)/evaluation_cbct')
    parser.add_argument('--bvtv_path', type=Path, default='../../Data/BVTV_test.csv')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--magnification', type=int, default=4)
    parser.add_argument('--num_threads', type=int, default=12)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--threshold', type=str, choices=['otsu', 'mean'], default='otsu')
    parser.add_argument('--weight', type=str, choices=['gaussian', 'mean'], default='gaussian')
    # µCT snapshot
    parser.add_argument('--snapshots', type=Path, default='../../Workdir/snapshots/')
    args = parser.parse_args()

    # Snapshots to be evaluated
    # µCT models

    snaps = ['2021_03_03_11_52_07_1_3D_mse_tv_1176_HR',  # High resolution 1176 model (mse+tv)
             '2021_02_25_07_51_17_1_3D_perceptualnet_cm_perceptual_pretrained',
             '2021_02_26_05_52_47_3D_perceptualnet_ds_mse_tv',  # Perceptualnet downscaled
             '2021_02_24_12_30_02_3D_perceptualnet_cm_mse_tv',  # Perceptualnet CBCT
             '2021_03_25_13_51_14_rn50_UNet_bcejci',
             '2021_03_25_13_51_14_rn50_fpn_bcejci',
             '2021_03_31_22_06_00_2D_perceptualnet_cm',
             '2021_01_08_09_49_45_2D_perceptualnet_ds_16'
             ]
    snaps = ['Verity_TCI_test']
    suffixes = ['_3d'] * len(snaps)
    snaps = [args.snapshots / snap for snap in snaps]

    # Iterate through snapshots
    args.save_dir.mkdir(exist_ok=True)
    for idx, snap in enumerate(snaps):
        start = time()

        # Create directories
        save_dir = args.save_dir / str(snap.stem)
        save_dir.mkdir(exist_ok=True)

        device = auto_detect_device()

        masks = True
        evaluation_runner(args, args.pred_path / snap.stem, masks=masks, suffix=suffixes[idx])

        dur = time() - start
        print(f'Metrics evaluated in {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
