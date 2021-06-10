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
from bone_enhance.utilities import load, calculate_bvtv, threshold, print_orthogonal

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def evaluation_runner(args):

    # Evaluation arguments
    args.save_dir.mkdir(exist_ok=True)

    save_dir = os.listdir(args.pred_path)
    save_dir.sort()
    if 'visualization' in save_dir:
        save_dir.remove('visualization')

    # Snapshots to be evaluated
    if type(save_dir) != list:
        save_dir = [save_dir]

    # Iterate through snapshots
    for experiment in save_dir:
        experiment = args.pred_path / experiment
        # Initialize results
        results = {'Sample': [], 'MSE': [], 'PSNR': [], 'SSIM': []}

        # Sample list
        all_samples = os.listdir(str(experiment))
        samples = []
        for i in range(len(all_samples)):
            if os.path.isdir(str(experiment / all_samples[i])):
                samples.append(all_samples[i])
        samples.sort()
        if 'visualizations' in samples:
            samples.remove('visualizations')

        # Loop for samples
        for idx, sample in tqdm(enumerate(samples), total=len(samples), desc=f'Running evaluation for snap {experiment.stem}'):
            #try:
            # Load image stacks
            pred, files_pred = load(str(experiment / sample), axis=(1, 2, 0), rgb=False, n_jobs=args.num_threads)
            target, files_target = load(str(args.ref_path), axis=(1, 2, 0), rgb=False, n_jobs=args.num_threads)

            print_orthogonal(pred, invert=True, res=0.1, title='Predicted', cbar=True, scale_factor=10)
            print_orthogonal(target, invert=True, res=0.1, title='Target', cbar=True, scale_factor=10)

            # Crop in case of inconsistency
            if pred.shape != target.shape:
                print('Inconsistent shapes! Cropping...')
                crop = np.min((pred.shape, target.shape), axis=0)
                target = target[:crop[0], :crop[1], :crop[2]]
                pred = pred[:crop[0], :crop[1], :crop[2]].squeeze()

            # Evaluate metrics
            mse = mean_squared_error(target / 255., pred / 255.)
            psnr = peak_signal_noise_ratio(target / 255., pred / 255.)
            ssim = structural_similarity(target / 255., pred / 255.)

            # Update results
            results['Sample'].append(sample)
            results['MSE'].append(mse)
            results['PSNR'].append(psnr)
            results['SSIM'].append(ssim)

        # Add average value to
        results['Sample'].append('Average values')
        results['MSE'].append(np.average(results['MSE']))
        results['PSNR'].append(np.average(results['PSNR']))
        results['SSIM'].append(np.average(results['SSIM']))

        # Display on console
        res = (results['MSE'][-1], results['PSNR'][-1], results['SSIM'][-1])
        print(f'Average results:\n'
              f'MSE = {res[0]},\n'
              f'PSNR = {res[1]},\n'
              f'SSIM = {res[2]},\n')

        # Write to excel
        writer = pd.ExcelWriter(str(args.save_dir / ('metrics_' + str(experiment.name))) + '.xlsx')
        df1 = pd.DataFrame(results)
        df1.to_excel(writer, sheet_name='Metrics')
        writer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=Path, default=f'../../Data/predictions_3D_clinical/ankle_experiments2')
    parser.add_argument('--save_dir', type=Path, default='../../Data/predictions_3D_clinical/ankle_experiments_eval2')
    parser.add_argument('--ref_path', type=Path, default=f'../../Data/Test set (KP02)/ANKLE_SCALED_SMALLVOI_filtered')
    parser.add_argument('--num_threads', type=int, default=12)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=2)
    # µCT snapshot
    args = parser.parse_args()

    start = time()

    evaluation_runner(args)

    dur = time() - start
    print(f'Metrics evaluated in {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
