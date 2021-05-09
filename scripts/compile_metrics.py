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
from time import strftime
import argparse
import dill
import yaml

from collagen.core.utils import auto_detect_device
from bone_enhance.inference.model_components import load_models
from bone_enhance.utilities import load, calculate_bvtv, threshold

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset_root', type=Path, default=f'/media/santeri/data/BoneEnhance/Data/evaluation_ankle')
    parser.add_argument('--dataset_root', type=Path, default=f'/media/santeri/data/BoneEnhance/Data/predictions_3D_clinical/ankle_experiments_eval')
    parser.add_argument('--save_dir', type=Path, default='/media/santeri/data/BoneEnhance/Data')
    parser.add_argument('--snapshots', type=Path, default='../../Workdir/snapshots/')
    args = parser.parse_args()

    # Load ground truth BVTV
    experiments = os.listdir(args.dataset_root)

    compiled_results = {'Experiment': [], 'MSE': [], 'PSNR': [], 'SSIM': []}
    for experiment in experiments:
        results = pd.read_excel(str(args.dataset_root / experiment), engine='openpyxl')

        # Remove unnecessary parts of experiment name
        exp = experiment.split('3D', 1)
        if len(exp) == 1:
            exp = '2D' + experiment.split('2D', 1)[-1][:-5]
        else:
            exp = '3D' + exp[-1][:-5]

        # Append to list
        compiled_results['Experiment'].append(exp)
        compiled_results['MSE'].append(results['MSE'].iloc[-1])
        compiled_results['PSNR'].append(results['PSNR'].iloc[-1])
        compiled_results['SSIM'].append(results['SSIM'].iloc[-1])

    # Write to excel
    writer = pd.ExcelWriter(str(args.save_dir / ('metrics_compiled_' + strftime(f'_%Y_%m_%d_%H_%M_%S'))) + '.xlsx')
    df1 = pd.DataFrame(compiled_results)
    df1.to_excel(writer, sheet_name='Metrics')
    writer.save()