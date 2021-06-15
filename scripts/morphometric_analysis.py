import os
import matplotlib.pyplot as plt
from pathlib import Path
from time import time, strftime
import pandas as pd
from tqdm import tqdm

import numpy as np
import argparse

from bone_enhance.utilities import load, save, print_orthogonal, threshold, calculate_bvtv
from bone_enhance.inference.thickness_analysis import _local_thickness


if __name__ == '__main__':
    # ******************************** 3D case ************************************
    start = time()

    # Prediction path
    path = Path('../../Data/Test set (full)/predictions_wacv_new')
    snaps = os.listdir(path)
    snaps = [snap for snap in snaps if os.path.isdir(os.path.join(path, snap))]

    # TODO Load ground truth morphometrics

    for snap in snaps:
        parser = argparse.ArgumentParser()
        parser.add_argument('--masks', type=Path, default=path / 'trabecular_VOI')
        parser.add_argument('--save', type=Path, default=path / 'masks_wacv_new' / snap)
        parser.add_argument('--preds', type=Path, default=path / 'predictions_wacv_new' / snap)
        parser.add_argument('--plot', type=bool, default=True)
        parser.add_argument('--scale_voi', type=bool, default=False)
        parser.add_argument('--save_h5', type=bool, default=False)
        parser.add_argument('--batch_id', type=int, default=None)
        parser.add_argument('--resolution', type=tuple, default=(50, 50, 50))  # in µm
        parser.add_argument('--mode', type=str,
                            choices=['med2d_dist3d_lth3d', 'stacked_2d', 'med2d_dist2d_lth3d'],
                            default='med2d_dist3d_lth3d')
        parser.add_argument('--trabecular_number', type=str, choices=['3d', 'plate', 'rod'], default='rod')
        parser.add_argument('--max_th', type=float, default=None)  # in µm

        args = parser.parse_args()

        # Sample list
        samples = os.listdir(args.masks)
        samples.sort()
        if 'visualization' in samples:
            samples.remove('visualization')
        if args.batch_id is not None:
            samples = [samples[args.batch_id]]

        samples_pred = os.listdir(str(args.preds))
        samples_pred.sort()
        if 'visualizations' in samples_pred:
            samples_pred.remove('visualizations')

        # Save paths
        args.save.parent.mkdir(exist_ok=True)
        args.save.mkdir(exist_ok=True)
        (args.save / 'visualization').mkdir(exist_ok=True)
        if args.save_h5:
            (args.save / 'h5').mkdir(exist_ok=True)

        # Table for results
        results = {'Sample': [],
                   'Trabecular thickness': [], 'Trabecular separation': [], 'BVTV': [], 'Trabecular number': []}
        t = strftime(f'%Y_%m_%d_%H_%M')

        # Loop for samples
        for idx in tqdm(range(len(samples)), desc=f'Processing snapshot {snap}'):
            time_sample = time()
            sample = samples[idx]
            sample_pred = samples_pred[idx]

            # Load prediction and volume-of-interest
            pred, _ = load(str(args.preds / sample_pred), axis=(1, 2, 0,))
            voi, files = load(str(args.masks / sample), axis=(1, 2, 0,))

            if len(np.unique(pred)) != 2:
                pred, _ = threshold(pred)

            if args.plot:
                print_orthogonal(pred, savepath=str(args.save / 'visualization' / (sample + '_pred.png')), res=50 / 1000)

            # Apply volume-of-interest
            pred = np.logical_and(pred, voi).astype(np.uint8) * 255

            if args.plot:
                print_orthogonal(pred, savepath=str(args.save / 'visualization' / (sample + '_voi.png')), res=50 / 1000)

            #                       #
            # Morphometric analysis #
            #                       #

            # Thickness map
            th_map = _local_thickness(pred, mode=args.mode, spacing_mm=args.resolution, stack_axis=1,
                                      thickness_max_mm=args.max_th)

            # Bone volume fraction
            bvtv = calculate_bvtv(pred, voi) / 100

            # Update results
            th_map = th_map[np.nonzero(th_map)].flatten()
            tb_th = np.mean(th_map)
            results['Sample'].append(sample)
            results['Trabecular thickness'].append(tb_th)
            results['BVTV'].append(bvtv * 100)

            # Separation map
            pred = np.logical_and(np.invert(pred), voi).astype(np.uint8) * 255
            th_map = _local_thickness(pred, mode=args.mode, spacing_mm=args.resolution, stack_axis=1,
                                      thickness_max_mm=args.max_th)
            th_map = th_map[np.nonzero(th_map)].flatten()
            tb_sep = np.mean(th_map)
            results['Trabecular separation'].append(tb_sep)

            # Trabecular number
            #
            if args.trabecular_number == '3d':
                results['Trabecular number'].append(1 / (tb_sep + tb_th))
            elif args.trabecular_number == 'plate':
                results['Trabecular number'].append(bvtv / tb_th)
            elif args.trabecular_number == 'rod':
                results['Trabecular number'].append(np.sqrt((4 / np.pi) * bvtv) / tb_th)
            else:
                results['Trabecular number'].append(0)

            if args.plot:
                print_orthogonal(th_map, cmap='hot', res=args.resolution[0] / 1000)

            # TODO statistics against ground truth

            # Pearson correlation

        dur = time() - start
        completed = strftime(f'%Y_%m_%d_%H_%M')
        print(f'Analysis completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds at time {completed}.')
