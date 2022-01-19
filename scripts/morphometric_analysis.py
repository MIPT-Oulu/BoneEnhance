"""
Calculate morphometric analysis, including the full volume.
"""

import os
import h5py
from pathlib import Path
from openpyxl import load_workbook
from time import time, strftime
import pandas as pd
from tqdm import tqdm

import numpy as np
import argparse

from bone_enhance.utilities import load, print_orthogonal, threshold, calculate_bvtv
from bone_enhance.inference.thickness_analysis import _local_thickness


if __name__ == '__main__':
    start = time()

    # Prediction path
    path = Path('../../Data/target_1176_HR')
    t = strftime(f'%Y_%m_%d_%H_%M')
    savepath = f'../../Data/evaluation_oof_wacv/Results_target{t}.xlsx'
    snaps = os.listdir(path)
    snaps = [snap for snap in snaps if os.path.isdir(os.path.join(path, snap))]
    snaps = [str(path)]

    for snap in snaps:
        # Remove timestamp from snapshot
        if snap[:2] == '2D' or snap[:2] == '3D':
            snap_short = snap
        else:
            snap_short = snap.split('_', 6)[-1]

        # Arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--save', type=Path, default=path.parent / 'masks_wacv_new' / snap_short)
        #parser.add_argument('--preds', type=Path, default=path / snap)
        parser.add_argument('--preds', type=Path, default=path)
        parser.add_argument('--final_results', type=Path, default='../../Data/final_results.csv')
        parser.add_argument('--plot', type=bool, default=False)
        parser.add_argument('--resolution', type=tuple, default=(50, 50, 50))  # in µm
        parser.add_argument('--mode', type=str, choices=['med2d_dist3d_lth3d', 'stacked_2d', 'med2d_dist2d_lth3d'],
                            default='med2d_dist3d_lth3d')
        parser.add_argument('--trabecular_number', type=str, choices=['3d', 'plate', 'rod'], default='rod')
        parser.add_argument('--max_th', type=float, default=None)  # in µm

        args = parser.parse_args()

        # Sample list
        samples = os.listdir(str(args.preds))
        samples.sort()
        if 'visualizations' in samples:
            samples.remove('visualizations')

        # Save paths
        args.save.parent.mkdir(exist_ok=True)
        if args.plot:
            args.save.mkdir(exist_ok=True)
            (args.save / 'visualization').mkdir(exist_ok=True)

        # Table for results
        results = {'Sample': [],
                   'Trabecular thickness': [], 'Trabecular separation': [], 'BVTV': [], 'Trabecular number': []}

        # Loop for samples
        for idx in tqdm(range(len(samples)), desc=f'Processing snapshot {snap_short}'):
            time_sample = time()
            sample = samples[idx]

            # Load image stacks
            if samples[idx].endswith('.h5'):
                with h5py.File(str(args.preds / samples[idx]), 'r') as f:
                    pred = f['data'][:]
            else:
                pred, _ = load(str(args.preds / samples[idx]), rgb=False, axis=(1, 2, 0))
            # Analysis conducted on the full volume
            voi = np.ones(pred.shape)

            # Binarize
            if len(np.unique(pred)) != 2:
                pred, _ = threshold(pred)

            if args.plot:
                print_orthogonal(pred, savepath=str(args.save / 'visualization' / (sample + '_pred.png')), res=50 / 1000)

            #                       #
            # Morphometric analysis #
            #                       #

            # Thickness map
            th_map = _local_thickness(pred, mode=args.mode, spacing_mm=args.resolution, stack_axis=1,
                                      thickness_max_mm=args.max_th, verbose=False)
            if args.plot:
                print_orthogonal(th_map, cmap='hot', res=args.resolution[0] / 1000)

            # Bone volume fraction
            bvtv = calculate_bvtv(pred, voi)

            # Update results
            th_map = th_map[np.nonzero(th_map)].flatten()
            tb_th = np.mean(th_map)
            results['Sample'].append(sample)
            results['Trabecular thickness'].append(tb_th)
            results['BVTV'].append(bvtv)

            # Separation map
            pred = np.invert(pred)
            th_map = _local_thickness(pred, mode=args.mode, spacing_mm=args.resolution, stack_axis=1,
                                      thickness_max_mm=args.max_th, verbose=False)
            th_map = th_map[np.nonzero(th_map)].flatten()
            tb_sep = np.mean(th_map)
            results['Trabecular separation'].append(tb_sep)

            # Trabecular number
            if args.trabecular_number == '3d':  # 3D model
                results['Trabecular number'].append(1 / (tb_sep + tb_th))
            elif args.trabecular_number == 'plate':  # 2D plate model
                results['Trabecular number'].append(bvtv / tb_th)
            elif args.trabecular_number == 'rod':  # 2D cylinder rod model
                results['Trabecular number'].append(np.sqrt((4 / np.pi) * bvtv) / tb_th)
            else:  # Append 0 for compatibility
                results['Trabecular number'].append(0)

        #               #
        # Statistics    #
        #               #

        # Remove NaNs
        results['Trabecular separation'] = list(np.nan_to_num(results['Trabecular separation']))
        results['Trabecular thickness'] = list(np.nan_to_num(results['Trabecular thickness']))
        results['Trabecular number'] = list(np.nan_to_num(results['Trabecular number']))

        # Save morphometric results to excel
        # Load existing file
        if os.path.isfile(savepath):
            book = load_workbook(savepath)
            writer = pd.ExcelWriter(savepath, engine='openpyxl', mode='a')
            writer.book = book
        else:
            writer = pd.ExcelWriter(savepath, engine='openpyxl')
        # Append new results
        df1 = pd.DataFrame(results)
        df1.to_excel(writer, sheet_name=snap_short)
        writer.save()

    dur = time() - start
    completed = strftime(f'%Y_%m_%d_%H_%M')
    print(f'Analysis completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds at time {completed}.')
