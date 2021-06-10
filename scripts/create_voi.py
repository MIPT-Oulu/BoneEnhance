import os
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from time import time, strftime
import pandas as pd
from tqdm import tqdm

import numpy as np
from scipy.ndimage import zoom
import argparse
from skimage.transform import resize

from bone_enhance.utilities import load, save, print_orthogonal, threshold
from bone_enhance.inference.thickness_analysis import _local_thickness


if __name__ == '__main__':
    # ******************************** 3D case ************************************
    start = time()

    # Ankle experiments
    path = '../../Workdir/wacv_experiments'
    snaps = os.listdir(path)
    snaps = [snap for snap in snaps if os.path.isdir(os.path.join(path, snap))]

    base_path = Path('../../Data/Test set (full)')
    #snap = '2021_05_28_13_52_02_2D_mse_tv_1176_seed30'
    for snap in snaps:
        #filter_size = 12
        parser = argparse.ArgumentParser()
        parser.add_argument('--masks', type=Path, default=base_path / 'trabecular_VOI')
        parser.add_argument('--save', type=Path, default=base_path / 'masks_wacv' / snap)
        parser.add_argument('--preds', type=Path, default=base_path / 'predictions_wacv' / snap)
        parser.add_argument('--plot', type=bool, default=True)
        parser.add_argument('--scale_voi', type=bool, default=False)
        parser.add_argument('--save_h5', type=bool, default=False)
        parser.add_argument('--batch_id', type=int, default=None)
        parser.add_argument('--resolution', type=tuple, default=(50, 50, 50))  # in µm
        parser.add_argument('--mode', type=str,
                            choices=['med2d_dist3d_lth3d', 'stacked_2d', 'med2d_dist2d_lth3d'],
                            default='med2d_dist3d_lth3d')
        parser.add_argument('--max_th', type=float, default=None)  # in µm
        parser.add_argument('--completed', type=int, default=0)

        args = parser.parse_args()

        # Sample list
        samples = os.listdir(args.masks)
        samples.sort()
        if 'visualization' in samples:
            samples.remove('visualization')
        if args.batch_id is not None:
            samples = [samples[args.batch_id]]
        elif args.completed > 0:
            samples = samples[args.completed:]

        # Remove unnecessary samples
        #samples.remove('33_L6TM_1_Rec')
        #samples.remove('34_R6_TL7_Rec')
        #samples.remove('34_R6_TM17_Rec')

        samples_pred = os.listdir(str(args.preds))
        samples_pred.sort()
        if 'visualizations' in samples_pred:
            samples_pred.remove('visualizations')

        # Save paths
        args.save.mkdir(exist_ok=True)
        (args.save / 'visualization').mkdir(exist_ok=True)
        if args.save_h5:
            (args.save / 'h5').mkdir(exist_ok=True)

        t = strftime(f'%Y_%m_%d_%H_%M')

        # Loop for samples
        for idx in tqdm(range(len(samples)), desc=f'Processing snapshot {snap}'):
            time_sample = time()
            sample = samples[idx]
            sample_pred = samples_pred[idx]
            #print(f'Processing sample {sample}')

            # Load full list of files
            if args.scale_voi:
                files = os.listdir(str(args.masks / sample))
                files.sort()
                newlist = []
                for file in files:
                    if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif') \
                            or file.endswith('.dcm') or file.endswith('.ima'):
                        try:
                            if file.endswith('.dcm') or file.endswith('.ima'):
                                newlist.append(file)
                                dicom = True
                                continue

                            int(file[-7:-4])

                            # Do not load files with different prefix into the stack
                            if len(newlist) != 0 and file.rsplit('_', 1)[0] != newlist[-1].rsplit('_', 1)[0]:
                                break

                            newlist.append(file)
                        except ValueError:
                            continue
                files_full = newlist[:]  # replace list


            pred, _ = load(str(args.preds / sample_pred), axis=(1, 2, 0,))
            voi, files = load(str(args.masks / sample), axis=(1, 2, 0,))

            # Rescale VOI
            if args.scale_voi:
                # Append the empty images
                limits_full = (Path(files_full[0]).stem[-8:], Path(files_full[-1]).stem[-8:])
                limits_voi = (Path(files[0]).stem[-8:], Path(files[-1]).stem[-8:])
                array_shape = (voi.shape[0], voi.shape[1], int(limits_voi[0]) - int(limits_full[0]))
                voi = np.append(np.zeros(array_shape, dtype='uint8'), voi, axis=2)
                array_shape = (voi.shape[0], voi.shape[1], int(limits_full[1]) - int(limits_voi[1]))
                voi = np.append(voi, np.zeros(array_shape, dtype='uint8'), axis=2)

                # Resize VOI
                factor = (pred.shape[0] / voi.shape[0], pred.shape[1] / voi.shape[1], pred.shape[2] / voi.shape[2])
                voi = zoom(voi, factor, order=0)


            # Fix size mismatch
            #size = np.min((voi.shape, pred.shape), axis=0)

            if len(np.unique(pred)) != 2:
                pred, _ = threshold(pred)

            if args.plot:
                print_orthogonal(pred, savepath=str(args.save / 'visualization' / (sample + '_pred.png')), res=50 / 1000)

            # Apply volume-of-interest
            pred = np.logical_and(pred, voi)

            if args.plot:
                print_orthogonal(pred, savepath=str(args.save / 'visualization' / (sample + '_voi.png')), res=50 / 1000)


            # H5PY save
            if args.save_h5:
                savepath = args.th_maps / 'h5' / (sample + '.h5')
                h5 = h5py.File(str(savepath), 'w')
                h5.create_dataset('data', data=pred)
                h5.close()
            else:
                # Save results
                save(str(args.save / sample), Path(sample).stem, pred, dtype='.bmp', verbose=False)

            #dur_sample = time() - time_sample
            #print(f'Sample processed in {(dur_sample % 3600) // 60} minutes, {dur_sample % 60} seconds.')

        dur = time() - start
        completed = strftime(f'%Y_%m_%d_%H_%M')
        print(f'Analysis completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds at time {completed}.')