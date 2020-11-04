import cv2
import os
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from time import time, strftime
import pandas as pd

import numpy as np
import scipy.ndimage as ndi
import argparse

from BoneEnhance.components.utilities import load, save, print_orthogonal, otsu_threshold
from BoneEnhance.components.inference.thickness_analysis import _local_thickness


if __name__ == '__main__':
    start = time()
    base_path = Path('../../Data/predictions')
    snap = 'dios-erc-gpu_2020_10_12_09_40_33_perceptualnet_newsplit_oof'
    #filter_size = 12
    parser = argparse.ArgumentParser()
    parser.add_argument('--masks', type=Path, default=base_path / snap)
    parser.add_argument('--th_maps', type=Path, default=base_path / (snap + f'_thickness'))
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--save_h5', type=bool, default=False)
    parser.add_argument('--batch_id', type=int, default=None)
    parser.add_argument('--resolution', type=tuple, default=(50, 50))  # in µm
    #parser.add_argument('--median', type=int, default=filter_size)
    parser.add_argument('--completed', type=int, default=0)

    args = parser.parse_args()

    # Sample list
    samples = glob(str(args.masks) + '/**/*.[pb][nm][gp]', recursive=True)
    samples.sort()
    if args.batch_id is not None:
        samples = [samples[args.batch_id]]
    elif args.completed > 0:
        samples = samples[args.completed:]

    # Save paths
    args.th_maps.mkdir(exist_ok=True)
    (args.th_maps / 'visualization').mkdir(exist_ok=True)
    if args.save_h5:
        (args.th_maps / 'h5').mkdir(exist_ok=True)

    results = {'Sample': [], 'Mean thickness': [], 'Median thickness': [], 'Thickness STD': [], 'Maximum thickness': []}
    t = strftime(f'%Y_%m_%d_%H_%M')

    # Loop for samples
    for sample in samples:
        time_sample = time()
        print(f'Processing sample {sample}')

        # Load prediction
        pred = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
        sample = os.path.basename(sample)[:-4]

        if len(np.unique(pred)) != 2:
            pred, _ = otsu_threshold(pred)

        if args.plot:
            plt.imshow(pred, cmap='gray')
            plt.savefig(str(args.th_maps / 'visualization' / (sample + '_input.png')))
            plt.close()

        # Median filter
        #pred = ndi.median_filter(pred, size=args.median)
        if args.plot:
            plt.imshow(pred, cmap='gray')
            plt.savefig(str(args.th_maps / 'visualization' / (sample + '_median.png')))
            plt.close()

        # Thickness analysis
        # Create array of correct size
        th_map = _local_thickness(pred, mode=None, spacing_mm=args.resolution, stack_axis=1)

        if args.plot:
            plt.imshow(th_map, cmap='hot')
            plt.savefig(str(args.th_maps / 'visualization' / (sample + '_th_map.png')))
            plt.close()

            plt.hist(x=th_map[np.nonzero(th_map)].flatten(), bins=30, density=True)
            plt.xlabel('Thickness (µm)')
            plt.savefig(str(args.th_maps / 'visualization' / (sample + '_histogram.png')))
            plt.show()

        # Save resulting thickness map with bmp and h5py
        cv2.imwrite(str(args.th_maps / sample) + '_th_map.bmp', th_map)

        # H5PY save
        if args.save_h5:
            savepath = args.th_maps / 'h5' / (sample + '.h5')
            h5 = h5py.File(str(savepath), 'w')
            h5.create_dataset('data', data=th_map)
            h5.close()

        # Update results
        th_map = th_map[np.nonzero(th_map)].flatten()
        results['Sample'].append(sample)
        results['Mean thickness'].append(np.mean(th_map))
        results['Median thickness'].append(np.median(th_map))
        results['Maximum thickness'].append(np.max(th_map))
        results['Thickness STD'].append(np.std(th_map))

        # Save results to excel
        writer = pd.ExcelWriter(str(args.th_maps / ('Results_' + t)) + '.xlsx')
        df1 = pd.DataFrame(results)
        df1.to_excel(writer, sheet_name='Thickness analysis')
        writer.save()

        dur_sample = time() - time_sample
        print(f'Sample processed in {(dur_sample % 3600) // 60} minutes, {dur_sample % 60} seconds.')

    dur = time() - start
    completed = strftime(f'%Y_%m_%d_%H_%M')
    print(f'Analysis completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds at time {completed}.')