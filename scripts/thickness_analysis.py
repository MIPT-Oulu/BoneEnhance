import os
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from time import time, strftime
import pandas as pd

import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
import argparse

from bone_enhance.utilities import load, save, print_orthogonal, threshold, calculate_bvtv
from bone_enhance.inference.thickness_analysis import _local_thickness
from bone_enhance.inference import largest_object


if __name__ == '__main__':
    # ******************************** 3D case ************************************
    start = time()
    save_path = Path('../../Data/predictions_3D')
    base_path = Path('/media/dios/kaappi/Sakke/Saskatoon/Verity/Registration')
    #snap = 'dios-erc-gpu_2020_11_04_14_10_25_3D_perceptualnet_scratch_bestfold'
    snap = 'thickness_dios-erc-gpu_2020_11_04_14_10_25_3D_perceptualnet_scratch'
    ds = True
    mag = 4
    #filter_size = 12
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=Path, default='../../Data/target_mag4')
    parser.add_argument('--masks', type=Path, default=base_path)
    #parser.add_argument('--vois', type=Path, default=base_path)
    parser.add_argument('--th_maps', type=Path, default=save_path / f'thickness_{snap}')
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--save_h5', type=bool, default=False)
    parser.add_argument('--batch_id', type=int, default=None)
    parser.add_argument('--resolution', type=tuple, default=(50, 50, 50))  # in µm
    parser.add_argument('--resolution_input', type=tuple, default=(50, 50, 50))  # in µm
    #parser.add_argument('--resolution', type=tuple, default=(12.8, 12.8, 12.8))  # in µm
    parser.add_argument('--mode', type=str,
                        choices=['med2d_dist3d_lth3d', 'stacked_2d', 'med2d_dist2d_lth3d'],
                        default='med2d_dist3d_lth3d')
    parser.add_argument('--max_th', type=float, default=None)  # in µm
    parser.add_argument('--completed', type=int, default=0)

    args = parser.parse_args()

    # Sample list
    all_samples = os.listdir(args.masks)
    samples = []
    for i in range(len(all_samples)):
        if os.path.isdir(str(args.masks / all_samples[i])):
            samples.append(all_samples[i])
    samples.sort()
    if 'visualizations' in samples:
        samples.remove('visualizations')
    if args.batch_id is not None:
        samples = [samples[args.batch_id]]
    elif args.completed > 0:
        samples = samples[args.completed:]

    # Save paths
    args.th_maps.mkdir(exist_ok=True)
    (args.th_maps / 'visualization').mkdir(exist_ok=True)
    if args.save_h5:
        (args.th_maps / 'h5').mkdir(exist_ok=True)

    results = {'Sample': [],
               #'Mean thickness': [], 'Median thickness': [], 'Maximum thickness': [],
               #'Mean thickness_input': [], 'Median thickness_input': [], 'Maximum thickness_input': [],
               'BVTV_pred': [], 'BVTV_input': []}
    t = strftime(f'%Y_%m_%d_%H_%M')

    # Loop for samples
    for sample in samples:
        time_sample = time()
        print(f'Processing sample {sample}')

        # Load prediction
        try:
            pred, files = load(str(Path(save_path) / snap / str(sample + '_Rec')), axis=(1, 2, 0,))
        except FileNotFoundError:
            pred, files = load(str(Path(save_path) / snap / str(sample)), axis=(1, 2, 0,))
        if ds:
            pth = str(args.ds / (sample + '.h5'))
            if not os.path.isdir(pth):
                pth = str(args.ds / (sample + '_Rec.h5'))
            with h5py.File(pth, 'r') as f:
                input = f['data'][:]
            # Resize target with the given magnification to provide the input image
            factor = (pred.shape[0] // mag, pred.shape[1] // mag, pred.shape[2] // mag)
            input = resize(input, factor, order=0, anti_aliasing=True, preserve_range=True)
        else:
            input, _ = load(str(args.masks / sample), axis=(1, 2, 0,))
        voi_input, _ = load(str(args.masks / sample / 'ROI'), axis=(1, 2, 0,))

        # Tricubic interpolation for the CBCT data
        input = zoom(input.squeeze(), (4, 4, 4), order=3)
        # Upscale VOI (nearest-neighbor)
        voi = zoom(voi_input.squeeze(), (4, 4, 4), order=0)

        save(str(args.th_maps / 'Verity_TCI' / sample), sample, input, dtype='.bmp')
        save(str(args.th_maps / 'VOI_upscaled' / sample), sample, voi, dtype='.bmp')
        pred = pred.squeeze()

        if args.plot:
            print_orthogonal(input, savepath=str(args.th_maps / 'visualization' / (sample + '_input.png')), res=args.resolution[0]/1000)
            print_orthogonal(pred, savepath=str(args.th_maps / 'visualization' / (sample + '_pred.png')), res=args.resolution[0]/1000)

        # Otsu thresholding
        if len(np.unique(pred)) != 2:
            pred, _ = threshold(pred, method='mean')
            #pred = largest_object(pred, area_limit=12)
        if len(np.unique(input)) != 2:
            input, _ = threshold(input, method='mean')
            #input = largest_object(input, area_limit=3)

        if args.plot:
            print_orthogonal(input, savepath=str(args.th_maps / 'visualization' / (sample + '_full_mask_input.png')), res=args.resolution[0]/1000)
            print_orthogonal(pred, savepath=str(args.th_maps / 'visualization' / (sample + '_full_mask.png')), res=args.resolution[0]/1000)

        # Apply volume-of-interest
        try:
            input = np.logical_and(input, voi)
        except ValueError:
            continue

        # Downscale
        #pred = (ndi.zoom(pred, 0.25) > 126).astype(np.bool)

        # Fix size mismatch
        size = np.min((voi.shape, pred.shape), axis=0)

        # Apply volume-of-interest
        pred = np.logical_and(pred[:size[0], :size[1], :size[2]],
                              voi[:size[0], :size[1], :size[2]])

        if args.plot:
            print_orthogonal(input, savepath=str(args.th_maps / 'visualization' / (sample + '_voi_mask_input.png')), res=args.resolution[0]/1000)
            print_orthogonal(pred, savepath=str(args.th_maps / 'visualization' / (sample + '_voi_mask.png')), res=args.resolution[0]/1000)

        # Thickness analysis
        # Create array of correct size
        #th_map = _local_thickness(pred, mode=args.mode, spacing_mm=args.resolution, stack_axis=1,
        #                          thickness_max_mm=args.max_th)

        #th_map_input = _local_thickness(input, mode=args.mode, spacing_mm=args.resolution_input, stack_axis=1,
        #                                thickness_max_mm=args.max_th)

        #if args.plot:
            #print_orthogonal(th_map, savepath=str(args.th_maps / 'visualization' / (sample + '_th_map.png')),
            #                 cmap='hot', res=args.resolution[0]/1000)
            #print_orthogonal(th_map_input, savepath=str(args.th_maps / 'visualization' / (sample + '_th_map_input.png')),
            #                 cmap='hot', res=args.resolution[0]/1000)

        #plt.hist(x=th_map[np.nonzero(th_map)].flatten(), bins='auto')
        #plt.title('Predicted')
        #plt.show()
        #plt.hist(x=th_map_input[np.nonzero(th_map_input)].flatten(), bins='auto')
        #plt.title('Input')
        #plt.show()

        # Save resulting thickness map with bmp and h5py
        #save(str(args.th_maps / sample), sample, th_map, dtype='.bmp')

        # H5PY save
        if args.save_h5:
            savepath = args.th_maps / 'h5' / (sample + '.h5')
            h5 = h5py.File(str(savepath), 'w')
            h5.create_dataset('data', data=th_map)
            h5.close()

        # Calculate BVTV
        bvtv = calculate_bvtv(pred, voi)
        bvtv_input = calculate_bvtv(input, voi)

        if bvtv == 0:
            results['Sample'].append(sample)
            #results['Mean thickness'].append(0)
            #results['Median thickness'].append(0)
            #results['Maximum thickness'].append(0)
            results['BVTV_pred'].append(bvtv)
            #results['Mean thickness_input'].append(0)
            #results['Median thickness_input'].append(0)
            #results['Maximum thickness_input'].append(0)
            results['BVTV_input'].append(bvtv_input)
        else:
            # Update results
            #th_map = th_map[np.nonzero(th_map)].flatten()
            #th_map_input = th_map_input[np.nonzero(th_map_input)].flatten()
            results['Sample'].append(sample)
            #results['Mean thickness'].append(np.mean(th_map))
            #results['Median thickness'].append(np.median(th_map))
            #results['Maximum thickness'].append(np.max(th_map))
            #results['Thickness STD'].append(np.std(th_map))
            results['BVTV_pred'].append(bvtv)
            #results['Mean thickness_input'].append(np.mean(th_map_input))
            #results['Median thickness_input'].append(np.median(th_map_input))
            #results['Maximum thickness_input'].append(np.max(th_map_input))
            results['BVTV_input'].append(bvtv_input)

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