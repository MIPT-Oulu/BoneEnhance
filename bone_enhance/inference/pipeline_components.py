import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import gc
import os
import cv2
import h5py
import random

from time import time
from pathlib import Path
from skimage.transform import resize
from copy import deepcopy
from tqdm import tqdm
from scipy.ndimage import zoom
from skimage import measure
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from pytorch_toolbelt.inference.tiles import CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

from .tiler3d import Tiler3D, TileMerger3D, ImageSlicer
from .model_components import InferenceModel, load_models
from .thickness_analysis import _local_thickness
from ..utilities import load, save, print_orthogonal, print_images, threshold, calculate_bvtv
from deeppipeline.segmentation.evaluation.metrics import calculate_iou, calculate_dice, \
    calculate_volumetric_similarity, calculate_confusion_matrix_from_arrays as calculate_conf


def inference(inference_model, args, config, img_full, device='cuda', weight='mean', plot=False, mean=None, std=None,
              step=2, cuda=True):
    """
    Calculates inference on one image.
    """

    # Check for mean and std
    if mean is None or std is None:
        print('No mean or std given! Data will not be scaled prior inference!')

    # Input variables
    tile = list(config.training.crop_small)
    x, y, ch = img_full.shape

    # Scale mean and std to appropriate range (float instead of uint8)
    if mean.mean() > 1 or std.mean() > 1:
        mean /= 255.
        std /= 255.

    # Segmentation model does not upscale the image
    if config.training.segmentation:
        mag = 1
        out = (x, y)
        tile[0] *= config.training.magnification
        tile[1] *= config.training.magnification
    else:
        mag = config.training.magnification
        out = (x * mag, y * mag)

    # Check the number of channels
    if ch == 3 and not config.training.rgb:
        img_full = np.expand_dims(np.mean(img_full, axis=-1), axis=-1)
        ch = 1
    elif ch == 1 and config.training.rgb:
        img_full = np.repeat(img_full, 3, axis=-1)

    # Cut large image into overlapping tiles
    tiler = Tiler3D(img_full.shape, tile=tile, out=out, step=step, mag=mag, weight=weight)

    # HCW -> CHW. Optionally, do normalization here
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

    # Allocate a CUDA buffer for holding entire mask
    merger = TileMerger3D(tiler.target_shape, channels=ch, weight=tiler.weight, cuda=cuda)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops_out)), batch_size=args.bs,
                                                pin_memory=True):
        # Move tile to GPU
        if mean is not None and std is not None:
            tiles_batch = tiles_batch.float()
            for c in range(tiles_batch.size(1)):
                tiles_batch[:, c, :, :] = (((tiles_batch[:, c, :, :] / 255.) - mean[c]) / std[c])
            tiles_batch = tiles_batch.to(device)
        else:
            tiles_batch = (tiles_batch.float() / 255.).to(device)
        # Predict and move back to CPU
        pred_batch = inference_model(tiles_batch)

        # Plot
        if plot:
            for i in range(args.bs):
                if args.bs != 1 and pred_batch.shape[0] != 1:
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').transpose(0, 2, 3, 1)[i, :, :].squeeze())
                else:
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze().transpose(1, 2, 0).squeeze())
                plt.show()

        # Check for inconsistencies
        #if pred_batch.shape[2] > x_out:
        #pred_batch = pred_batch[:, :, :x_tile, :]
        #if pred_batch.shape[3] > y_out:
        #pred_batch = pred_batch[:, :, :, :y_tile]

        if not cuda:
            pred_batch = pred_batch.cpu()

        # Merge on GPU
        merger.integrate_batch(pred_batch, coords_batch)

    # Normalize accumulated mask and convert back to numpy
    merged_pred = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype('float32')
    merged_pred = tiler.crop_to_orignal_size(merged_pred).squeeze()

    # Plot
    if plot:
        for i in range(args.bs):
            if args.bs != 1:
                plt.imshow(merged_pred)
            else:
                plt.imshow(merged_pred.squeeze())
            plt.show()

    torch.cuda.empty_cache()
    gc.collect()

    if len(merged_pred.shape) == 3:
        return merged_pred[:, :, 0]
    else:
        return merged_pred


def inference_3d(inference_model, args, config, img_full, device='cuda', plot=False,
                 mean=None, std=None, step=2, cuda=True, weight='mean'):
    """
    Calculates inference on one image.
    """

    # Check for mean and std
    if mean is None or std is None:
        print('No mean or std given! Data will not be scaled prior inference!')

    # Input variables
    mag = config.training.magnification
    tile = list(config.training.crop_small)

    x, y, z, ch = img_full.shape
    out = (x * mag, y * mag, z * mag)

    # Scale mean and std to appropriate range (float instead of uint8)
    if mean.mean() > 1 or std.mean() > 1:
        mean /= 255.
        std /= 255.

    # Check the number of channels
    if ch == 3 and not config.training.rgb:
        img_full = np.expand_dims(np.mean(img_full, axis=-1), axis=-1)
        ch = 1
    elif ch == 1 and config.training.rgb:
        img_full = np.repeat(img_full, 3, axis=-1)

    # Cut large image into overlapping tiles
    tiler = Tiler3D(img_full.shape, tile=tile, out=out, step=step, mag=mag, weight=weight)

    # HWZC -> CHWZ. Optionally, do normalization here
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

    # Allocate a CUDA buffer for holding entire mask
    merger = TileMerger3D(tiler.target_shape, channels=ch, weight=tiler.weight, cuda=cuda)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops_out)), batch_size=config.training.bs,
                                                pin_memory=True):
        # Move tile to GPU
        if mean is not None and std is not None:
            tiles_batch = tiles_batch.float()
            for c in range(len(mean)):
                tiles_batch[:, c, :, :] = (((tiles_batch[:, c, :, :] / 255.) - mean[c]) / std[c])
            tiles_batch = tiles_batch.to(device)
        else:
            tiles_batch = (tiles_batch.float() / 255.).to(device)

        # Predict and move back to CPU
        pred_batch = inference_model(tiles_batch)

        # Plot
        if plot and random.uniform(0, 1) > 0.98:
            for i in range(args.bs):
                if args.bs != 1 and pred_batch.shape[0] != 1:
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32')[i, 0, 0, :, :])
                    plt.show()
                    plt.imshow(tiles_batch.cpu().detach().numpy().astype('float32')[i, 0, 0, :, :])
                else:
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze()[i, 0, 0, :, :])
                plt.show()

        # Check for inconsistencies
        #if pred_batch.shape[2] > x_out:
        #pred_batch = pred_batch[:, :, :x_tile, :]
        #if pred_batch.shape[3] > y_out:
        #pred_batch = pred_batch[:, :, :, :y_tile]

        if not cuda:
            pred_batch = pred_batch.cpu()

        # Merge on GPU
        merger.integrate_batch(pred_batch, coords_batch)

    # Normalize accumulated mask and convert back to numpy
    merged_pred = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype('float32')
    merged_pred = tiler.crop_to_orignal_size(merged_pred)

    # Plot
    if plot:
        for i in range(args.bs):
            if args.bs != 1:
                plt.imshow(merged_pred)
            else:
                plt.imshow(merged_pred.squeeze())
            plt.show()

    torch.cuda.empty_cache()
    gc.collect()

    return merged_pred[:, :, :, 0]


def inference_runner_oof(args, config, split_config, device, plot=False, verbose=False):
    """
    Runs inference on a dataset.
    :param args: Training arguments (paths, etc.)
    :param config:
    :param split_config:
    :param device:
    :param plot:
    :param weight:
    :return:
    """
    # Timing
    start_inf = time()

    # Inference arguments
    if args.save_dir is None:
        args.save_dir = args.data_location / 'predictions_oof'
    args.save_dir.mkdir(exist_ok=True)
    sigma = 0.5  # Antialiasing filter for downscaling

    # Tiling weight
    if config.inference.weight is not None:
        weight = config.inference.weight
    else:
        weight = 'mean'
    # Tile step
    if config.inference.step is not None:
        step = config.inference.step
    else:
        step = 2

    # Create save directories
    save_dir = args.save_dir / str(config['training']['snapshot'] + '_oof')
    save_dir.mkdir(exist_ok=True)
    (save_dir / 'visualizations').mkdir(exist_ok=True)

    # Load models
    crop = config.training.crop_small
    ds = not config.training.crossmodality
    mag = config.training.magnification
    if ds:
        mean_std_path = args.snapshots_dir / f"mean_std_{crop}_ds.pth"
    else:
        mean_std_path = args.snapshots_dir / f"mean_std_{crop}_cm.pth"

    ms = torch.load(mean_std_path)
    mean, std = ms['mean'], ms['std']

    # List the models
    model_list = load_models(str(args.snapshots_dir / config.training.snapshot), config, n_gpus=args.gpus)
    print(f'Found {len(model_list)} models.')

    # Loop for all images
    for fold in range(len(model_list)):

        # List validation images
        if ds:
            validation_files = split_config[f'fold_{fold}']['eval'].target_fname.values
        else:
            validation_files = split_config[f'fold_{fold}']['eval'].fname.values

        # Model corresponding to the validation fold images
        model = InferenceModel([model_list[fold]]).to(device)
        model.eval()

        for sample in tqdm(validation_files, desc=f'Running inference for fold {fold}', disable=not verbose):

            # Correct name for cluster experiments
            if sample.parts[1] == 'run' and str(sample).endswith('.h5'):
                # 3D -> Samples are in the same folder
                sample = args.data_location / sample.parent.name / sample.name
            elif sample.parts[1] == 'run':
                # 2D -> Sample folder included
                sample = args.data_location / sample.parent.parent.name / sample.parent.name / sample.name


            ### Replace target with input ### # TODO
            sample = Path(str(sample).replace('target', 'input'))
            ds = False

            # Do not calculate inference for data copies
            if 'copy' in str(sample):
                continue

            # Load image stacks
            if str(sample).endswith('.h5'):
                with h5py.File(str(sample), 'r') as f:
                    data = f['data'][:]

                # Resize target with the given magnification to provide the input image
                if ds:
                    orig_size = data.shape
                    factor = (data.shape[0] // mag, data.shape[1] // mag, data.shape[2] // mag)
                    data = resize(data.astype('float64'), factor, order=0, anti_aliasing=True, preserve_range=True,
                                  anti_aliasing_sigma=sigma)
                else:
                    orig_size = (data.shape[0] * mag, data.shape[1] * mag, data.shape[2] * mag)

                if config.training.segmentation:
                    data = resize(data.astype('float64'), orig_size, order=3, preserve_range=True)

            else:
                data = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)

                if ds:
                    orig_size = data.shape
                    # Get the downscaled input image
                    k = 5

                    # Antialiasing and downscaling to input size
                    new_size = (data.shape[1] // mag, data.shape[0] // mag)
                    data = cv2.resize(cv2.GaussianBlur(data, ksize=(k, k), sigmaX=0), new_size)
                else:
                    orig_size = (data.shape[1] * mag, data.shape[0] * mag)

                if config.training.segmentation:
                    data = cv2.resize(data, orig_size, interpolation=cv2.INTER_CUBIC)

            # 3-channel or 1-channel
            if config.training.rgb:
                data = np.stack((data,) * 3, axis=-1)
            else:
                data = np.expand_dims(data, axis=-1)

            if len(data.shape) == 4 and plot:
                print_orthogonal(data[:, :, :, 0], invert=True, res=0.2, title='Input', cbar=True,
                                 savepath=str(save_dir / 'visualizations' / (str(sample.stem) + '_input.png')),
                                 scale_factor=1000)

            # Loop for image slices
            # 1st orientation
            with torch.no_grad():  # Do not update gradients
                if len(data.shape) == 4:
                    data = inference_3d(model, args, config, data, step=step, mean=mean, std=std, plot=plot,
                                        weight=weight)
                else:
                    data = inference(model, args, config, data, step=step, mean=mean, std=std, plot=plot,
                                     weight=weight)

            # Scale the dynamic range
            #data -= np.min(data)
            #data /= np.max(data)

            # Convert to uint8
            data = (data * 255).astype('uint8')

            # Save predicted full mask
            if len(data.shape) == 3:
                (save_dir / sample.stem).mkdir(exist_ok=True)
                save(str(save_dir / sample.stem), str(sample.stem), data, dtype='.png', verbose=False)

                if plot:
                    print_orthogonal(data, invert=True, res=0.2 / 4, title='Output', cbar=True,
                                     savepath=str(save_dir / 'visualizations' / (str(sample.stem) + '_prediction.png')),
                                     scale_factor=1000)
            else:
                (save_dir / sample.parent.stem).mkdir(exist_ok=True)
                cv2.imwrite(str(save_dir / sample.parent.stem / sample.name), data)

            # Free memory
            torch.cuda.empty_cache()
            gc.collect()

    dur_inf = time() - start_inf
    print(f'Inference completed in {(dur_inf % 3600) // 60} minutes, {dur_inf % 60} seconds.')
    return save_dir


def evaluation_runner(args, config, save_dir, use_bvtv=True, suffix='_3d'):
    start_eval = time()

    # Evaluation arguments
    args.target_path = args.data_location / f'target{suffix}'
    if args.save_dir is None:
        args.save_dir = args.data_location / 'predictions_oof'
    if args.eval_dir is None:
        args.eval_dir = args.data_location / 'evaluation_oof'
    args.eval_dir.mkdir(exist_ok=True)

    # Snapshots to be evaluated
    if type(save_dir) != list:
        save_dir = [save_dir]

    # Iterate through snapshots
    for snap in save_dir:

        # Initialize results
        if use_bvtv:
            targets = {'Sample': [], 'Trabecular thickness': [], 'Trabecular separation': [], 'BVTV': [], 'Trabecular number': []}
            results = {'Sample': [], 'MSE': [], 'PSNR': [], 'SSIM': [],
                       'Trabecular thickness': [], 'Trabecular separation': [], 'BVTV': [], 'Trabecular number': []}
        else:
            results = {'Sample': [], 'MSE': [], 'PSNR': [], 'SSIM': []}

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

        # Loop for samples
        for idx, sample in enumerate(samples):
            try:
                # Load image stacks
                if samples_target[idx].endswith('.h5'):
                    with h5py.File(str(args.target_path / samples_target[idx]), 'r') as f:
                        target = f['data'][:]
                else:
                    target, files = load(str(args.target_path / samples_target[idx]), rgb=False, axis=(1, 2, 0))

                pred, files_pred = load(str(args.save_dir / snap.name / sample), axis=(1, 2, 0), rgb=False)

                # Crop in case of inconsistency
                crop = min(pred.shape, target.shape)
                target = target[:crop[0], :crop[1], :crop[2]]
                pred = pred[:crop[0], :crop[1], :crop[2]].squeeze()

                # Evaluate metrics
                mse = mean_squared_error(target / 255., pred / 255.)
                psnr = peak_signal_noise_ratio(target / 255., pred / 255.)
                ssim = structural_similarity(target / 255., pred / 255.)

                print(f'Sample {sample}: MSE = {mse}, PSNR = {psnr}, SSIM = {ssim}')

                # Morphometric analysis
                if use_bvtv:
                    results = morphometric_analysis(pred, results)
                    targets = morphometric_analysis(targets, targets)

                # Update results
                results['Sample'].append(sample)
                results['MSE'].append(mse)
                results['PSNR'].append(psnr)
                results['SSIM'].append(ssim)

            except (AttributeError, ValueError):
                print(f'Sample {sample} failing. Skipping to next one.')
                continue

        # Add average value
        results['Sample'].append('Average values')
        results['MSE'].append(np.average(results['MSE']))
        results['PSNR'].append(np.average(results['PSNR']))
        results['SSIM'].append(np.average(results['SSIM']))

        if use_bvtv:
            results['Trabecular thickness'].append(np.average(results['Trabecular thickness']))
            results['BVTV'].append(np.average(results['BVTV']))
            results['Trabecular separation'].append(np.average(results['Trabecular separation']))
            results['Trabecular number'].append(np.average(results['Trabecular number']))

        # Write to excel
        metric_value = str(round(results['SSIM'][-1], 3)).replace('.', '-')
        writer = pd.ExcelWriter(str(args.eval_dir / ('metrics_SSIM_' + metric_value + '_' + str(snap.name))) + '.xlsx')
        df1 = pd.DataFrame(results)
        df1.to_excel(writer, sheet_name='Metrics')

        if use_bvtv:
            df2 = pd.DataFrame(targets)
            df2.to_excel(writer, sheet_name='Targets')
        writer.save()

        print(f'Metrics evaluated in {(time() - start_eval) // 60} minutes, {(time() - start_eval) % 60} seconds.')


def largest_object(input_mask, area_limit=None):
    """
    Keeps the largest connected component of a binary segmentation mask.

    If area_limit is given, all disconnected bone_enhance < area_limit are discarded.
    """

    output_mask = np.zeros(input_mask.shape, dtype=np.uint8)

    # Label connected bone_enhance
    binary_img = input_mask.astype(np.bool)
    blobs = measure.label(binary_img, connectivity=1)

    # Measure area
    proportions = measure.regionprops(blobs)

    if not proportions:
        print('No mask detected! Returning original mask')
        return input_mask

    area = [ele.area for ele in proportions]

    if area_limit is not None:

        for blob_ind, blob in enumerate(tqdm(area)):
            if blob > area_limit:
                label = proportions[blob_ind].label
                output_mask[blobs == label] = 255

    else:
        largest_blob_ind = np.argmax(area)
        largest_blob_label = proportions[largest_blob_ind].label

        output_mask[blobs == largest_blob_label] = 255

    return output_mask


def morphometric_analysis(pred, results, mode='med2d_dist3d_lth3d', resolution=(50, 50, 50), max_th=None, tb_n='rod'):
    #                       #
    # Morphometric analysis #
    #                       #

    voi = np.ones(pred.shape)

    # Thickness map
    th_map = _local_thickness(pred, mode=mode, spacing_mm=resolution, stack_axis=1,
                              thickness_max_mm=max_th, verbose=False)
    # Bone volume fraction
    bvtv = calculate_bvtv(pred, voi)

    # Update results
    th_map = th_map[np.nonzero(th_map)].flatten()
    tb_th = np.mean(th_map)
    results['Trabecular thickness'].append(tb_th)
    results['BVTV'].append(bvtv)

    # Separation map
    pred = np.logical_and(np.invert(pred), voi).astype(np.uint8) * 255
    th_map = _local_thickness(pred, mode=mode, spacing_mm=resolution, stack_axis=1,
                              thickness_max_mm=max_th, verbose=False)
    th_map = th_map[np.nonzero(th_map)].flatten()
    tb_sep = np.mean(th_map)
    results['Trabecular separation'].append(tb_sep)

    # Trabecular number
    if tb_n == '3d':  # 3D model
        results['Trabecular number'].append(1 / (tb_sep + tb_th))
    elif tb_n == 'plate':  # 2D plate model
        results['Trabecular number'].append(bvtv / tb_th)
    elif tb_n == 'rod':  # 2D cylinder rod model
        results['Trabecular number'].append(np.sqrt((4 / np.pi) * bvtv) / tb_th)
    else:  # Append 0 for compatibility
        results['Trabecular number'].append(0)

    return results