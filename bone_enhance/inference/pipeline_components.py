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
from ..utilities import load, save, print_orthogonal, print_images, threshold, calculate_bvtv
from deeppipeline.segmentation.evaluation.metrics import calculate_iou, calculate_dice, \
    calculate_volumetric_similarity, calculate_confusion_matrix_from_arrays as calculate_conf


def inference(inference_model, args, config, img_full, device='cuda', weight='mean', plot=False, mean=None, std=None,
              tile=2):
    """
    Calculates inference on one image.
    """

    # Check for mean and std
    if mean is None or std is None:
        print('No mean or std given! Data will not be scaled prior inference!')

    # Input variables
    input_x = config.training.crop_small[0]
    input_y = config.training.crop_small[1]
    x, y, ch = img_full.shape

    # Segmentation model does not upscale the image
    if config.training.architecture == 'encoderdecoder':
        mag = 1
        x_out, y_out = x, y
        input_x *= config.training.magnification
        input_y *= config.training.magnification
    else:
        mag = config.training.magnification
        x_out, y_out = x * mag, y * mag

    # Cut large image into overlapping tiles
    tiler = ImageSlicer(img_full.shape, tile_size=(input_x, input_y),
                        tile_step=(input_x // tile, input_y // tile), weight=weight)

    x_tile = np.min((input_x * mag, x_out))
    y_tile = np.min((input_y * mag, y_out))
    tiler_out = ImageSlicer((x_out, y_out, ch), tile_size=(x_tile, y_tile),
                            tile_step=(x_tile // tile, y_tile // tile), weight=weight)

    # HCW -> CHW. Optionally, do normalization here
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

    # Allocate a CUDA buffer for holding entire mask
    merger = CudaTileMerger(tiler_out.target_shape, channels=3, weight=tiler_out.weight)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler_out.crops)), batch_size=config['training']['bs'],
                                                pin_memory=True):
        # Move tile to GPU
        if mean is not None and std is not None:
            tiles_batch = tiles_batch.float()
            for ch in range(len(mean)):
                tiles_batch[:, ch, :, :] = ((tiles_batch[:, ch, :, :] - mean[ch]) / std[ch])
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
        pred_batch = pred_batch[:, :, :x_tile, :]
        #if pred_batch.shape[3] > y_out:
        pred_batch = pred_batch[:, :, :, :y_tile]

        # Merge on GPU
        merger.integrate_batch(pred_batch, coords_batch)

    # Normalize accumulated mask and convert back to numpy
    merged_pred = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype('float32')
    merged_pred = tiler_out.crop_to_orignal_size(merged_pred)
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

    return merged_pred.squeeze()[:, :, 0]


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
    tile = list(config['training']['crop_small'])

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
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops_out)), batch_size=config['training']['bs'],
                                                #pin_memory=True,
                                                num_workers=16):
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

            if len(data.shape) == 4:
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
                    data = inference(model, args, config, data, tile=step, mean=mean, std=std, plot=plot,
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
    args.image_path = args.data_location / 'input'
    args.target_path = args.data_location / f'target{suffix}'
    args.masks = Path('/media/dios/kaappi/Sakke/Saskatoon/Verity/Registration')
    args.pred_path = args.data_location / 'predictions_oof'
    args.save_dir = args.data_location / 'evaluation_oof'
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

        # Do not calculate BV/TV if VOI is not defined for the dataset
        if len(samples_voi) != len(samples_target):
            use_bvtv = False

        # Loop for samples
        for idx, sample in enumerate(samples):
            try:
                # Load image stacks
                if samples_target[idx].endswith('.h5'):
                    with h5py.File(str(args.target_path / samples_target[idx]), 'r') as f:
                        target = f['data'][:]
                else:
                    target, files = load(str(args.target_path / samples_target[idx]), rgb=False, axis=(1, 2, 0))

                pred, files_pred = load(str(args.pred_path / snap.name / sample), axis=(1, 2, 0), rgb=False)

                # Crop in case of inconsistency
                crop = min(pred.shape, target.shape)
                target = target[:crop[0], :crop[1], :crop[2]]
                pred = pred[:crop[0], :crop[1], :crop[2]].squeeze()

                # Evaluate metrics
                mse = mean_squared_error(target / 255., pred / 255.)
                psnr = peak_signal_noise_ratio(target / 255., pred / 255.)
                ssim = structural_similarity(target / 255., pred / 255.)

                # Binarize and calculate BVTV
                if use_bvtv:

                    # Otsu thresholding
                    if len(np.unique(pred)) != 2:
                        pred, _ = threshold(pred, method='otsu')

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

                print(f'Sample {sample}: MSE = {mse}, PSNR = {psnr}, SSIM = {ssim}, BVTV: {bvtv}')

                # Update results
                results['Sample'].append(sample)
                results['MSE'].append(mse)
                results['PSNR'].append(psnr)
                results['SSIM'].append(ssim)
                results['BVTV'].append(bvtv)

            except (AttributeError, ValueError):
                print(f'Sample {sample} failing. Skipping to next one.')
                continue

        # Add average value to
        results['Sample'].append('Average values')
        results['MSE'].append(np.average(results['MSE']))
        results['PSNR'].append(np.average(results['PSNR']))
        results['SSIM'].append(np.average(results['SSIM']))
        results['BVTV'].append(np.average(results['BVTV']))

        # Write to excel
        metric_value = str(round(results['SSIM'][-1], 3)).replace('.', '-')
        writer = pd.ExcelWriter(str(args.save_dir / ('metrics_SSIM_' + metric_value + '_' + str(snap.name))) + '.xlsx')
        df1 = pd.DataFrame(results)
        df1.to_excel(writer, sheet_name='Metrics')
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
