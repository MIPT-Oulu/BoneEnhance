import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import gc
import os
import cv2

from time import time
from copy import deepcopy
from tqdm import tqdm
from skimage import measure
from torch.utils.data import DataLoader
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

from BoneEnhance.components.inference.tiler3d import Tiler3D, CudaTileMerger3D
from BoneEnhance.components.inference.model_components import InferenceModel, load_models
from BoneEnhance.components.utilities.main import load, print_orthogonal, print_images
from deeppipeline.segmentation.evaluation.metrics import calculate_iou, calculate_dice, \
    calculate_volumetric_similarity, calculate_confusion_matrix_from_arrays as calculate_conf


def inference(inference_model, args, config, img_full, device='cuda', weight='mean', plot=False, mean=None, std=None):
    """
    Calculates inference on one image.
    """

    mag = config.training.magnification
    input_x = config['training']['crop_small'][0]
    input_y = config['training']['crop_small'][1]
    x, y, ch = img_full.shape
    x_out, y_out = x * mag, y * mag

    # Cut large image into overlapping tiles
    tiler = ImageSlicer(img_full.shape, tile_size=(input_x, input_y),
                        tile_step=(input_x // 2, input_y // 2), weight=weight)
                        #tile_step=(input_x, input_y), weight=weight)

    x_tile = np.min((input_x * mag, x_out))
    y_tile = np.min((input_y * mag, y_out))
    tiler_out = ImageSlicer((x_out, y_out, ch), tile_size=(x_tile, y_tile),
                            tile_step=(x_tile // 2, y_tile // 2), weight=weight)
                            #tile_step=(x_tile, y_tile), weight=weight)

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
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').transpose(0, 2, 3, 1)[i, :, :])
                else:
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze().transpose(1, 2, 0))
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


def inference_3d(inference_model, args, config, img_full, device='cuda', weight='mean', plot=False,
                 mean=None, std=None, step=2, cuda=True):
    """
    Calculates inference on one image.
    """

    mag = config.training.magnification
    tile = list(config['training']['crop_small'])

    x, y, z, ch = img_full.shape
    out = (x * mag, y * mag, z * mag)

    # Cut large image into overlapping tiles
    tiler = Tiler3D(img_full.shape, tile=tile, out=out, step=step, mag=mag)

    #tiler_out = ImageSlicer((out[0], out[1], out[2], ch), tile_size=tile,
    #                        tile_step=step, weight=weight)

    # HCW -> CHW. Optionally, do normalization here
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

    # Allocate a CUDA buffer for holding entire mask
    if cuda:
        merger = CudaTileMerger3D(tiler.target_shape, channels=ch, weight=tiler.weight)
    else:
        tile_list = None

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops_out)), batch_size=config['training']['bs'],
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
        if config.training.rgb:
            pred_batch = inference_model(tiles_batch)
        else:
            pred_batch = inference_model(tiles_batch[:, 0, :, :].unsqueeze(1))

        # Plot
        if plot:
            for i in range(args.bs):
                if args.bs != 1 and pred_batch.shape[0] != 1:
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').transpose(0, 2, 3, 1)[i, :, :])
                else:
                    plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze().transpose(1, 2, 0))
                plt.show()

        # Check for inconsistencies
        #if pred_batch.shape[2] > x_out:
        #pred_batch = pred_batch[:, :, :x_tile, :]
        #if pred_batch.shape[3] > y_out:
        #pred_batch = pred_batch[:, :, :, :y_tile]

        # Merge on GPU
        if cuda:
            merger.integrate_batch(pred_batch, coords_batch)
        else:  # List patches, merge later
            if tile_list is None:
                tile_list = pred_batch.cpu().numpy()
            else:
                tile_list = np.concatenate((tile_list, pred_batch.cpu().numpy().astype(np.float16)), axis=0)

    # Normalize accumulated mask and convert back to numpy
    if cuda:
        merged_pred = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype('float32')
        merged_pred = tiler.crop_to_orignal_size(merged_pred)
    else:
        del tiles
        merged_pred = tiler.merge(tile_list, channels=ch)

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

    return merged_pred.squeeze()[:, :, :, 0]


def inference_runner_oof(args, config, split_config, device, plot=False, weight='mean'):
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
    args.images = args.data_location / 'images'
    args.save_dir = args.data_location / 'predictions'
    threshold = config['inference']['threshold']

    # Create save directories
    args.save_dir.mkdir(exist_ok=True)
    if type(threshold) is list:
        len_th = len(threshold)
        save_dir = []
        for th in threshold:
            save_dir.append(args.save_dir / str(config['training']['snapshot'] + f'_{th}_oof'))
            save_dir[-1].mkdir(exist_ok=True)
    else:
        len_th = 1
        save_dir = args.save_dir / str(config['training']['snapshot'] + '_oof')
        save_dir.mkdir(exist_ok=True)

    # Load models
    unet = config['model']['decoder'].lower() == 'unet'
    model_list = load_models(str(args.snapshots_dir / config['training']['snapshot']), config, unet=unet, n_gpus=args.gpus)
    print(f'Found {len(model_list)} models.')

    # Loop for all images
    for fold in range(len(model_list)):
        # List validation images
        validation_files = split_config[f'fold_{fold}']['val'].fname.values

        # Model without validation fold
        model = InferenceModel([model_list[fold]]).to(device)
        model.eval()

        for file in tqdm(validation_files, desc=f'Running inference for fold {fold}'):
            img_full = cv2.imread(str(file))

            with torch.no_grad():  # Do not update gradients
                merged_mask = inference(model, args, config, img_full, weight=weight, device=device, plot=plot)

            # Copy list of thresholds
            th = deepcopy(threshold)
            for ind in range(len_th):
                # Save multiple thresholds
                if len_th > 1:
                    save_dir = save_dir[ind]
                    threshold = th[ind]

                mask_final = (merged_mask >= threshold).astype('uint8') * 255

                # Save largest mask
                largest_mask = largest_object(mask_final)

                if config['training']['experiment'] == '3D':
                    # When saving 3D stacks, file structure should be preserved
                    (save_dir / file.parent.stem).mkdir(exist_ok=True)
                    cv2.imwrite(str(save_dir / file.parent.stem / file.stem) + '.bmp', largest_mask)
                else:
                    # Otherwise, save images directly
                    cv2.imwrite(str(save_dir / file.stem) + '.bmp', largest_mask)

            # Free memory
            torch.cuda.empty_cache()
            gc.collect()

    dur_inf = time() - start_inf
    print(f'Inference completed in {(dur_inf % 3600) // 60} minutes, {dur_inf % 60} seconds.')
    return save_dir


def evaluation_runner(args, config, save_dir):
    start_eval = time()

    # Evaluation arguments
    args.image_path = args.data_location / 'images'
    args.mask_path = args.data_location / 'masks'
    args.pred_path = args.data_location / 'predictions'
    args.save_dir = args.data_location / 'evaluation'
    args.save_dir.mkdir(exist_ok=True)
    args.n_labels = 2

    # Snapshots to be evaluated
    if type(save_dir) != list:
        save_dir = [save_dir]

    # Iterate through snapshots
    for snap in save_dir:

        # Initialize results
        results = {'Sample': [], 'Dice': [], 'IoU': [], 'Similarity': []}

        # Loop for samples
        (args.save_dir / ('visualizations_' + snap.name)).mkdir(exist_ok=True)
        samples = os.listdir(str(args.mask_path))
        samples.sort()
        try:
            for idx, sample in enumerate(samples):

                print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample}')

                # Load image stacks
                if config['training']['experiment'] == '3D':
                    mask, files_mask = load(str(args.mask_path / sample), axis=(0, 2, 1), rgb=False, n_jobs=args.n_threads)

                    pred, files_pred = load(str(args.pred_path / snap.name / sample), axis=(0, 2, 1), rgb=False,
                                            n_jobs=args.n_threads)
                    data, files_data = load(str(args.image_path / sample), axis=(0, 2, 1), rgb=False, n_jobs=args.n_threads)

                    # Crop in case of inconsistency
                    crop = min(pred.shape, mask.shape)
                    mask = mask[:crop[0], :crop[1], :crop[2]]
                    pred = pred[:crop[0], :crop[1], :crop[2]]

                else:
                    data = cv2.imread(str(args.image_path / sample))
                    mask = cv2.imread(str(args.mask_path / sample), cv2.IMREAD_GRAYSCALE)
                    pred = cv2.imread(str(args.pred_path / snap.name / sample), cv2.IMREAD_GRAYSCALE)
                    if pred is None:
                        sample = sample[:-4] + '.bmp'
                        pred = cv2.imread(str(args.pred_path / snap.name / sample), cv2.IMREAD_GRAYSCALE)
                    elif mask is None:
                        mask = cv2.imread(str(args.mask_path / sample), cv2.IMREAD_GRAYSCALE)

                    # Crop in case of inconsistency
                    crop = min(pred.shape, mask.shape)
                    mask = mask[:crop[0], :crop[1]]
                    pred = pred[:crop[0], :crop[1]]

                # Evaluate metrics
                conf_matrix = calculate_conf(pred.astype(np.bool), mask.astype(np.bool), args.n_labels)
                dice = calculate_dice(conf_matrix)[1]
                iou = calculate_iou(conf_matrix)[1]
                sim = calculate_volumetric_similarity(conf_matrix)[1]

                print(f'Sample {sample}: dice = {dice}, IoU = {iou}, similarity = {sim}')

                # Save predicted full mask
                if config['training']['experiment'] == '3D':
                    print_orthogonal(data, invert=False, res=3.2, cbar=True,
                                     savepath=str(args.save_dir / ('visualizations_' + snap.name) / (sample + '_input.png')),
                                     scale_factor=1500)
                    print_orthogonal(data, mask=mask, invert=False, res=3.2, cbar=True,
                                     savepath=str(args.save_dir / ('visualizations_' + snap.name) / (sample + '_reference.png')),
                                     scale_factor=1500)
                    print_orthogonal(data, mask=pred, invert=False, res=3.2, cbar=True,
                                     savepath=str(
                                         args.save_dir / ('visualizations_' + snap.name) / (sample + '_prediction.png')),
                                     scale_factor=1500)

                # Update results
                results['Sample'].append(sample)
                results['Dice'].append(dice)
                results['IoU'].append(iou)
                results['Similarity'].append(sim)

        except AttributeError:
            print(f'Sample {sample} failing. Skipping to next one.')
            continue

        # Add average value to
        results['Sample'].append('Average values')
        results['Dice'].append(np.average(results['Dice']))
        results['IoU'].append(np.average(results['IoU']))
        results['Similarity'].append(np.average(results['Similarity']))

        # Write to excel
        writer = pd.ExcelWriter(str(args.save_dir / ('metrics_' + str(snap.name))) + '.xlsx')
        df1 = pd.DataFrame(results)
        df1.to_excel(writer, sheet_name='Metrics')
        writer.save()

        print(f'Metrics evaluated in {(time() - start_eval) // 60} minutes, {(time() - start_eval) % 60} seconds.')


def largest_object(input_mask, area_limit=None):
    """
    Keeps the largest connected component of a binary segmentation mask.

    If area_limit is given, all disconnected components < area_limit are discarded.
    """

    output_mask = np.zeros(input_mask.shape, dtype=np.uint8)

    # Label connected components
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
