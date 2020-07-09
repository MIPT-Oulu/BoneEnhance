import pathlib
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import torch
import torch.nn as nn
import dill
import json
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial

from collagen.data import DataProvider, ItemLoader
from collagen.core.utils import auto_detect_device
from collagen.callbacks.meters import RunningAverageMeter, ItemWiseBinaryJaccardDiceMeter
from collagen.callbacks.logging import ScalarMeterLogger
from collagen.callbacks import ModelSaver, ImageMaskVisualizer, SimpleLRScheduler
from collagen.losses.segmentation import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss

from solt import DataContainer

from BoneEnhance.components.transforms.main import train_test_transforms


def init_experiment():
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', type=pathlib.Path, default='../../Data')
    parser.add_argument('--workdir', type=pathlib.Path, default='../../Workdir/')
    parser.add_argument('--experiment', type=pathlib.Path, default='../experiments/run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--magnification', type=int, default=4)
    parser.add_argument('--num_threads', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=2)
    args = parser.parse_args()

    # Initialize working directories
    args.snapshots_dir = args.workdir / 'snapshots'
    args.snapshots_dir.mkdir(exist_ok=True)

    # List configuration files
    config_paths = os.listdir(str(args.experiment))
    config_paths.sort()

    # Open configuration files and add to list
    config_list = []
    for config_path in config_paths:
        if config_path[-4:] == '.yml':
            with open(args.experiment / config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                config_list.append(config)

        # Snapshot directory
        encoder = config['model']['backbone']
        decoder = config['model']['decoder']
        experiment = config['training']['experiment']
        snapshot_name = time.strftime(f'{socket.gethostname()}_%Y_%m_%d_%H_%M_%S_{experiment}_{encoder}_{decoder}')
        (args.snapshots_dir / snapshot_name).mkdir(exist_ok=True, parents=True)
        config['training']['snapshot'] = snapshot_name

        # Save the experiment parameters
        with open(args.snapshots_dir / snapshot_name / 'config.yml', 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=False)
        # Save args
        with open(args.snapshots_dir / snapshot_name / 'args.dill', 'wb') as f:
            dill.dump(args, f)

    # Seeding
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Calculation resource
    device = auto_detect_device()

    return args, config_list, device


def init_callbacks(fold_id, config, snapshots_dir, snapshot_name, model, optimizer, data_provider, mean, std):
    # Snapshot directory
    current_snapshot_dir = snapshots_dir / snapshot_name
    crop = config['training']['crop_small']
    log_dir = current_snapshot_dir / f"fold_{fold_id}_log"
    device = next(model.parameters()).device

    # Tensorboard
    writer = SummaryWriter(comment='RabbitCCS', log_dir=log_dir, flush_secs=15, max_queue=1)
    prefix = f"{crop[0]}x{crop[1]}_fold_{fold_id}"

    # Set threshold
    if 'threshold' in config['training']:  # Threshold in config file
        threshold = config['training']['threshold']
    else:  # Not given
        threshold = 0.3 if config['training']['log_jaccard'] else 0.5

    # Callbacks
    train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                 ScalarMeterLogger(writer, comment='training', log_dir=str(log_dir))
                 )

    val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
               ImageMaskVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std),
               ModelSaver(metric_names='eval/loss',
                          prefix=prefix,
                          save_dir=str(current_snapshot_dir),
                          conditions='min', model=model),
               # Reduce LR on plateau
               SimpleLRScheduler('eval/loss', ReduceLROnPlateau(optimizer,
                                                                patience=int(config['training']['patience']),
                                                                factor=float(config['training']['factor']),
                                                                eps=float(config['training']['eps']))),
               ScalarMeterLogger(writer=writer, comment='validation', log_dir=log_dir))

    return train_cbs, val_cbs


def init_loss(config, device='cuda'):
    loss = config['training']['loss']
    if loss == 'bce':
        return BCEWithLogitsLoss2d().to(device)
    elif loss == 'jaccard':
        return SoftJaccardLoss(use_log=config['training']['log_jaccard']).to(device)
    elif loss == 'mse':
        return nn.MSELoss().to(device)
    elif loss == 'combined':
        return CombinedLoss([BCEWithLogitsLoss2d(),
                            SoftJaccardLoss(use_log=config['training']['log_jaccard'])]).to(device)
    else:
        raise Exception('No compatible loss selected in experiment_config.yml! Set training->loss accordingly.')


def create_data_provider(args, config, parser, metadata, mean, std):
    # Compile ItemLoaders
    item_loaders = dict()
    for stage in ['train', 'val']:
        item_loaders[f'loader_{stage}'] = ItemLoader(meta_data=metadata[stage],
                                                     transform=train_test_transforms(config, mean, std)[stage],
                                                     parse_item_cb=parser,
                                                     batch_size=config['training']['bs'], num_workers=args.num_threads,
                                                     shuffle=True if stage == "train" else False)

    return DataProvider(item_loaders)


def parse_multi_label(x, cls, threshold=0.5):
    out = x[:, cls, :, :].unsqueeze(1).gt(threshold)
    return torch.cat((1 - out, out), dim=1).squeeze()


def parse_binary_label(x, threshold=0.5):
    out = x.gt(threshold)
    #return torch.cat((~out, out), dim=1).squeeze().float()
    return out.squeeze().float()


def parse_item_test(root, entry, transform, data_key, target_key):
    img = cv2.imread(str(entry.fname), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dc = DataContainer((img, ), 'I',  transform_settings={0: {'interpolation': 'bilinear'}})
    img = transform(dc)[0]
    #img = torch.cat([img, img, img], 0) / 255.
    img = img.permute(2, 0, 1) / 255.

    return {data_key: img}


def parse_grayscale(root, entry, transform, data_key, target_key, debug=False, args=None):
    # Read image and target
    img = cv2.imread(str(entry.fname), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]

    target = cv2.imread(str(entry.target_fname), -1)
    try:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    except:
        pass
    target[:, :, 1] = target[:, :, 0]
    target[:, :, 2] = target[:, :, 0]

    # Resize target to 4x magnification respect to input
    if args is not None:
        resize = (img.shape[1] * args.magnification, img.shape[0] * args.magnification)
        target = cv2.resize(target, resize)

    # Apply random transforms
    img, target = transform[0]((img, target))

    # Small crop for input
    img = transform[1](img)[0]
    # Large crop for target
    target = transform[2](target)[0]

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    img = img.permute(2, 0, 1) / 255.
    target = target.permute(2, 0, 1) / 255.

    # Debugging
    if debug:
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(121)
        ax1.imshow(np.asarray(img.permute(1, 2, 0)))
        plt.title('Input')

        ax2 = fig.add_subplot(122)
        ax2.imshow(np.asarray(target.permute(1, 2, 0)))
        plt.title('Target')
        plt.show()

    return {data_key: img, target_key: target}


def parse_color(root, entry, transform, data_key, target_key, debug=False):
    # Image and mask generation
    img = cv2.imread(str(entry.fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(entry.mask_fname), 0) / 255.

    if img.shape[0] != mask.shape[0]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    elif img.shape[1] != mask.shape[1]:
        mask = mask[:, :img.shape[1]]

    img, mask = transform((img, mask))
    img = img.permute(2, 0, 1) / 255.  # img.shape[0] is the color channel after permute

    # Debugging
    if debug:
        plt.imshow(np.asarray(img).transpose((1, 2, 0)))
        plt.imshow(np.asarray(mask).squeeze(), alpha=0.3)
        plt.show()

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    return {data_key: img, target_key: mask}


def save_config(path, config, args):
    """
    Alternate way to save model parameters.
    """
    with open(path + '/experiment_config.txt', 'w') as f:
        f.write(f'\nArguments file:\n')
        f.write(f'Seed: {args.seed}\n')
        f.write(f'Batch size: {args.bs}\n')
        f.write(f'N_epochs: {args.n_epochs}\n')

        f.write('Configuration file:\n\n')
        for key, val in config.items():
            f.write(f'{key}\n')
            for key2 in config[key].items():
                f.write(f'\t{key2}\n')


def save_transforms(path, config, args, mean, std):
    transforms = train_test_transforms(config, mean, std)
    # Save the experiment parameters
    with open(path / 'transforms.yaml', 'w') as f:
        yaml.dump(transforms['train_list'][1].to_yaml(), f)
