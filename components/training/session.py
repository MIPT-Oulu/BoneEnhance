import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import torch
import torch.nn as nn
import dill
import cv2
import os
from pathlib import Path
from random import uniform
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collagen.data import DataProvider, ItemLoader
from collagen.core.utils import auto_detect_device
from collagen.callbacks import RunningAverageMeter, ModelSaver, ImagePairVisualizer, RandomImageVisualizer, \
    SimpleLRScheduler, ScalarMeterLogger
from collagen.losses.segmentation import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss
from collagen.losses.superresolution import PSNRLoss

from BoneEnhance.components.transforms.main import train_test_transforms
from BoneEnhance.components.training.models import EnhanceNet, EncoderDecoder


def init_experiment():
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', type=Path, default='../../Data')
    parser.add_argument('--workdir', type=Path, default='../../Workdir/')
    parser.add_argument('--experiment', type=Path, default='../experiments/run')
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
        snapshot_name = time.strftime(f'{socket.gethostname()}_%Y_%m_%d_%H_%M_%S')
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


def init_callbacks(fold_id, config, snapshots_dir, snapshot_name, model, optimizer, mean, std):
    # Snapshot directory
    current_snapshot_dir = snapshots_dir / snapshot_name
    crop = config.training.crop_small
    log_dir = current_snapshot_dir / f"fold_{fold_id}_log"

    # Tensorboard
    writer = SummaryWriter(comment='BoneEnhance', log_dir=log_dir, flush_secs=15, max_queue=1)
    prefix = f"{crop[0]}x{crop[1]}_fold_{fold_id}"

    # Callbacks
    train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                 #RandomImageVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std),
                 ScalarMeterLogger(writer, comment='training', log_dir=str(log_dir)))

    val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
               #ImagePairVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std),
               RandomImageVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std),
               ModelSaver(metric_names='eval/loss',
                          prefix=prefix,
                          save_dir=str(current_snapshot_dir),
                          conditions='min', model=model),
               # Reduce LR on plateau
               SimpleLRScheduler('eval/loss', ReduceLROnPlateau(optimizer,
                                                                patience=int(config.training.patience),
                                                                factor=float(config.training.factor),
                                                                eps=float(config.training.eps))),
               ScalarMeterLogger(writer=writer, comment='validation', log_dir=log_dir))

    return train_cbs, val_cbs


def init_loss(config, device='cuda'):
    loss = config.training.loss
    available_losses = {
        'mse': nn.MSELoss(),
        'L1': nn.L1Loss(),
        'psnr': PSNRLoss(),
        # Segmentation losses
        'bce': BCEWithLogitsLoss2d(),
        'jaccard': SoftJaccardLoss(use_log=config.training.log_jaccard),
        'combined': CombinedLoss([BCEWithLogitsLoss2d(), SoftJaccardLoss(use_log=config.training.log_jaccard)])
    }

    return available_losses[loss].to(device)


def init_model(config, device='cuda', gpus=1):
    config.model.magnification = config.training.magnification
    architecture = config.training.architecture

    available_models = {
        'encoderdecoder': EncoderDecoder(**config['model']),
        'cnn': EnhanceNet(config.training.crop_small, config.training.magnification)
    }

    if gpus > 1:
        model = nn.DataParallel(available_models[architecture])
    else:
        model = available_models[architecture]

    return model.to(device)


def create_data_provider(args, config, parser, metadata, mean, std):
    # Compile ItemLoaders
    item_loaders = dict()
    for stage in ['train', 'val']:
        item_loaders[f'loader_{stage}'] = ItemLoader(meta_data=metadata[stage],
                                                     transform=train_test_transforms(config, mean, std)[stage],
                                                     parse_item_cb=parser,
                                                     batch_size=config.training.bs, num_workers=args.num_threads,
                                                     shuffle=True if stage == "train" else False)

    return DataProvider(item_loaders)


def parse_grayscale(root, entry, transform, data_key, target_key, debug=False, args=None):
    # Read image and target
    img = cv2.imread(str(entry.fname), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]

    target = cv2.imread(str(entry.target_fname), -1)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
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

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.99:
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(np.asarray(img.permute(1, 2, 0)), cmap='gray')
        plt.colorbar(im)
        plt.title('Input')

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(np.asarray(target.permute(1, 2, 0)), cmap='gray')
        plt.colorbar(im2)
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
