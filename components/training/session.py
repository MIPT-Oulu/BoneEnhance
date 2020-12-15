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
import h5py
from pathlib import Path
from random import uniform, choice
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.transform import resize
from scipy.signal import decimate
from scipy.ndimage import zoom

from collagen.data import DataProvider, ItemLoader
from collagen.core.utils import auto_detect_device
from collagen.callbacks import RunningAverageMeter, ModelSaver, RandomImageVisualizer, \
    SimpleLRScheduler, ScalarMeterLogger, ImagePairVisualizer
from collagen.losses import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss, PSNRLoss

from BoneEnhance.components.transforms import train_test_transforms
from BoneEnhance.components.models import EnhanceNet, EncoderDecoder, \
    WGAN_VGG_generator, WGAN_VGG_discriminator, WGAN_VGG, ConvNet, PerceptualNet
from BoneEnhance.components.training.loss import PerceptualLoss, TotalVariationLoss
from BoneEnhance.components.utilities import print_orthogonal
from BoneEnhance.components.training.initialize_weights import InitWeight, init_weight_normal


def init_experiment(experiments='../experiments/run'):
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', type=Path, default='../../Data')
    parser.add_argument('--workdir', type=Path, default='../../Workdir/')
    parser.add_argument('--experiment', type=Path, default=experiments)
    parser.add_argument('--seed', type=int, default=42)
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

        loss = config['training']['loss']
        architecture = config['training']['architecture']
        lr = config['training']['lr']
        mag = config['training']['magnification']

        # Snapshot directory
        #snapshot_name = time.strftime(f'{socket.gethostname()}_%Y_%m_%d_%H_%M_%S_{architecture}_{loss}_{lr}_mag{mag}')
        snapshot_name = time.strftime(f'%Y_%m_%d_%H_%M_%S_{config_path[:-4]}')
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

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
                 ScalarMeterLogger(writer, comment='training', log_dir=str(log_dir)))

    val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
               ModelSaver(metric_names='eval/loss',
                          prefix=prefix,
                          save_dir=str(current_snapshot_dir),
                          conditions='min', model=model),
               ImagePairVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std,
                                   scale=None,
                                   plot_interp=True,
                                   sigmoid=False),  # (0, 1)),
               # Reduce LR on plateau
               SimpleLRScheduler('eval/loss', ReduceLROnPlateau(optimizer,
                                                                patience=int(config.training.patience),
                                                                factor=float(config.training.factor),
                                                                eps=float(config.training.eps))),
               ScalarMeterLogger(writer=writer, comment='validation', log_dir=log_dir))

    if len(config.training.crop_small) == 2:
        val_cbs += (RandomImageVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std,
                                          sigmoid=False), )

    return train_cbs, val_cbs


def init_loss(loss, config, device='cuda', mean=None, std=None):
    vol = len(config.training.crop_small) == 3
    available_losses = {
        'mse': nn.MSELoss(),
        'L1': nn.L1Loss(), 'l1': nn.L1Loss(),
        'tv': TotalVariationLoss(),
        'psnr': PSNRLoss(),
        'perceptual': PerceptualLoss(),
        'perceptual_layers': PerceptualLoss(criterion=nn.MSELoss(),
                                            compare_layer=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],  #['relu1_2', 'relu2_2'],
                                            mean=mean, std=std,
                                            imagenet_normalize=config.training.imagenet_normalize_loss,
                                            gram=config.training.gram,
                                            vol=vol),
        'combined': CombinedLoss([PerceptualLoss().to(device), nn.L1Loss().to(device)], weights=[0.8, 0.2]),
        'combined_layers': CombinedLoss([PerceptualLoss(criterion=nn.MSELoss(),
                                                        compare_layer=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],  #['relu1_2', 'relu2_2'],
                                                        mean=mean, std=std,
                                                        imagenet_normalize=config.training.imagenet_normalize_loss,
                                                        gram=config.training.gram,
                                                        vol=vol)
                                        .to(device),
                                        nn.L1Loss().to(device)],
                                        weights=[0.8, 0.2]),
        'combined_tv': CombinedLoss([PerceptualLoss(criterion=nn.MSELoss(),
                                                    compare_layer=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                                                    mean=mean, std=std,
                                                    imagenet_normalize=config.training.imagenet_normalize_loss,
                                                    gram=config.training.gram,
                                                    vol=vol)
                                     .to(device),
                                     nn.MSELoss().to(device),
                                     TotalVariationLoss()],
                                    weights=[0.6, 0.2, 0.2]),
        # GAN
        # Segmentation losses
        'bce': BCEWithLogitsLoss2d(),
        'jaccard': SoftJaccardLoss(use_log=config.training.log_jaccard),

    }

    return available_losses[loss].to(device)


def init_model(config, device='cuda', gpus=1, args=None):
    config.model.magnification = config.training.magnification
    architecture = config.training.architecture
    vol = len(config.training.crop_small) == 3

    # List available model architectures
    available_models = {
        'encoderdecoder': EncoderDecoder(**config['model']),
        'enhance': EnhanceNet(config.training.crop_small, config.training.magnification,
                              activation=config.training.activation,
                              add_residual=config.training.add_residual,
                              upscale_input=config.training.upscale_input),
        'convnet': ConvNet(config.training.magnification,
                           activation=config.training.activation,
                           upscale_input=config.training.upscale_input,
                           n_blocks=config.training.n_blocks,
                           normalization=config.training.normalization),
        'perceptualnet': PerceptualNet(config.training.magnification,
                                       resize_convolution=config.training.upscale_input,
                                       norm=config.training.normalization,
                                       vol=vol),
        #'wgan': WGAN_VGG(input_size=config.training.crop_small[0]),
        #'wgan_g': WGAN_VGG_generator(),
        #'wgan_d': WGAN_VGG_discriminator(config.training.crop_small[0]),
    }


    # Check for multi-gpu
    if gpus > 1:
        #model = nn.DataParallel(available_models[architecture])
        model = nn.DataParallel(PerceptualNet(config.training.magnification,
                                       resize_convolution=config.training.upscale_input,
                                       norm=config.training.normalization,
                                       vol=vol))
    else:
        model = available_models[architecture]

    # Save the model architecture
    with open(args.snapshots_dir / config.training.snapshot / 'architecture.yml', 'w') as f:
        print(model, file=f)

    # Pretrained model from a previous snapshot
    if config.training.pretrain:
        # Set up path
        model_path = args.snapshots_dir / config.training.existing_model
        model_path = glob(str(model_path) + '/*fold_*.pth')
        model_path.sort()
        # Load weights
        model.load_state_dict(torch.load(model_path[0]))
    else:
        init = InitWeight(init_weight_normal, [0.0, 0.02], type='conv')
        model.apply(init)

    return model.to(device)


def create_data_provider(args, config, parser, metadata, mean, std):
    # Compile ItemLoaders
    item_loaders = dict()
    for stage in ['train', 'eval']:
        item_loaders[f'loader_{stage}'] = ItemLoader(meta_data=metadata[stage],
                                                     transform=train_test_transforms(config, mean, std)[stage],
                                                     parse_item_cb=parser,
                                                     batch_size=config.training.bs, num_workers=args.num_threads,
                                                     shuffle=True if stage == "train" else False)

    return DataProvider(item_loaders)


# 16x16x16 crops for the input data
# 8x8x8 patches randomly inside the 16x16x16
# 32x32x32 with 4x magnification

def parse_grayscale(root, entry, transform, data_key, target_key, debug=False, config=None):

    target = cv2.imread(str(entry.target_fname), -1)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    target[:, :, 1] = target[:, :, 0]
    target[:, :, 2] = target[:, :, 0]

    # Magnification
    mag = config.training.magnification
    k = choice([5])

    # Resize target to 4x magnification respect to input
    if config is not None and not config.training.crossmodality:

        # Resize target to a relevant size (from the 3.2µm resolution to 51.2µm
        new_size = (target.shape[1] // 16, target.shape[0] // 16)

        # Antialiasing
        target = cv2.GaussianBlur(target, ksize=(k, k), sigmaX=0)

        target = cv2.resize(target.copy(), new_size)  # .transpose(1, 0, 2)
        #target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True).astype('uint8')

        new_size = (target.shape[1] // mag, target.shape[0] // mag)

        # No antialias
        #img = cv2.resize(target, new_size, interpolation=cv2.INTER_LANCZOS4)
        # Antialias
        img = cv2.resize(cv2.GaussianBlur(target, ksize=(k, k), sigmaX=0), new_size)
        #img = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True, anti_aliasing_sigma=k).astype('uint8')
    elif config is not None:

        # Read image and target
        img = cv2.imread(str(entry.fname), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[:, :, 1] = img[:, :, 0]
        img[:, :, 2] = img[:, :, 0]

        new_size = (img.shape[1] * mag, img.shape[0] * mag)
        target = cv2.GaussianBlur(target, ksize=(k, k), sigmaX=0)
        target = cv2.resize(target, new_size)
        #target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True, anti_aliasing_sigma=k).astype('uint8')
    else:
        raise NotImplementedError

    # Apply random transforms
    img, target = transform((img, target))

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    #img = img.permute(2, 0, 1)# / 255. # TODO Experiment for the change in input scaling
    target = target / 255.#.permute(2, 0, 1) / 255.

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.99:
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(np.asarray(img.permute(1, 2, 0) / 255.), cmap='gray')
        plt.colorbar(im, orientation='horizontal')
        plt.title('Input')

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(np.asarray(target.permute(1, 2, 0)), cmap='gray')
        plt.colorbar(im2, orientation='horizontal')
        plt.title('Target')
        plt.show()

    return {data_key: img, target_key: target}


def parse_3d(root, entry, transform, data_key, target_key, debug=False, config=None):

    # Load target with hdf5
    with h5py.File(entry.target_fname, 'r') as f:
        target = f['data'][:]

    # Magnification
    mag = config.training.magnification

    cm = choice([True, False])
    cm = config.training.crossmodality

    # Resize target to 4x magnification respect to input
    #if config is not None and not config.training.crossmodality:
    if not cm:

        # Resize target with the given magnification to provide the input image
        new_size = (target.shape[0] // mag, target.shape[1] // mag, target.shape[2] // mag)

        sigma = choice([1, 2, 3, 4, 5])
        img = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True, anti_aliasing_sigma=sigma).astype('uint8')

    elif config is not None:

        # Load input with hdf5
        with h5py.File(entry.fname, 'r') as f:
            img = f['data'][:]

        # Resize the target to match input in case of a mismatch
        new_size = (int(img.shape[0] * mag), int(img.shape[1] * mag), int(img.shape[2] * mag))
        if target.shape != new_size:
            target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True).astype('uint8')
    else:
        raise NotImplementedError

    # Channel dimension
    img = np.expand_dims(img, -1)
    target = np.expand_dims(target, -1)

    # Apply random transforms
    img, target = transform((img, target))

    # Images are in the format 3xHxWxD
    # and scaled to 0-1 range
    img = img.repeat(3, 1, 1, 1)
    target = target.repeat(3, 1, 1, 1)
    #img /= 255.
    target /= 255.

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.95:
        res = 0.2  # In mm
        print_orthogonal(img[0, :, :, :].numpy() / 255, title='Input', res=res)

        print_orthogonal(target[0, :, :, :].numpy(), title='Target', res=res / mag)

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
