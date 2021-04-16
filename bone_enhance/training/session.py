import argparse
import yaml
import numpy as np
import time
import torch
import torch.nn as nn
import dill
import os
from pathlib import Path
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collagen.data import DataProvider, ItemLoader
from collagen.core.utils import auto_detect_device
from collagen.callbacks import RunningAverageMeter, ModelSaver, RandomImageVisualizer, \
    SimpleLRScheduler, ScalarMeterLogger, ImagePairVisualizer
from collagen.losses import CombinedLoss, PSNRLoss, BCEWithLogitsLoss2d, SoftJaccardLoss

from bone_enhance.transforms import train_test_transforms
from bone_enhance.models import EnhanceNet, \
    ConvNet, PerceptualNet
from bone_enhance.training.loss import PerceptualLoss, TotalVariationLoss
from bone_enhance.training.initialize_weights import InitWeight, init_weight_normal
from collagen.modelzoo.segmentation import EncoderDecoder


def init_experiment(experiments='../experiments/run'):
    """
    Initialize general parameters that do not need to be specified in the experiment config file.
    Returns a list of experiments to be conducted.

    :param experiments: Path to the run_experiments folder.
    :return: List of DL experiments
    """

    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', type=Path, default='../../Data', help='Location of input and target images')
    parser.add_argument('--workdir', type=Path, default='../../Workdir/', help='Location of snapshots folder')
    parser.add_argument('--experiment', type=Path, default=experiments, help='Location of the experiments folder')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_threads', type=int, default=16, help='Number of CPUs')
    parser.add_argument('--gpus', type=int, default=2, help='Number of GPUs')
    #parser.add_argument('--segmentation', type=bool, default=False, help='Super-resolution or segmentation pipeline?')  # TODO Debug
    parser.add_argument('--exp_idx', type=int, default=None, help='Index for the corresponding training experiment')
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
            # Load file
            with open(args.experiment / config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                config_list.append(config)

        # Snapshot directory
        snapshot_name = time.strftime(f'%Y_%m_%d_%H_%M_%S_{config_path[:-4]}_seed{args.seed}')
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

    if args.exp_idx is not None:
        print(f'Running experiment: {config_paths[args.exp_idx][:-4]}')

    return args, config_list, config_paths, device


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
               # Save model with lowest evaluation loss
               ModelSaver(metric_names='eval/loss',
                          prefix=prefix,
                          save_dir=str(current_snapshot_dir),
                          conditions='min', model=model),
               # Visualize result images (best and worst minibatch)
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


def init_loss(loss, config, device='cuda', mean=None, std=None, args=None):
    """
    Initialize the loss function (or functions).
    """

    vol = len(config.training.crop_small) == 3
    model_path = str(args.snapshots_dir / config.training.autoencoder_pretrained)

    # Mean squared error
    if loss == 'mse':
        return nn.MSELoss().to(device)
    # Mean absolute error
    elif loss == 'L1' or loss == 'l1':
        return nn.L1Loss().to(device)
    # Total variation
    elif loss == 'tv':
        return TotalVariationLoss().to(device)
    # Peak signal-to-noise ratio
    elif loss == 'psnr':
        return PSNRLoss().to(device)
    # Combined mean squared error and total variation (good baseline for perceptual loss)
    elif loss == 'mse_tv':
        return CombinedLoss([nn.MSELoss().to(device),
                             TotalVariationLoss().to(device)], weights=[0.8, 0.2]).to(device)
    # Perceptual loss (default mode)
    elif loss == 'perceptual':
        return PerceptualLoss(config=config).to(device)
    # Perceptual loss (compare activations from different layers)
    elif loss == 'perceptual_layers':
        return PerceptualLoss(criterion=nn.MSELoss(), config=config,
                              compare_layer=['relu1_2', 'relu2_2'],  #['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                              mean=mean, std=std).to(device)
    # Combine L1 and Perceptual loss (default)
    elif loss == 'combined':
        return CombinedLoss([PerceptualLoss(config=config).to(device), nn.L1Loss().to(device)], weights=[0.8, 0.2]).to(device)
    # Combine L1 and Perceptual loss (different layers)
    elif loss == 'combined_layers':
        return CombinedLoss([PerceptualLoss(criterion=nn.MSELoss(), config=config,
                                            compare_layer=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],  #['relu1_2', 'relu2_2'],
                                            mean=mean, std=std).to(device),
                            nn.L1Loss().to(device)],
                            weights=[0.8, 0.2]).to(device)
    # Perceptual loss (layers), L1 and total variation
    elif loss == 'combined_tv':
        return CombinedLoss([PerceptualLoss(criterion=nn.MSELoss(), config=config,
                                            compare_layer=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                                            mean=mean, std=std).to(device),
                             nn.L1Loss().to(device),
                             TotalVariationLoss().to(device)],
                            weights=[0.1, 1, 1]).to(device)
    # Autoencoder loss and total variation
    elif loss == 'autoencoder_tv':
        crop_size = tuple([crop * config.training.magnification for crop in config.training.crop_small])
        return CombinedLoss([PerceptualLoss(criterion=nn.MSELoss(), config=config,
                                            compare_layer=model_path,
                                            mean=mean, std=std, plot=False, gpus=args.gpus).to(device),
                            nn.L1Loss().to(device),
                            TotalVariationLoss().to(device)],
                            weights=[0.5, 1, 1]).to(device)
    # Binary cross-entropy and soft Jaccard
    elif loss == 'bce_combined':
        return CombinedLoss([BCEWithLogitsLoss2d(),
                             SoftJaccardLoss(use_log=config['training']['log_jaccard'])]).to(device)
    # Binary cross-entropy
    elif loss == 'bce':
        return nn.BCELoss().to(device)
    else:
        raise Exception('Loss not implememnted!')


def init_model(config, device='cuda', gpus=1, args=None):
    """

    :param config:
    :param device:
    :param gpus:
    :param args:
    :return:
    """
    architecture = config.training.architecture
    vol = len(config.training.crop_small) == 3

    # List available model architectures

    # Collagen encoderdecoder for super-resolution
    if architecture == 'srencoderdecoder':
        config.model.magnification = config.training.magnification
        model = SREncoderDecoder(**config['model'])
    # Collagen encoderdecoder for segmentation
    elif architecture == 'encoderdecoder':
        model = EncoderDecoder(**config['model'])
    # Architecture inspired from the reconstruction paper
    elif architecture == 'enhance':
        model = EnhanceNet(config.training.crop_small, config.training.magnification,
                           activation=config.training.activation,
                           add_residual=config.training.add_residual,
                           upscale_input=config.training.upscale_input)
    # Simple CNN architecture
    elif architecture == 'convnet':
        model = ConvNet(config.training.magnification,
                        activation=config.training.activation,
                        upscale_input=config.training.upscale_input,
                        n_blocks=config.training.n_blocks,
                        normalization=config.training.normalization)
    # Architecture used by Johnson et al. in the Perceptual loss paper
    elif architecture == 'perceptualnet':
        model = PerceptualNet(config.training.magnification,
                              resize_convolution=config.training.upscale_input,
                              norm=config.training.normalization,
                              vol=vol, rgb=config.training.rgb)
    else:
        raise Exception('Model architecture unavailable.')

    # Check for multi-gpu
    if gpus > 1:
        model = nn.DataParallel(model)

    # Save the model architecture
    with open(args.snapshots_dir / config.training.snapshot / 'architecture.yml', 'w') as f:
        print(model, file=f)

    # Pretrained model from a previous snapshot
    if config.training.pretrain:
        # Set up path
        model_path = args.snapshots_dir / config.training.existing_model
        model_path = glob(str(model_path) + '/*fold_*.pth')
        model_path.sort()
        # Load weights from first fold
        model.load_state_dict(torch.load(model_path[0]))
    # Randomly initialized weights (from Gaussian distribution)
    else:
        init = InitWeight(init_weight_normal, [0.0, 0.02], type='conv')
        model.apply(init)

    return model.to(device)


def create_data_provider(args, config, parser, metadata, mean, std):
    """
    Creates the dataloader object (for Collagen framework)
    """
    # Compile ItemLoaders
    item_loaders = dict()
    for stage in ['train', 'eval']:
        item_loaders[f'loader_{stage}'] = ItemLoader(meta_data=metadata[stage],
                                                     transform=train_test_transforms(config, args, mean, std)[stage],
                                                     parse_item_cb=parser,
                                                     batch_size=config.training.bs, num_workers=args.num_threads,
                                                     shuffle=True if stage == "train" else False)

    return DataProvider(item_loaders)


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
    """
    Save used augmentations.
    :param path: Path for the augmentation list.
    :return:
    """
    # Build the augmentations
    transforms = train_test_transforms(config, args, mean, std)
    # Save the experiment parameters
    with open(path / 'transforms.yaml', 'w') as f:
        yaml.dump(transforms['train_list'][1].to_yaml(), f)
