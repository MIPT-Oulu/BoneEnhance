import pandas as pd
import pathlib
import torch
import dill
from collagen import ItemLoader
from sklearn import model_selection
from tqdm import tqdm
from pathlib import Path

from BoneEnhance.components.transforms import train_test_transforms


def build_meta_from_files(base_path, config, args=None):
    # Dataframe for the metadata
    metadata = {'fname': [], 'target_fname': []}

    # Data path
    if len(config.training.crop_small) == 3 and config.training.suffix == '':
        suffix = '_3d'
    else:
        suffix = config.training.suffix

    if config.training.suffix is not None:
        target_loc = base_path / f'target{suffix}'
        input_loc = base_path / f'input{suffix}'
    else:
        target_loc = base_path / 'target'
        input_loc = base_path / 'input'

    if config.autoencoder:
        input_loc = target_loc

    if not config.training.crossmodality and len(config.training.crop_small) == 3:
        input_loc = Path(str(input_loc) + '_ds')

    # 3D metadata
    if len(config.training.crop_small) == 3:
        input_images = list(map(lambda x: pathlib.Path(x), input_loc.glob('*.h5')))
        target_images = list(map(lambda x: pathlib.Path(x), target_loc.glob('*.h5')))
    else:
        # List files
        input_images = list(map(lambda x: pathlib.Path(x), input_loc.glob('**/*[0-9].[pb][nm][gp]')))
        target_images = list(map(lambda x: pathlib.Path(x), target_loc.glob('**/*[0-9].[pb][nm][gp]')))
    input_images.sort()
    target_images.sort()

    # Check for data consistency
    assert len(input_images), len(target_images)

    # Dataframe
    if len(config.training.crop_small) == 3:
        [metadata['fname'].append((input_loc / img_name.name)) for img_name in input_images]
        [metadata['target_fname'].append(target_loc / img_name.name) for img_name in target_images]
    else:
        # Folders arranged in subfolders based on sample
        [metadata['fname'].append((input_loc / img_name.parent / img_name.name)) for img_name in input_images]
        [metadata['target_fname'].append(target_loc / img_name.parent / img_name.name) for img_name in target_images]

    return pd.DataFrame(data=metadata)


def build_splits(data_dir, args, config, parser, snapshots_dir, snapshot_name):
    # Metadata
    metadata = build_meta_from_files(data_dir, config, args=args)
    # Group_ID
    #metadata['subj_id'] = metadata.fname.apply(lambda x: '_'.join(x.stem.split('_', 4)[:-1]), 0)
    metadata['subj_id'] = metadata.fname.apply(lambda x: '_'.join(x.stem.split('_', 4)[:2]), 0)
    # Special case for samples with Group name separated by -
    metadata['subj_id'] = metadata.subj_id.apply(lambda x: '_'.join(x.split('-', 4)[:1]), 0)

    # Mean and std
    crop = config['training']['crop_small']
    if config['training']['crossmodality']:
        cm = 'cm'
    else:
        cm = 'ds'
    mean_std_path = snapshots_dir / f"mean_std_{crop}_{cm}.pth"
    if mean_std_path.is_file() and not config['training']['calc_meanstd']:  # Load
        print('==> Loading mean and std from cache')
        tmp = torch.load(mean_std_path)
        mean, std = tmp['mean'], tmp['std']
    else:  # Calculate
        print('==> Estimating mean and std')
        mean, std = estimate_mean_std(config, args, metadata, parser)
        torch.save({'mean': mean, 'std': std}, mean_std_path)

    print('==> Mean:', mean)
    print('==> STD:', std)

    # Group K-Fold by patient ID
    gkf = model_selection.GroupKFold(n_splits=config['training']['n_folds'])
    # K-fold by random shuffle
    #gkf = model_selection.KFold(n_splits=config['training']['n_folds'], shuffle=True, random_state=args.seed)

    # Create splits for all folds
    splits_metadata = dict()
    iterator = gkf.split(metadata.fname.values, groups=metadata.subj_id.values)
    for fold in range(config['training']['n_folds']):
        train_idx, val_idx = next(iterator)
        splits_metadata[f'fold_{fold}'] = {'train': metadata.iloc[train_idx],
                                           'eval': metadata.iloc[val_idx]}

    # Add mean and std to metadata
    splits_metadata['mean'] = mean
    splits_metadata['std'] = std

    with open(snapshots_dir / snapshot_name / 'split_config.dill', 'wb') as f:
        dill.dump(splits_metadata, f)

    return splits_metadata


def estimate_mean_std(config, args, metadata, parse_item_cb):
    mean_std_loader = ItemLoader(meta_data=metadata,
                                 transform=train_test_transforms(config, args)['train'],
                                 parse_item_cb=parse_item_cb,
                                 batch_size=config.training.bs, num_workers=args.num_threads,
                                 shuffle=False)

    mean = None
    std = None
    for _ in tqdm(range(len(mean_std_loader)), desc='Calculating mean and standard deviation'):
        for batch in mean_std_loader.sample():
            if mean is None:
                mean = torch.zeros(batch['data'].size(1))
                std = torch.zeros(batch['data'].size(1))
            # for channel in range(batch['data'].size(1)):
            #     mean[channel] += batch['data'][:, channel, :, :].mean().item()
            #     std[channel] += batch['data'][:, channel, :, :].std().item()
            mean += batch['data'].mean().item()
            std += batch['data'].std().item()

    mean /= len(mean_std_loader)
    std /= len(mean_std_loader)

    return mean, std