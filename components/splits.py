import cv2
import pandas as pd
import pathlib
import torch
import dill
from sklearn import model_selection

from BoneEnhance.components.transforms.main import estimate_mean_std


def build_meta_from_files(base_path, phase='train'):
    if phase == 'train':
        target_loc = base_path / 'target'
        input_loc = base_path / 'input'
    else:
        target_loc = base_path / 'target_test'
        input_loc = base_path / 'input_test'

    # List files
    input_images = set(map(lambda x: x.stem, input_loc.glob('**/*[0-9].[pb][nm][gp]')))
    target_images = set(map(lambda x: x.stem, target_loc.glob('**/*[0-9].[pb][nm][gp]')))
    res = target_images.intersection(input_images)

    #target_images = list(map(lambda x: pathlib.Path(x).with_suffix('.png'), target_images))
    input_images = list(map(lambda x: pathlib.Path(x.name), input_loc.glob('**/*[0-9].[pb][nm][gp]')))
    target_images = list(map(lambda x: pathlib.Path(x.name), target_loc.glob('**/*[0-9].[pb][nm][gp]')))
    input_images.sort()
    target_images.sort()

    #assert len(res), len(target_images)

    d_frame = {'fname': [], 'target_fname': []}

    # Making dataframe

    [d_frame['fname'].append((input_loc / str(img_name).rsplit('_', 1)[0] / img_name)) for img_name in input_images]
    [d_frame['target_fname'].append(target_loc / str(img_name).rsplit('_', 1)[0] / img_name) for img_name in target_images]

    metadata = pd.DataFrame(data=d_frame)

    return metadata


def build_splits(data_dir, args, config, parser, snapshots_dir, snapshot_name):
    # Metadata
    metadata = build_meta_from_files(data_dir)
    # Group_ID
    metadata['subj_id'] = metadata.fname.apply(lambda x: '_'.join(x.stem.split('_', 4)[:-1]), 0)

    # Mean and std
    crop = config['training']['crop_small']
    mean_std_path = snapshots_dir / f"mean_std_{crop[0]}x{crop[1]}.pth"
    if mean_std_path.is_file() and not config['training']['calc_meanstd']:  # Load
        print('==> Loading mean and std from cache')
        tmp = torch.load(mean_std_path)
        mean, std = tmp['mean'], tmp['std']
    else:  # Calculate
        print('==> Estimating mean and std')
        mean, std = estimate_mean_std(config, metadata, parser, args.num_threads, config['training']['bs'])
        torch.save({'mean': mean, 'std': std}, mean_std_path)

    print('==> Mean:', mean)
    print('==> STD:', std)

    # Group K-Fold by rabbit ID
    gkf = model_selection.GroupKFold(n_splits=config['training']['n_folds'])
    # K-fold by random shuffle
    #gkf = model_selection.KFold(n_splits=config['training']['n_folds'], shuffle=True, random_state=args.seed)

    # Create splits for all folds
    splits_metadata = dict()
    iterator = gkf.split(metadata.fname.values, groups=metadata.subj_id.values)
    for fold in range(config['training']['n_folds']):
        train_idx, val_idx = next(iterator)
        splits_metadata[f'fold_{fold}'] = {'train': metadata.iloc[train_idx],
                                           'val': metadata.iloc[val_idx]}

    # Add mean and std to metadata
    splits_metadata['mean'] = mean
    splits_metadata['std'] = std

    with open(snapshots_dir / snapshot_name / 'split_config.dill', 'wb') as f:
        dill.dump(splits_metadata, f)

    return splits_metadata


