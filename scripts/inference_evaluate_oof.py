import cv2
from omegaconf import OmegaConf
from pathlib import Path
from time import time
import argparse
import dill
import yaml

from collagen.core.utils import auto_detect_device
from BoneEnhance.components.inference.model_components import load_models
from BoneEnhance.components.inference.pipeline_components import inference_runner_oof, evaluation_runner

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=Path, default='../../Data/evaluation_oof')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--magnification', type=int, default=4)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--weight', type=str, choices=['gaussian', 'mean'], default='gaussian')
    # µCT snapshot
    parser.add_argument('--snapshots', type=Path, default='../../Workdir/snapshots/')
    args = parser.parse_args()

    # Snapshots to be evaluated
    # µCT models

    snaps = ['2021_03_03_11_52_07_1_3D_mse_tv_1176_HR', '2021_03_04_10_11_34_1_3D_mse_tv_1176', '2021_02_26_05_52_47_3D_perceptualnet_ds_mse_tv']
    suffixes = ['_1176_HR', '_1176', '_3d']
    snaps = [args.snapshots / snap for snap in snaps]

    # Iterate through snapshots
    args.save_dir.mkdir(exist_ok=True)
    for idx, snap in enumerate(snaps):
        start = time()

        # Load snapshot configuration
        with open(snap / 'config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config = OmegaConf.create(config)

        with open(snap / 'args.dill', 'rb') as f:
            args_experiment = dill.load(f)

        with open(snap / 'split_config.dill', 'rb') as f:
            split_config = dill.load(f)

        if not config.training.crossmodality:
            save_dir = args.save_dir / str(snap.stem + '_downscale_oof')
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = args.save_dir / str(snap.stem + '_oof')
            save_dir.mkdir(exist_ok=True)

        device = auto_detect_device()

        # Load models
        model_list = load_models(str(snap), config, n_gpus=args_experiment.gpus)

        print(f'Found {len(model_list)} models.')

        # Create directories
        save_dir.mkdir(exist_ok=True)
        input_x = config['training']['crop_small'][0]
        input_y = config['training']['crop_small'][1]

        save_d = inference_runner_oof(args_experiment, config, split_config, device, plot=args.plot)

        masks = snap == '2021_02_26_05_52_47_3D_perceptualnet_ds_mse_tv'
        evaluation_runner(args_experiment, config, save_d, masks=masks, suffix=suffixes[idx])

        dur = time() - start
        print(f'Inference completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
