import cv2
from omegaconf import OmegaConf
from pathlib import Path
from time import time
import argparse
import dill
import os
import yaml

from collagen.core.utils import auto_detect_device
from bone_enhance.inference.model_components import load_models
from bone_enhance.inference.pipeline_components import inference_runner_oof, evaluation_runner

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=Path, default='../../Data/predictions_oof_wacv')
    parser.add_argument('--eval_dir', type=Path, default='../../Data/evaluation_oof_wacv')
    parser.add_argument('--data_location', type=Path, default='../../Data')
    parser.add_argument('--snap_id', type=int, default=None)
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

    #path = '../../Workdir/wacv_experiments_new'
    #path = '../../Workdir/IVD_experiments_2D'
    path = args.snapshots
    snaps = os.listdir(path)
    suffixes = ['']
    snaps = [Path(os.path.join(path, snap)) for snap in snaps if os.path.isdir(os.path.join(path, snap))]
    if args.snap_id is not None:
        snaps = [snaps[args.snap_id - 1]]
    #snaps = [args.snapshots / '2021_06_29_15_08_12_3D_perceptual_tv_IVD_4x_pretrained_isotropic_seed42']

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
            args_experiment.bs = args.bs
            args_experiment.snapshots_dir = Path(path)

        with open(snap / 'split_config.dill', 'rb') as f:
            split_config = dill.load(f)

        device = auto_detect_device()

        # Load models
        model_list = load_models(str(snap), config, n_gpus=args_experiment.gpus)

        print(f'Found {len(model_list)} models.')

        # Correct data location in case of a cluster experiment
        args_experiment.data_location = args.data_location
        args_experiment.eval_dir = args.eval_dir
        args_experiment.save_dir = args.save_dir

        # Inference
        save_d = inference_runner_oof(args_experiment, config, split_config, device, plot=args.plot, verbose=False)
        #save_d = Path('../../Data/predictions_oof_wacv') / str(config['training']['snapshot'] + '_oof')

        # Evaluate predictions on validation images
        evaluation_runner(args_experiment, config, save_d, suffix=config.training.suffix, use_bvtv=True)

        dur = time() - start
        print(f'Inference completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
