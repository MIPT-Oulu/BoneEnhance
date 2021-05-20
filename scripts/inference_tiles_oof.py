import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pathlib import Path
from time import time
from copy import deepcopy
import argparse
import dill
import torch
import yaml
from tqdm import tqdm
from glob import glob

from collagen.core.utils import auto_detect_device
from bone_enhance.inference.model_components import InferenceModel, load_models
from bone_enhance.inference.pipeline_components import inference, largest_object, inference_runner_oof

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default='../../Data/images')
    parser.add_argument('--save_dir', type=Path, default='../../Data/predictions')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--magnification', type=int, default=4)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--threshold', type=float, default=0.8)
    # µCT snapshot
    parser.add_argument('--snapshots', type=Path, default='../../Workdir/snapshots/')
    args = parser.parse_args()

    # Snapshots to be evaluated
    # µCT models

    snaps = ['2021_05_14_13_50_15_2D_perceptual_tv_1176_HR_seed42']
    #snaps = ['dios-erc-gpu_2020_10_12_12_50_52_perceptualnet_newsplit_cm_bg']



    snaps = [args.snapshots / snap for snap in snaps]

    # Iterate through snapshots
    args.save_dir.mkdir(exist_ok=True)
    for snap in snaps:
        start = time()

        # Load snapshot configuration
        with open(snap / 'config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config = OmegaConf.create(config)

        with open(snap / 'args.dill', 'rb') as f:
            args_experiment = dill.load(f)

        with open(snap / 'split_config.dill', 'rb') as f:
            split_config = dill.load(f)

        device = auto_detect_device()
        args_experiment.bs = config.training.bs

        save_dir = inference_runner_oof(args_experiment, config, split_config, device)

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

        dur = time() - start
        print(f'Inference completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds.')