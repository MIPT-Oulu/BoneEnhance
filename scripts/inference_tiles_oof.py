import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
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
from BoneEnhance.components.inference.model_components import InferenceModel, load_models
from BoneEnhance.components.inference.pipeline_components import inference, largest_object

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

    snaps = ['dios-erc-gpu_2020_07_17_11_32_31_enhance_combined']

    snaps = [args.snapshots / snap for snap in snaps]

    # Iterate through snapshots
    args.save_dir.mkdir(exist_ok=True)
    for snap in snaps:
        start = time()

        save_dir = args.save_dir / str(snap.stem + '_oof')
        save_dir.mkdir(exist_ok=True)

        # Load snapshot configuration
        with open(snap / 'config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        with open(snap / 'args.dill', 'rb') as f:
            args_experiment = dill.load(f)

        with open(snap / 'split_config.dill', 'rb') as f:
            split_config = dill.load(f)

        device = auto_detect_device()

        # Load models
        model_list = load_models(str(snap), config, n_gpus=args_experiment.gpus)

        print(f'Found {len(model_list)} models.')

        # Create directories
        save_dir.mkdir(exist_ok=True)
        input_x = config['training']['crop_small'][0]
        input_y = config['training']['crop_small'][1]

        # Loop for all images
        for fold in range(len(model_list)):
            # List validation images
            validation_files = split_config[f'fold_{fold}']['val'].target_fname.values

            # Model without validation images
            model = InferenceModel([model_list[fold]]).to(device)
            model.eval()

            for file in tqdm(validation_files, desc=f'Running inference for fold {fold}'):

                img_full = cv2.imread(str(file))

                resize = (img_full.shape[1] // 32, img_full.shape[0] // 32)
                img_full = cv2.resize(img_full.copy(), resize)

                with torch.no_grad():  # Do not update gradients
                    prediction = inference(model, args, config, img_full, weight=args.weight,
                                           #mean=split_config['mean'], std=split_config['std']
                                           )[:, :, 0]
                    plt.imshow(prediction)
                    plt.colorbar()
                    plt.show()

                prediction = (prediction * 255).astype('uint8')

                # When saving 3D stacks, file structure should be preserved
                (save_dir / file.parent.stem).mkdir(exist_ok=True)
                cv2.imwrite(str(save_dir / file.parent.stem / file.stem) + '.bmp', prediction)

                # Free memory
                torch.cuda.empty_cache()
                gc.collect()

        dur = time() - start
        print(f'Inference completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds.')