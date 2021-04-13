import os
import h5py
from pathlib import Path
from BoneEnhance.components.utilities import print_orthogonal
from BoneEnhance.components.training.session import init_experiment

if __name__ == "__main__":
    # Initialize experiment
    args, _, _, _ = init_experiment()
    base_path = args.data_location
    images_loc = base_path / 'target_3d'

    # List files
    samples = os.listdir(images_loc)
    samples.sort()
    for sample in samples:
        im_path = images_loc / Path(sample)

        with h5py.File(str(im_path), 'r') as f:
            data_xy = f['data'][:]

        print_orthogonal(data_xy, title=sample[:-3], res=3.2, scale_factor=1000)
