import cv2
import os
import numpy as np
from pathlib import Path
from bone_enhance.training.session import init_experiment
from bone_enhance.utilities.main import load, save, print_orthogonal
from skimage.transform import resize
import h5py
from scipy.ndimage import zoom
from PIL import Image

if __name__ == "__main__":
    # Magnification
    mag = 4

    # Initialize experiment
    images_loc = Path('/media/dios/kaappi/Sakke/Saskatoon/µCT/Recs_bone')
    input_loc = Path('/media/santeri/data/BoneEnhance/Data/input_original')

    images_save = Path(f'/media/santeri/data/BoneEnhance/Data/target_mag{mag}')
    input_save = Path(f'/media/santeri/data/BoneEnhance/Data/input_mag{mag}')
    images_save.mkdir(exist_ok=True)
    input_save.mkdir(exist_ok=True)

    # List samples
    samples = os.listdir(images_loc)
    samples = [name for name in samples if os.path.isdir(os.path.join(images_loc, name))]
    samples_input = os.listdir(input_loc)
    samples_input = [name for name in samples_input if os.path.isdir(os.path.join(input_loc, name))]
    # Check for consistency
    assert len(samples) == len(samples_input)

    samples.sort()
    samples_input.sort()
    #samples = samples[22:]
    samples_input = samples_input[:]
    for i in range(len(samples)):
        print(f'Processing sample: {samples[i]}')

        input_path = input_loc / samples_input[i]
        im_path = images_loc / samples[i]

        # (1, 2, 0) = Original dimension
        data_input, _ = load(input_path, axis=(1, 2, 0))
        data, _ = load(im_path, axis=(1, 2, 0))

        print_orthogonal(data_input)
        print_orthogonal(data)

        factor = (data.shape[0] * mag // data_input.shape[0],
                  data.shape[1] * mag // data_input.shape[1],
                  data.shape[2] * mag // data_input.shape[2])

        factor = (data_input.shape[0] * mag,
                  data_input.shape[1] * mag,
                  data_input.shape[2] * mag)
        print(factor)

        # Gaussian blur antialiasing and preserve 8-bit range
        data = resize(data, factor, order=0, anti_aliasing=True, preserve_range=True)

        # Save target to hdf5
        fname = str(images_save / f'{samples[i]}.h5')
        with h5py.File(fname, 'w') as f:
            f.create_dataset('data', data=data)

        # Save input to hdf5
        fname = str(input_save / f'{samples_input[i]}.h5')
        with h5py.File(fname, 'w') as f:
            f.create_dataset('data', data=data_input)
