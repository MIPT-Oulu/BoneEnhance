import os
import h5py
import numpy as np
from pathlib import Path
from glob import glob
import cv2
from bone_enhance.utilities.main import load, save, print_orthogonal, load_logfile
from skimage.transform import resize

if __name__ == "__main__":
    # Initialize experiment
    images_loc = Path('/media/santeri/data/BoneEnhance/Data/target_IVD_isotropic_3D')

    # Save path
    images_save = Path('/media/santeri/data/BoneEnhance/Data/input_IVD_isotropic_3D_ds')
    images_save.mkdir(exist_ok=True)
    # Output resolution
    mag = 4
    res = 50
    # Stack size
    crop_size = np.array([128, 128, 128])
    # Antialiasing sigma
    sigma = 0.5

    # List samples
    samples = glob(str(images_loc / '*.h5'))
    samples.sort()

    #samples = samples[3:]

    # Resample datasets, create 3D stack
    for sample in samples:
        sample = str(Path(sample).name)
        print(f'Processing sample: {sample}')
        #try:
        # Load log file to check resolution
        im_path = images_loc / sample

        # Load images
        with h5py.File(im_path, 'r') as f:
            data = f['data'][:]

        # Visualize full stack
        #print_orthogonal(data, res=res/1e3, invert=True, cbar=True, scale_factor=10)

        # Downscale to input size
        new_size = (data.shape[0] // mag, data.shape[1] // mag, data.shape[2] // mag)
        data_out = resize(data, new_size, order=0, anti_aliasing=True, preserve_range=True,
                     anti_aliasing_sigma=sigma).astype('uint8')

        # Save the cropped volume to hdf5
        fname = str(images_save / f'{sample}')
        with h5py.File(fname, 'w') as f:
            f.create_dataset('data', data=data_out)

        #except (ValueError, FileNotFoundError):
        #    print(f'Error in sample {sample}')
        #    continue
