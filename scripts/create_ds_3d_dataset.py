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
    images_loc = Path('/media/santeri/data/BoneEnhance/Data/target_IVD_isotropic_3D_HR_largecrop')
    images_loc = Path(f'../../Data/dental/Hampaat_target')
    # Save path
    images_save = Path('/media/santeri/data/BoneEnhance/Data/input_IVD_isotropic_3D_HR_largecrop')
    images_save = Path(f'../../Data/dental/Hampaat_input')
    images_save.mkdir(exist_ok=True)
    # Output resolution

    mag = 4
    res = 50
    save_h5 = True
    resize_3d = True
    # Antialiasing sigma
    sigma = 1
    k = 5

    # List samples
    # samples = glob(str(images_loc / '*.h5'))
    samples = os.listdir(images_loc)
    samples.sort()

    #samples = samples[3:]

    # Resample datasets, create 3D stack
    for sample in samples:
        sample = str(Path(sample).name)
        print(f'Processing sample: {sample}')

        # Load image stacks
        if sample.endswith('.h5'):
            with h5py.File(str(images_loc / sample), 'r') as f:
                data = f['data'][:]
        else:
            data, files = load(str(images_loc / sample), rgb=True, axis=(1, 2, 0))


        # Downscale to input size
        if resize_3d:
            new_size = (data.shape[0] // mag, data.shape[1] // mag, data.shape[2] // mag)
            data = resize(data, new_size, order=0, anti_aliasing=True, preserve_range=True, anti_aliasing_sigma=sigma).astype('uint8')
        else:
            new_size = (data.shape[2] // mag, data.shape[1] // mag)
            for image in range(data.shape[0]):

                data[image, :, :] = cv2.resize(
                    cv2.resize(
                        cv2.GaussianBlur(data[image, :, :], ksize=(k, k), sigmaX=sigma, sigmaY=sigma), new_size),
                    (data.shape[2], data.shape[1]), cv2.INTER_CUBIC)

        # Save the cropped volume to hdf5
        fname = str(images_save / f'{sample}')
        if save_h5:
            with h5py.File(fname, 'w') as f:
                f.create_dataset('data', data=data)
        else:
            save(fname, sample, data, dtype='.png')
        #except (ValueError, FileNotFoundError):
        #    print(f'Error in sample {sample}')
        #    continue
