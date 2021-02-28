import os
import h5py
import numpy as np
from pathlib import Path
import cv2
from BoneEnhance.components.utilities.main import load, save, print_orthogonal, load_logfile
from skimage.transform import resize

if __name__ == "__main__":
    # Initialize experiment
    images_loc = Path('/media/santeri/Transcend/1176 Reconstructions')

    # Save path
    images_save = Path('/media/santeri/data/BoneEnhance/Data/target_1176_HR')
    images_save.mkdir(exist_ok=True)
    # Output resolution
    res_out = 50
    # Stack size
    crop_size = np.array([128, 128, 128])
    # Antialiasing sigma
    sigma = 0.5

    # List samples
    samples = os.listdir(images_loc)
    samples = [name for name in samples if os.path.isdir(os.path.join(images_loc, name))]
    samples.sort()

    #samples = samples[3:]

    # Resample datasets, create 3D stack
    for sample in samples:
        print(f'Processing sample: {sample}')
        #try:
        # Load log file to check resolution
        im_path = images_loc / sample
        log = load_logfile(str(im_path))
        res = float(log['Image Pixel Size (um)'])
        #res = 34.84

        # Scale factors and scaled crops
        factor = res_out / res
        crop_large = np.floor(crop_size * factor).astype('uint32')

        # Load images
        data, files = load(im_path, axis=(1, 2, 0))

        # Visualize full stack
        print_orthogonal(data, res=res/1e3, invert=True, cbar=True, scale_factor=10)

        # Create Save directory
        #(images_save / sample).mkdir(exist_ok=True)

        # Crop small samples and scale to proper size
        n_crops = data.shape // crop_large
        crop_begin = (data.shape - n_crops * crop_large) // 2
        for x in range(n_crops[0]):
            for y in range(n_crops[1]):
                for z in range(n_crops[2]):

                    # Crop according to scale of crop_large
                    data_out = data[
                               x * crop_large[0] + crop_begin[0]:(x + 1) * crop_large[0] + crop_begin[0],
                               y * crop_large[1] + crop_begin[1]:(y + 1) * crop_large[1] + crop_begin[1],
                               z * crop_large[2] + crop_begin[2]:(z + 1) * crop_large[2] + crop_begin[2]
                               ]

                    # Crop is now in resolution "res_out"
                    data_out = resize(data_out, crop_size, order=0, anti_aliasing=True, preserve_range=True,
                                      anti_aliasing_sigma=sigma).astype('uint8')

                    # Save the cropped volume to hdf5
                    fname = str(images_save / f'{sample}_{str(x).zfill(3)}{str(y).zfill(3)}{str(z).zfill(3)}.h5')
                    with h5py.File(fname, 'w') as f:
                        f.create_dataset('data', data=data_out)

        #except (ValueError, FileNotFoundError):
        #    print(f'Error in sample {sample}')
        #    continue
