import os
import h5py
import argparse
import numpy as np
import cv2

from pathlib import Path
from bone_enhance.utilities.main import load, save, print_orthogonal, load_logfile
from skimage.transform import resize

if __name__ == "__main__":
    # Initialize experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_loc', type=Path, default='../../Data/dental/Hampaat_rec')
    #'/media/santeri/Transcend/1176 Reconstructions')
    parser.add_argument('--images_save', type=Path, default='../../Data/dental/Hampaat_target')
    parser.add_argument('--res_out', type=int, default=50, help='Target resolution for training data (in µm)')
    parser.add_argument('--completed', type=int, default=0, help='Samples already processed and skipped.')
    parser.add_argument('--crop_size', type=list, default=[100, 100, 100], help='Size of one training patch')
    parser.add_argument('--sigma', type=float, default=0.5, help='Standard deviation of gaussian blur (antialiasing).')
    parser.add_argument('--hdf5', type=bool, default=False, help='Save as 3D data (True) or a stack of 2D images.')

    args = parser.parse_args()

    # Save path
    args.images_save.mkdir(exist_ok=True)

    # List samples
    samples = os.listdir(args.images_loc)
    samples = [name for name in samples if os.path.isdir(os.path.join(args.images_loc, name))]
    samples.sort()

    if args.completed > 0:
        samples = samples[args.completed:]

    # Resample datasets, create 3D stack
    for sample in samples:
        print(f'Processing sample: {sample}')
        #try:
        # Load log file to check resolution
        im_path = args.images_loc / sample
        log = load_logfile(str(im_path))
        res = float(log['Image Pixel Size (um)'])
        #res = 132.75

        # Scale factors and scaled crops
        factor = args.res_out / res
        crop_large = np.floor(np.array(args.crop_size) * factor).astype('uint32')

        # Load images
        data, files = load(im_path, axis=(1, 2, 0))

        # Visualize full stack
        #print_orthogonal(data, res=res/1e3, invert=True, cbar=True, scale_factor=10)

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
                    #data_out = resize(data_out, args.crop_size, order=0, anti_aliasing=True, preserve_range=True,
                    #                  anti_aliasing_sigma=args.sigma).astype('uint8')

                    # Save the cropped volume to hdf5
                    if args.hdf5:
                        fname = str(args.images_save / f'{sample}_{str(x).zfill(3)}{str(y).zfill(3)}{str(z).zfill(3)}.h5')
                        with h5py.File(fname, 'w') as f:
                            f.create_dataset('data', data=data_out)
                    else:
                        fname = Path(f'{sample}_{str(x).zfill(3)}{str(y).zfill(3)}{str(z).zfill(3)}')

                        save(str(args.images_save / fname), fname.name, data_out, verbose=False)
        #except (ValueError, FileNotFoundError):
        #    print(f'Error in sample {sample}')
        #    continue
