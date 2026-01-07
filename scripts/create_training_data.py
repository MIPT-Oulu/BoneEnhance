import os
import h5py
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2

from pathlib import Path
from scipy.ndimage import zoom
from bone_enhance.utilities.main import load, save, print_orthogonal, load_logfile, downscale_image
from skimage.transform import resize

if __name__ == "__main__":
    # Initialize experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_loc', type=Path, default=\
        #'/media/santeri/data2/BoneEnhance_Data/target_1176_16bit_3D/')
        #'/media/santeri/data2/BoneEnhance_Data/Skyscan_1176_50um')
        '/media/santeri/data2/BoneEnhance_Data/target_1176_16bit_full_antialiasing/')
        #'/media/dios/dios3/Numminen/Reconstructions')
    parser.add_argument('--images_save', type=Path, default=\
        #'/media/santeri/data2/BoneEnhance_Data/input_1176_16bit_full/')
        '/media/santeri/data2/BoneEnhance_Data/target_1176_16bit_full_3D_antialiasing/')
    parser.add_argument('--res_out', type=int, default=200, help='Target resolution for training data (in µm)')
    parser.add_argument('--completed', type=int, default=0, help='Samples already processed and skipped.')
    parser.add_argument('--crop_size', type=list, default=[1500, 1500, 50], help='Size of one training patch')
    parser.add_argument('--crop', type=bool, default=False, help='Split in patches')
    parser.add_argument('--plot', type=bool, default=False, help='Plot images')
    parser.add_argument('--resample', type=bool, default=False, help='Resample or keep image dimensions')
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.tif')
    parser.add_argument('--sigma', type=float, default=4, help='Standard deviation of gaussian blur (antialiasing).')
    parser.add_argument('--hdf5', type=bool, default=True, help='Save as 3D data (True) or a stack of 2D images.')

    args = parser.parse_args()

    # Save path
    args.images_save.mkdir(exist_ok=True)

    # List samples
    samples = os.listdir(args.images_loc)
    # Include directories and .h5 files
    samples = [
        name for name in samples
        if os.path.isdir(os.path.join(args.images_loc, name)) or name.endswith('.h5')]
    samples.sort()

    if args.completed > 0:
        samples = samples[args.completed:]

    # Resample datasets, create 3D stack
    for sample in samples:
        print(f'Processing sample: {sample}')
        #try:
        # Load log file to check resolution
        im_path = args.images_loc / sample
        #log = load_logfile(str(im_path))
        #res = float(log['Image Pixel Size (um)'])
        res = 50

        # Scale factors and scaled crops
        factor = args.res_out / res
        crop_large = np.floor(np.array(args.crop_size) * factor).astype('uint32')

        # Load images
        if sample.endswith('.h5'):
            with h5py.File(im_path, 'r') as f:
                stack = f['data'][:]
        else:
            stack, files = load(im_path, axis=(1, 2, 0))

        # Reshape to have the smallest axis last
        stack = np.moveaxis(stack, np.argmin(stack.shape), -1)

        data = np.memmap(str(args.images_save / 'temp_array.dat'), dtype=np.uint16, mode='w+', shape=stack.shape)
        data[:, :, :] = stack
        del stack

        # Visualize full stack
        #print_orthogonal(data, res=res/1e3, invert=True, cbar=True, scale_factor=10)

        # Create Save directory
        #(images_save / sample).mkdir(exist_ok=True)

        # Crop small samples and scale to proper size
        if args.crop:
            n_crops = data.shape // crop_large
            n_crops = np.maximum(n_crops, 1)
            crop_begin = (data.shape - n_crops * crop_large) // 2
            crop_begin = np.maximum(crop_begin, 0)
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
                        if args.resample:
                            data_out = downscale_image(data_out,
                                                       im_size=(data_out.shape[0] // factor, data_out.shape[1] // factor,
                                                                data_out.shape[2] // factor),
                                                       add_noise='poisson')

                            # Make sure rescaled data is divisible by 4
                            new_shape = tuple((s // 4) * 4 for s in data_out.shape)
                            data_out = data_out[:new_shape[0], :new_shape[1], :new_shape[2]]

                        # Save the cropped volume to hdf5
                        if args.hdf5:
                            fname = str(args.images_save / f'{sample}_{str(x).zfill(3)}{str(y).zfill(3)}{str(z).zfill(3)}.h5')
                            with h5py.File(fname, 'w') as f:
                                f.create_dataset('data', data=data_out)
                        else:
                            fname = Path(f'{sample}_{str(x).zfill(3)}{str(y).zfill(3)}{str(z).zfill(3)}')

                            save(str(args.images_save / fname), fname.name, data_out, dtype=args.dtype, verbose=False)
        else:
            # Crop is now in resolution "res_out"
            #data_out = resize(data, args.crop_size, order=0, anti_aliasing=True, preserve_range=True,
            #                  anti_aliasing_sigma=args.sigma).astype('uint8')
            #data_out = downscale_image(data, im_size=(data.shape[0] // factor, data.shape[1] // factor, data.shape[2] // factor),
            #                add_noise=None)

            if args.plot:
                print_orthogonal(data, savepath=str(args.images_save / 'Visualizations' / f'{sample}_original'))
                # Create histograms for data and data_out
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                ax1.hist(data.flatten(), bins=256, color='blue', alpha=0.7)
                ax1.set_xlabel('Pixel Intensity')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Original Data Histogram')
                ax1.grid(True, alpha=0.3)

            # Calculate new shape
            #new_shape = tuple(int(s / factor) for s in data.shape)
            #indices = [np.linspace(0, data.shape[i] - 1, new_shape[i]).astype(int) for i in range(3)]
            if args.resample:
                data = downscale_image(data, im_size=(data.shape[0] // factor, data.shape[1] // factor,
                                                      #data.shape[2] // factor),
                                                      data.shape[2]),
                                       sigma=args.sigma, add_noise='poisson', blur=True)

            #data_out = zoom(data, factor, order=1, mode='nearest', prefilter=True)

            if args.plot:
                print_orthogonal(data, savepath=str(args.images_save / 'Visualizations' / f'{sample}_rescaled'))
                ax2.hist(data.flatten(), bins=256, color='red', alpha=0.7)
                ax2.set_xlabel('Pixel Intensity')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Rescaled Data Histogram')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(str(args.images_save / 'Visualizations' / f'{sample}_histograms.png'), dpi=150)
                plt.close()

            # Save the cropped volume to hdf5
            if args.hdf5:
                if sample.endswith('.h5'):
                    fname = str(args.images_save / f'{sample}')
                else:
                    fname = str(args.images_save / f'{sample}.h5')
                with h5py.File(fname, 'w') as f:
                    f.create_dataset('data', data=data)
            else:
                fname = Path(sample)

                save(str(args.images_save / fname), fname.name, data, dtype=args.dtype, verbose=False)

        #except (ValueError, FileNotFoundError):
        #    print(f'Error in sample {sample}')
        #    continue
