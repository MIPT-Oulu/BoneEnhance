import os
import numpy as np
from pathlib import Path
import cv2
import h5py
from bone_enhance.training.session import init_experiment
from bone_enhance.utilities.main import load, save, print_orthogonal
from scipy.ndimage import zoom, median_filter
from skimage.transform import resize

if __name__ == "__main__":
    # Initialize experiment
    args, config, _, device = init_experiment()
    snap = '2021_08_04_09_10_16_2D_ssim_IVD_4x_seed42'
    #images_loc = Path('/media/dios/kaappi/Santeri/BoneEnhance/Clinical data')
    #images_loc = Path('/media/santeri/data/BoneEnhance/Data/target_1176_HR_2D')
    #images_loc = Path('/media/santeri/data/BoneEnhance/Data/MRI_IVD/9.4T MRI Scans')
    #images_loc = Path(f'../../Data/dental/Hampaat_rec')
    images_loc = Path(f'../../Data/predictions_3D_clinical/dental_experiments/test/')
    #images_loc = Path(f'../../Data/dental/')
    #images_loc = Path('/media/santeri/data/BoneEnhance/Data/MRI_IVD/3T scans dicom')

    images_save = Path('/media/santeri/data/BoneEnhance/Data/target_IVD_2D_HR')
    #images_save = Path(f'../../Data/dental/Hampaat_dataset')
    images_save = Path(f'../../Data/predictions_3D_clinical/dental_experiments/test_dcm')

    images_save.mkdir(exist_ok=True)

    #subdir = 'trabecular_data/Binned4x/bonemask'
    resample = False
    normalize = False
    factor = 50/19.8
    #factor = 4
    #factor_slice = 1361.4/90
    sigma = 1
    dtype = '.dcm'
    k = 3
    hdf5 = False

    # Resample large number of slices
    samples = os.listdir(images_loc)
    #samples = [name for name in samples if os.path.isdir(os.path.join(images_loc, name))]
    samples.sort()
    #samples = [samples[6]]
    samples = [samples[0]]
    if 'visualizations' in samples:
        samples.remove('visualizations')
    for sample in samples:
        print(f'Processing sample: {sample}')
        try:
            if resample:  # Resample slices
                if sample.endswith('.h5'):
                    with h5py.File(str(images_loc / sample), 'r') as f:
                        data = f['data'][:]
                else:
                    data, files = load(str(images_loc / sample), rgb=False, axis=(1, 2, 0))


                # Upscale
                # Make MRI data "isotropic"
                #new_size = (data.shape[0], data.shape[1], int(data.shape[2] * factor_slice))
                #data = resize(data, new_size, order=3, preserve_range=True)
                #new_size = (int(data.shape[0] * factor), int(data.shape[1] * factor), int(data.shape[2]))
                #data = resize(data, new_size, order=3, preserve_range=True)
                #data = (data / 2 ** 8).astype('uint8')
                # Downscale
                new_size = (data.shape[0] // factor, data.shape[1] // factor, data.shape[2] // factor)
                data = resize(data, new_size, order=0, anti_aliasing=True, preserve_range=True, anti_aliasing_sigma=sigma)
                data = median_filter(data, size=3)
                if normalize:
                    min_data = np.min(data)
                    max_data = np.max(data)
                    data = (data - min_data) / (max_data - min_data) * 255

                if hdf5:
                    fname = str(images_save / f'{sample}.h5')
                    with h5py.File(fname, 'w') as f:
                        f.create_dataset('data', data=data)
                else:
                    save(str(images_save / sample), sample, data, dtype=dtype)

                #(images_save / sample).mkdir(exist_ok=True)

                #for i in range(data.shape[0]):
                #    image = data[i]
                #    if factor != 1:
                #        resize_target = (image.shape[1] // factor, image.shape[0] // factor)

                        # Antialiasing
                        #image = cv2.GaussianBlur(image, ksize=(k, k), sigmaX=0)
                        #image = cv2.resize(image.copy(), resize_target, interpolation=cv2.INTER_NEAREST)

                    #cv2.imwrite(str(images_save / sample / files[i]), image)



                #save(str(images_save / sample), sample + '_cor', data[:, :, :], dtype='.bmp')
            else:  # Move segmented samples to training data
                if sample.endswith('.h5'):
                    with h5py.File(str(images_loc / sample), 'r') as f:
                        data = f['data'][:]
                    sample = sample[:-3]
                else:
                    data, files = load(str(images_loc / sample), rgb=False, axis=(1, 2, 0))#, dicom=True)

                save(str(images_save / sample), sample, data, dtype=dtype)

        except (ValueError, FileNotFoundError):
            print(f'Error in sample {sample}')
            continue
