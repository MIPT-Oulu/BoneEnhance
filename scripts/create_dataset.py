import os
from pathlib import Path
import cv2
from BoneEnhance.components.training.session import init_experiment
from BoneEnhance.components.utilities.main import load, save, print_orthogonal
from scipy.ndimage import zoom

if __name__ == "__main__":
    # Initialize experiment
    args, config, _ device = init_experiment()
    images_loc = Path('/media/santeri/data/BoneEnhance/Data/target')
    #images_loc = Path('/media/santeri/data/BoneEnhance/Data/Test set (KP02)/target')

    images_save = Path('/media/santeri/data/BoneEnhance/Data/target_ds')

    images_save.mkdir(exist_ok=True)

    #subdir = 'trabecular_data/Binned4x/bonemask'
    resample = True
    #factor = 200/2.75
    factor = 16
    k = 5
    #n_slices = 100

    # Resample large number of slices
    samples = os.listdir(images_loc)
    samples = [name for name in samples if os.path.isdir(os.path.join(images_loc, name))]
    samples.sort()
    #samples = samples[25:]
    for sample in samples:
        print(f'Processing sample: {sample}')
        try:
            if resample:  # Resample slices
                im_path = images_loc / sample #/ subdir

                data, files = load(im_path, axis=(0, 1, 2))  #axis=(1, 2, 0))

                (images_save / sample).mkdir(exist_ok=True)

                for i in range(data.shape[0]):
                    image = data[i]
                    if factor != 1:
                        resize_target = (image.shape[1] // factor, image.shape[0] // factor)

                        # Antialiasing
                        image = cv2.GaussianBlur(image, ksize=(k, k), sigmaX=0)
                        image = cv2.resize(image.copy(), resize_target, interpolation=cv2.INTER_NEAREST)

                    cv2.imwrite(str(images_save / sample / files[i]), image)



                #save(str(images_save / sample), sample + '_cor', data[:, :, :], dtype='.bmp')
            else:  # Move segmented samples to training data
                im_path = str(images_loc / sample)
                files = os.listdir(im_path)
                if subdir in files:
                    data, _ = load(im_path, axis=(1, 2, 0))

                    save(str(images_save / sample), sample + '_cor', data, dtype='.bmp')

        except (ValueError, FileNotFoundError):
            print(f'Error in sample {sample}')
            continue
