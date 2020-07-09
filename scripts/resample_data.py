import os
from pathlib import Path
from BoneEnhance.components.training.session import init_experiment
from BoneEnhance.components.utilities.main import load, save, print_orthogonal
from scipy.ndimage import zoom

if __name__ == "__main__":
    # Initialize experiment
    args, config, device = init_experiment()
    images_loc = Path('/media/dios/kaappi/Sakke/Saskatoon/µCT/Recs_bone')

    images_save = Path('/media/santeri/data/BoneEnhance/Data/target')

    images_save.mkdir(exist_ok=True)

    subdir = ''
    resample = True
    factor = 200/2.75
    n_slices = 100

    # Resample large number of slices
    samples = os.listdir(images_loc)
    samples = [name for name in samples if os.path.isdir(os.path.join(images_loc, name))]
    samples.sort()
    for sample in samples:
        print(f'Processing sample: {sample}')
        try:
            if resample:  # Resample slices
                im_path = images_loc / sample

                data, _ = load(im_path, axis=(1, 2, 0))

                data_resampled = zoom(data, (1, 1, 1 / factor), order=0)  # nearest interpolation
                #print_orthogonal(data_resampled)

                save(str(images_save / sample), sample, data_resampled[:, :, :n_slices], dtype='.bmp')
            else:  # Move segmented samples to training data
                im_path = str(images_loc / sample)
                files = os.listdir(im_path)
                if subdir in files:
                    data, _ = load(im_path, axis=(1, 2, 0))

                    save(str(images_save / sample), sample, data, dtype='.bmp')

        except ValueError:
            print(f'Error in sample {sample}')
            continue
