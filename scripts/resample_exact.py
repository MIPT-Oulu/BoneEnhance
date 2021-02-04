import cv2
import os
import numpy as np
from pathlib import Path
from BoneEnhance.components.training.session import init_experiment
from BoneEnhance.components.utilities.main import load, save, print_orthogonal
from scipy.ndimage import zoom
from PIL import Image

if __name__ == "__main__":
    # Initialize experiment
    #images_loc = Path('/media/dios/kaappi/Sakke/Saskatoon/µCT/Recs_bone')
    images_loc = Path('/media/santeri/data/BoneEnhance/Data/target_binary')
    input_loc = Path('/media/santeri/data/BoneEnhance/Data/input_original')

    images_save = Path('/media/santeri/data/BoneEnhance/Data/target_binary_resampled')
    images_save.mkdir(exist_ok=True)

    # List samples
    samples = os.listdir(images_loc)
    samples = [name for name in samples if os.path.isdir(os.path.join(images_loc, name))]
    samples_input = os.listdir(input_loc)
    samples_input = [name for name in samples_input if os.path.isdir(os.path.join(input_loc, name))]
    assert len(samples) == len(samples_input)

    samples.sort()
    samples_input.sort()
    #samples = samples[30:]
    #samples_input = samples_input[30:]
    for i in range(len(samples)):
        print(f'Processing sample: {samples[i]}')

        input_path = input_loc / samples_input[i]
        im_path = images_loc / samples[i]

        # (1, 2, 0) = Original dimension
        data_input, _ = load(input_path, axis=(1, 2, 0))  # axis=(1, 2, 0))
        data, _ = load(im_path, axis=(1, 2, 0))  # axis=(1, 2, 0))

        data_resample = np.zeros((data.shape[0], data.shape[1], data_input.shape[2]))

        for slice in range(data.shape[1]):
            data_resample[:, slice, :] = cv2.resize(data[:, slice, :], (data_input.shape[2], data.shape[0]))

        save(str(images_save / samples[i]), samples[i], data_resample[:, :, :], dtype='.bmp')
