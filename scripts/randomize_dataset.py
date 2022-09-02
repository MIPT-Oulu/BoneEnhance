import os
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from bone_enhance.training.session import init_experiment
from bone_enhance.utilities.main import load, save, print_orthogonal
from skimage.transform import resize
from scipy.ndimage import zoom

if __name__ == "__main__":
    # Initialize experiment
    args, config, _, device = init_experiment()


    images_loc = Path(f'/media/santeri/Transcend/Super-resolution png')
    #images_loc = Path(f'/media/santeri/data/BoneEnhance/Data/dental')
    table_path = Path(f'/media/santeri/Transcend/Randomize_patients.xlsx')

    images_save = Path(f'/media/santeri/Transcend/randomized_dataset')

    images_save.mkdir(exist_ok=True)

    resample = False
    normalize = False
    factor = 4
    sigma = 1
    dtype = '.dcm'
    k = 3
    hdf5 = False

    table_keys = pd.read_excel(table_path, engine='openpyxl')
    table_keys = table_keys.iloc[:36, :3].values.astype('uint32')

    # Resample large number of slices
    models = os.listdir(images_loc)
    for table_idx, idx_random in enumerate(table_keys[:, 2]):
        model = models[table_keys[table_idx, 1] - 1]

        samples = os.listdir(images_loc / model)
        #samples = [name for name in samples if os.path.isdir(os.path.join(images_loc, name))]
        samples.sort()
        #samples = [samples[6]]
        #samples = samples[16:25]
        if 'visualizations' in samples:
            samples.remove('visualizations')

        sample = samples[table_keys[table_idx, 0] - 1]
        print(f'Processing sample: {sample}, model {model}')
        #try:

        #randomized_index = np.where((table_keys[:, :2] == (1, 1)).all(axis=1))
        #randomized_index = table_keys[randomized_index, 2][0][0]

        data, files = load(str(images_loc / model / sample), rgb=False, axis=(1, 2, 0))#, dicom=True)

        # Interpolate
        #data = zoom(data, (factor, factor, factor), order=3)  # Tricubic

        save(str(images_save / f'CBCT_{idx_random}'), f'CBCT_{idx_random}', data, dtype=dtype)

        #except (ValueError, FileNotFoundError):
        #    print(f'Error in sample {sample}')
        #    continue
