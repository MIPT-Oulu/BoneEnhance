import os
from pathlib import Path
from bone_enhance.training.session import init_experiment
from bone_enhance.utilities.main import load, save, print_orthogonal
from bone_enhance.inference import largest_object
from scipy.ndimage import zoom

if __name__ == "__main__":
    # Initialize experiment
    args, config, _, device = init_experiment()
    images_loc = Path('/media/dios/kaappi/Santeri/Vessels')

    images_save = Path('/media/santeri/data/RabbitSegmentation/Vessels/Processed')

    images_save.mkdir(exist_ok=True)

    subdir = ''
    resample = True

    # Resample large number of slices
    samples = os.listdir(images_loc)
    samples.sort()
    samples = ['Vessel']
    #samples = samples[25:]
    for sample in samples:
        print(f'Processing sample: {sample}')


        data, _ = load(str(images_loc), axis=(1, 2, 0))
        print_orthogonal(data, res=3, scale_factor=1000)
        data = largest_object(data > 0, area_limit=100).astype('bool')
        print_orthogonal(data, res=3, scale_factor=1000)
        save(str(images_save), sample, data * 255, dtype='.bmp')

