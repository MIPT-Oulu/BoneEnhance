import os
from pathlib import Path
from BoneEnhance.components.training.session import init_experiment

if __name__ == "__main__":
    # Initialize experiment
    args, _, _ = init_experiment()
    base_path = args.data_location
    images_loc = base_path / 'Test set (KP02)' / 'target'
    #images_loc = base_path / 'input'

    # List files
    samples = os.listdir(images_loc)
    samples.sort()
    for sample in samples:
        im_path = images_loc / Path(sample)
        if '_Rec' in str(im_path):
            os.rename(str(im_path), str(im_path)[:-4])
        images = list(map(lambda x: x, im_path.glob('**/*[0-9].[pb][nm][gp]')))
        images.sort()
        for slice in range(len(images)):
            # Image
            new_name = images_loc / sample / Path(sample + f'_{str(slice).zfill(8)}{str(images[slice])[-4:]}')
            os.rename(images[slice], new_name)
