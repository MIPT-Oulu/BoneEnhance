import os
from pathlib import Path
from BoneEnhance.components.training.session import init_experiment

if __name__ == "__main__":
    # Initialize experiment
    args, _, _ = init_experiment()
    base_path = args.data_location
    #base_path = Path('/media/dios/dios2/RabbitSegmentation/µCT/Should_be_resegmented_with_neural_network/New Manual CC segmentation')
    masks_loc = base_path / 'target'
    images_loc = base_path / 'input'

    # List files
    samples = os.listdir(masks_loc)
    samples.sort()
    for sample in samples:
        im_path = images_loc / Path(sample)
        mask_path = masks_loc / Path(sample)
        if '_Rec' in str(mask_path):
            os.rename(str(mask_path), str(mask_path)[:-4])
        images = list(map(lambda x: x, im_path.glob('**/*[0-9].[pb][nm][gp]')))
        masks = list(map(lambda x: x, mask_path.glob('**/*[0-9].[pb][nm][gp]')))
        images.sort()
        masks.sort()
        for slice in range(len(images)):
            # Image
            new_name = images_loc / sample / Path(sample + f'_{str(slice).zfill(8)}{str(images[slice])[-4:]}')
            os.rename(images[slice], new_name)
            new_name = masks_loc / sample / Path(sample + f'_{str(slice).zfill(8)}{str(masks[slice])[-4:]}')
            os.rename(masks[slice], new_name)
