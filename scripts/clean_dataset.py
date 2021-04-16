import os
from pathlib import Path
from bone_enhance.training.session import init_experiment
import shutil

if __name__ == "__main__":
    # Initialize experiment
    args, _, _, _ = init_experiment()
    base_path = args.data_location
    images_loc = base_path / 'input_original'
    #images_loc = base_path / 'input'

    # List sample folders
    samples = os.listdir(images_loc)
    samples.sort()
    for sample in samples:
        im_path = images_loc / Path(sample)
        files = os.listdir(im_path)

        # Find directories inside sample folder
        junk = [name for name in files if os.path.isdir(os.path.join(im_path, name))]
        # Find specific file types
        junk.extend(list(map(lambda x: x.name, im_path.glob('**/*.tf'))))
        junk.extend(list(map(lambda x: x.name, im_path.glob('**/*.roi'))))
        junk.extend(list(map(lambda x: x.name, im_path.glob('**/*.stl'))))
        # Find files with specific name
        junk.extend(list(map(lambda x: x.name, im_path.glob('**/*ctan*'))))
        junk.extend(list(map(lambda x: x.name, im_path.glob('**/*batman*'))))
        junk.extend(list(map(lambda x: x.name, im_path.glob('**/*spr*'))))

        # Remove
        for file in junk:
            f = os.path.join(im_path, file)

            if os.path.isdir(f):  # Recursive remove for folders
                shutil.rmtree(f)
            else:
                os.remove(f)
