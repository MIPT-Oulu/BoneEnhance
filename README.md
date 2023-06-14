# BoneEnhance
Methods for improving image quality on computed tomography -imaged bone

(c) Santeri Rytky, University of Oulu, 2020-2021

![Analysis pipeline](https://github.com/MIPT-Oulu/BoneEnhance/blob/master/images/Flowchart.png)

## Background

## Prerequisites
- [Anaconda installation](https://docs.anaconda.com/anaconda/install/) 
```
git clone https://github.com/MIPT-Oulu/BoneEnhance.git
cd BoneEnhance
conda env create -f environment.yml
```

## Usage

### Model training

- Create a training dataset: input images in folder `images` and target masks in `masks`. 
For 2D data, just add the images to the corresponding folders, making sure that the image and mask names match.
For 3D data, create a subfolder for each scan (sample name for the folder), and include the slices in the subfolder.

- Set the path name for training data in [session.py](../master/rabbitccs/training/session.py) (`init_experiment()` function)

- Create a configuration file to the `experiments/run` folder. Four example experiments are included. 
All experiments are conducted subsequently during training.

```
conda activate bone-enhance-env
python scripts/train.py
```

### Inference

For 2D prediction, use `inference_tiles_2d.py`. For 3D data, use `inference_tiles_3d.py`. 
Running `inference_tiles_large_3d.py` allows to run inference on larger samples, and merge on CPU.
Update the `snap` variable, image path and save directory.

## License

This software is distributed under the MIT License.