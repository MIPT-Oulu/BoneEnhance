# BoneEnhance
Methods for improving image quality on computed tomography -imaged bone

(c) Santeri Rytky, University of Oulu, 2021-2023

![Analysis pipeline](https://github.com/MIPT-Oulu/BoneEnhance/blob/master/images/Flowchart.png)

## Background
Clinical cone-beam computed tomography (CBCT) devices are limited to imaging tissues of submillimeter scale. This repository is used to create super-resolution models trained on high-resolution micro-computed tomography (µCT) images. For a detailed description of the method, refer to the publication by Rytky SJO et al.

## Prerequisites
- [Anaconda installation](https://docs.anaconda.com/anaconda/install/) 
```
git clone https://github.com/MIPT-Oulu/BoneEnhance.git
cd BoneEnhance
conda env create -f environment.yml
```

## Usage

### Model training

- Create a training dataset: Use the script `create_training_data.py` to simulate image pairs from high-resolution µCT scans. Set the data and save paths as well as resolution and save parameters at the beginning of the script. 

- Set the path name for training data in [session.py](../master/rabbitccs/training/session.py) (`init_experiment()` function)

- Create a configuration file to the `experiments/run` folder. Example experiments are included in the folder. 
All experiments are conducted subsequently during training.

```
conda activate bone-enhance-env
python scripts/train.py
```

### Inference

For 2D prediction, use `inference_tiles_2d.py`. For 3D data, use `inference_tiles_3d.py`. 
Running `inference_tiles_large_3d.py` allows to run inference on larger samples, and merge on CPU. Using `inference_tiles_large_pseudo3d.py` allows merging 2D predictions on orthogonal planes.
Update the `snap` variable, image path and save directory.

## License

This software is distributed under the MIT License.
