import torch
import torch.nn as nn
from glob import glob

from segmentation_models_pytorch import Unet
from collagen.modelzoo.segmentation import EncoderDecoder


class InferenceModel(nn.Module):
    def __init__(self, models_list):
        super(InferenceModel, self).__init__()
        self.n_folds = len(models_list)
        modules = {}
        for idx, m in enumerate(models_list):
            modules[f'fold_{idx}'] = m

        self.__dict__['_modules'] = modules

    def forward(self, x):
        res = 0
        for idx in range(self.n_folds):
            fold = self.__dict__['_modules'][f'fold_{idx}']
            res += fold(x).sigmoid()

        return res / self.n_folds


def load_models(model_path, config, n_gpus=1, unet=True):
    # Load models
    models = glob(model_path + '/*fold_*.pth')
    models.sort()

    # List the models
    model_list = []
    for fold in range(len(models)):
        if unet and n_gpus > 1:
            model = nn.DataParallel(
                Unet(config['model']['backbone'], encoder_weights="imagenet", activation='sigmoid'))
        elif unet:
            model = Unet(config['model']['backbone'], encoder_weights="imagenet", activation='sigmoid')
        elif n_gpus > 1:
            model = nn.DataParallel(EncoderDecoder(**config['model']))
        else:
            model = EncoderDecoder(**config['model'])
        model.load_state_dict(torch.load(models[fold]))
        model_list.append(model)

    return model_list
