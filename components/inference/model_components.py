import torch
import torch.nn as nn
from glob import glob

from BoneEnhance.components.training.models import EnhanceNet


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


def load_models(model_path, config, n_gpus=1, magnification=4):
    # Load models
    models = glob(model_path + '/*fold_*.pth')
    models.sort()

    # List the models
    model_list = []
    for fold in range(len(models)):
        if n_gpus > 1:
            model = nn.DataParallel(
                EnhanceNet(config['training']['crop_small'], magnification))
        else:
            model = EnhanceNet(config['training']['crop_small'], magnification)
        model.load_state_dict(torch.load(models[fold]))
        model_list.append(model)

    return model_list
