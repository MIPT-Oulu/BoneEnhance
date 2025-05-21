import torch
import torch.nn as nn
from glob import glob

from ..training.session import load_model


class InferenceModel(nn.Module):
    def __init__(self, models_list, sigmoid=False):
        super(InferenceModel, self).__init__()
        self.n_folds = len(models_list)
        modules = {}
        for idx, m in enumerate(models_list):
            modules[f'fold_{idx}'] = m

        self.__dict__['_modules'] = modules
        self.sigmoid = sigmoid

    def forward(self, x):
        res = 0
        preds = []
        for idx in range(self.n_folds):
            fold = self.__dict__['_modules'][f'fold_{idx}']

            if self.sigmoid:
                pred = fold(x).sigmoid()
            # Scale the tanh activation back to 0 and 1 during inference
            else:
                pred = (fold(x) + 1) / 2
            res += pred
            preds.append(pred)

        return res / self.n_folds


def load_and_list_models(model_path, config, n_gpus=1, magnification=4, fold=None):
    # Load models
    if fold is not None:
        models = glob(model_path + f'/*fold_{fold}*.pth')
    else:
        models = glob(model_path + '/*fold_*.pth')
    models.sort()

    # List the models
    model_list = []
    for fold in range(len(models)):
        # Load model architecture
        model = load_model(config)

        # Parallel GPU processing
        if n_gpus > 1:
            model = nn.DataParallel(model)

        # Load saved weights for the model architecture
        model.load_state_dict(torch.load(models[fold]))
        model_list.append(model)

    return model_list
