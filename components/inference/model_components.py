import torch
import torch.nn as nn
from glob import glob

from BoneEnhance.components.models import EnhanceNet, ConvNet, PerceptualNet, SREncoderDecoder
from collagen.modelzoo.segmentation import EncoderDecoder


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
            pred = fold(x)
            if self.sigmoid:
                pred = pred.sigmoid()
            res += pred
            preds.append(pred)

        return res / self.n_folds


def load_models(model_path, config, n_gpus=1, magnification=4, fold=None):
    # Load models
    if fold is not None:
        models = glob(model_path + f'/*fold_{fold}*.pth')
    else:
        models = glob(model_path + '/*fold_*.pth')
    models.sort()
    vol = len(config.training.crop_small) == 3
    architecture = config.training.architecture

    # List the models
    model_list = []
    for fold in range(len(models)):

        if architecture == 'srencoderdecoder':
            config.model.magnification = config.training.magnification
            model = SREncoderDecoder(**config['model'])
        elif architecture == 'encoderdecoder':
            model = EncoderDecoder(**config['model'])
        elif architecture == 'enhance':
            model = EnhanceNet(config.training.crop_small, config.training.magnification,
                               activation=config.training.activation,
                               add_residual=config.training.add_residual,
                               upscale_input=config.training.upscale_input)
        elif architecture == 'convnet':
            model = ConvNet(config.training.magnification,
                            activation=config.training.activation,
                            upscale_input=config.training.upscale_input,
                            n_blocks=config.training.n_blocks,
                            normalization=config.training.normalization)
        elif architecture == 'perceptualnet':
            model = PerceptualNet(config.training.magnification,
                                  resize_convolution=config.training.upscale_input,
                                  norm=config.training.normalization,
                                  vol=vol, rgb=config.training.rgb)
        else:
            raise Exception('Model architecture unavailable.')

        if n_gpus > 1:
            model = nn.DataParallel(model)

        model.load_state_dict(torch.load(models[fold]))
        model_list.append(model)

    return model_list
