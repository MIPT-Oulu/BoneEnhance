import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from BoneEnhance.components.models.wgan import WGAN_VGG_FeatureExtractor
from BoneEnhance.components.models import Vgg16, load_models
from BoneEnhance.components.training.initialize_weights import InitWeight, init_weight_normal
from BoneEnhance.components.utilities import convert_3d_tensor_to_random_2d
from random import uniform


class PerceptualLoss(nn.Module):
    """
    Calculates the perceptual loss based on difference between the VGG19 features.

    Possible layers to be compared: ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'].
    A single layer can be provided or a list in which case all losses are added together.
    """

    def __init__(self, criterion=nn.L1Loss(), compare_layer=None, mean=None, std=None, imagenet_normalize=True,
                 gram=True, plot=True, vol=False, zeros=True, gpus=1, rgb=True, crop=None):
        super(PerceptualLoss, self).__init__()
        # vol=False  # Test 2D loss on 3D model
        if compare_layer is None:
            self.feature_extractor = WGAN_VGG_FeatureExtractor()
        elif isinstance(compare_layer, str):
            # Load first fold of the trained autoencoder
            self.feature_extractor = load_models(compare_layer, crop, vol=vol, rgb=False, fold=0, gpus=gpus)
        else:
            self.feature_extractor = Vgg16(vol=vol, zeros=zeros)

            # Weight init
            #init = InitWeight(init_weight_normal, [0.0, 0.02], type='conv')
            #self.feature_extractor.apply(init)

        self.p_criterion = criterion
        self.compare_layer = compare_layer
        self.imagenet_normalize = imagenet_normalize
        self.imagenet_mean = (0.485, 0.456, 0.406)
        self.imagenet_std = (0.229, 0.224, 0.225)
        self.mean = mean
        self.std = std
        self.calculate_gram = gram
        self.plot = plot
        self.vol = vol
        self.rgb = rgb

    def forward(self, logits, targets):

        # Convert to RGB
        if logits.size(1) != 3 and self.rgb:
            if logits.dim() == 5:
                logits = logits.repeat(1, 3, 1, 1, 1)
                targets = targets.repeat(1, 3, 1, 1, 1)
            else:
                logits = logits.repeat(1, 3, 1, 1,)
                targets = targets.repeat(1, 3, 1, 1)

        # Use 2D loss for 3D data
        if not self.vol and logits.dim() == 5:
            logits, targets = convert_3d_tensor_to_random_2d([logits, targets], mag=1)

        # Scale to imagenet mean and std
        if self.imagenet_normalize:
            for channel in range(len(self.imagenet_mean)):
                logits[:, channel, :, :] -= self.imagenet_mean[channel]
                targets[:, channel, :, :] -= self.imagenet_mean[channel]
                logits[:, channel, :, :] /= self.imagenet_std[channel]
                targets[:, channel, :, :] /= self.imagenet_std[channel]

        # Comparison for multiple layers or single activation
        if self.compare_layer is None:
            pred_feature = self.feature_extractor(logits)
            target_feature = self.feature_extractor(targets)
            loss = self.p_criterion(pred_feature, target_feature)
        else:
            # Obtain features from different layers
            pred_feature = self.feature_extractor(logits)
            target_feature = self.feature_extractor(targets)

            if not isinstance(pred_feature, dict):
                pred_feature = {self.compare_layer: pred_feature}
                target_feature = {self.compare_layer: target_feature}

            # Plot feature maps
            if self.plot and uniform(0, 1) >= 0.95:
                self.plot_features(logits, targets, pred_feature, target_feature, num=1, f_map=24)

            # Calculate gram matrices
            if self.calculate_gram:
                for key in pred_feature:
                    if self.vol:
                        pred_feature[key] = self.gram_3d(pred_feature[key])
                        target_feature[key] = self.gram_3d(target_feature[key])
                    else:
                        pred_feature[key] = self.gram(pred_feature[key])
                        target_feature[key] = self.gram(target_feature[key])

            # Calculate loss layer by layer
            layer = self.compare_layer
            loss = 0
            for key in pred_feature:
                loss += self.p_criterion(pred_feature[key], target_feature[key])
            loss /= len(pred_feature)

            # Weight the gram matrix loss to a reasonable range
            if self.calculate_gram and self.vol:
                loss *= 1e7
            elif self.calculate_gram:
                loss *= 1e5
            elif self.vol:
                loss *= 1#1e-7
        return loss

    @staticmethod
    def gram(x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    @staticmethod
    def gram_3d(x):
        (bs, ch, d, h, w) = x.size()
        f = x.view(bs, ch, d * w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * d * h * w)
        return G

    @staticmethod
    def plot_features(logits, targets, pred_feature, target_feature, num=1, f_map=24):
        for i in range(num):
            fig, axs = plt.subplots(2, 5)

            if logits.dim() == 5:
                d = logits.size(2) // 2
                # Plot input
                axs[0, 0].imshow(logits.detach().cpu()[i, 0, d, :, :], cmap='gray')
                axs[1, 0].imshow(targets.detach().cpu()[i, 0, d, :, :], cmap='gray')
            else:
                # Plot input
                axs[0, 0].imshow(logits.detach().cpu()[i, 0, :, :], cmap='gray')
                axs[1, 0].imshow(targets.detach().cpu()[i, 0, :, :], cmap='gray')

            axs[0, 0].set_title('Prediction')
            axs[1, 0].set_title('Target')

            # Plot activations
            for key, j in zip(pred_feature.keys(), range(1, 5)):

                if logits.dim() == 5:
                    d = pred_feature[key].size(2) // 2
                    axs[0, j].imshow(pred_feature[key].detach().cpu()[i, f_map, d, :, :], cmap='gray')
                    axs[1, j].imshow(target_feature[key].detach().cpu()[i, f_map, d, :, :], cmap='gray')

                else:
                    axs[0, j].imshow(pred_feature[key].detach().cpu()[i, f_map, :, :], cmap='gray')
                    axs[1, j].imshow(target_feature[key].detach().cpu()[i, f_map, :, :], cmap='gray')

                axs[0, j].set_title(key)
                axs[1, j].set_title(key)
            fig.show()


class TotalVariationLoss(nn.Module):
    """Total variation loss in 2D and 3D."""
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor):
        tv = 0
        if len(input.size()) == 5:
            # Width
            tv += torch.pow(input[:, :, 1:, :, :] - input[:, :, :-1, :, :], 2).sum()
            # Height
            tv += torch.pow(input[:, :, :, 1:, :] - input[:, :, :, :-1, :], 2).sum()
            # Depth
            tv += torch.pow(input[:, :, :, :, 1:] - input[:, :, :, :, :-1], 2).sum()
        else:
            # Width
            tv += torch.pow(input[:, :, 1:, :] - input[:, :, :-1, :], 2).sum()
            # Height
            tv += torch.pow(input[:, :, :, 1:] - input[:, :, :, :-1], 2).sum()
        return tv / input.numel()
