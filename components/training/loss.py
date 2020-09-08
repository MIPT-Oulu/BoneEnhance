from torch import nn
from BoneEnhance.components.models.wgan import WGAN_VGG_FeatureExtractor
from BoneEnhance.components.models.perceptual import Vgg16


class PerceptualLoss(nn.Module):
    """
    Calculates the perceptual loss based on difference between the VGG19 features.

    Possible layers to be compared: ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'].
    A single layer can be provided or a list in which case all losses are added together.
    """

    def __init__(self, criterion=nn.L1Loss(), compare_layer=None, mean=None, std=None):
        super(PerceptualLoss, self).__init__()
        if compare_layer is None:
            self.feature_extractor = WGAN_VGG_FeatureExtractor()
        else:
            self.feature_extractor = Vgg16()
            self.feature_extractor.eval()
        self.p_criterion = criterion
        self.compare_layer = compare_layer
        self.imagenet_mean = (0.485, 0.456, 0.406)
        self.imagenet_std = (0.229, 0.224, 0.225)
        self.mean = mean
        self.std = std

    def forward(self, logits, targets):

        # TODO: Unnormalize
        #"""
        if self.mean is not None and self.std is not None:
            for channel in range(len(self.imagenet_mean)):
                logits[:, channel, :, :] += self.mean[channel]
                targets[:, channel, :, :] += self.mean[channel]
                logits[:, channel, :, :] *= self.std[channel]
                targets[:, channel, :, :] *= self.std[channel]
        #"""

        # Scale to imagenet mean and std
        for channel in range(len(self.imagenet_mean)):
            logits[:, channel, :, :] -= self.imagenet_mean[channel]
            targets[:, channel, :, :] -= self.imagenet_mean[channel]
            logits[:, channel, :, :] /= self.imagenet_std[channel]
            targets[:, channel, :, :] /= self.imagenet_std[channel]

        if self.compare_layer is None:
            pred_feature = self.feature_extractor(logits)
            target_feature = self.feature_extractor(targets)
            loss = self.p_criterion(pred_feature, target_feature)
        else:
            # Obtain features from different layers
            pred_feature = self.feature_extractor(logits)
            target_feature = self.feature_extractor(targets)

            # Calculate gram matrices
            pred_gram, target_gram = {}, {}
            for key in pred_feature:
                pred_gram[key] = self.gram(pred_feature[key])
                target_gram[key] = self.gram(target_feature[key])

            # TODO: Compare 3D activations
            # Compute distances between the gram matrices

            layer = self.compare_layer
            if isinstance(layer, list):
                loss = 0
                for l in layer:
                    loss += self.p_criterion(pred_gram[l], target_gram[l])
                loss /= len(layer)
                loss *= 1e5
            else:
                loss = self.p_criterion(pred_feature[layer], target_feature[layer])
        return loss

    @staticmethod
    def gram(x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G
