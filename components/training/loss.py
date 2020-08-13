from torch import nn, sigmoid
from BoneEnhance.components.models.wgan import WGAN_VGG_FeatureExtractor
from BoneEnhance.components.models.perceptual import Vgg16


class PerceptualLoss(nn.Module):
    """
    Calculates the perceptual loss based on difference between the VGG19 features.

    Possible layers to be compared: ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'].
    A single layer can be provided or a list in which case all losses are added together.
    """

    def __init__(self, criterion=nn.L1Loss(), compare_layer=None):
        super(PerceptualLoss, self).__init__()
        if compare_layer is None:
            self.feature_extractor = WGAN_VGG_FeatureExtractor()
        else:
            self.feature_extractor = Vgg16()
            self.feature_extractor.eval()
        self.p_criterion = criterion
        self.compare_layer = compare_layer

    def forward(self, logits, targets):

        if self.compare_layer is None:
            pred_feature = self.feature_extractor(logits)
            target_feature = self.feature_extractor(targets)
            loss = self.p_criterion(pred_feature, target_feature)
        else:
            pred_feature = self.feature_extractor(logits)
            target_feature = self.feature_extractor(targets)

            layer = self.compare_layer
            if isinstance(layer, list):
                loss = 0
                for l in layer:
                    loss += self.p_criterion(pred_feature[l], target_feature[l])
            else:
                loss = self.p_criterion(pred_feature[layer], target_feature[layer])
        return loss
