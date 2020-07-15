from torch import nn, sigmoid
from BoneEnhance.components.models.wgan import WGAN_VGG_FeatureExtractor


class PerceptualLoss(nn.Module):
    """
    Calculates the Peak signal-to-noise ratio and returns 100 - psnr as the loss value.
    """

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss()

    def forward(self, logits, targets):
        """Calculates the perceptual loss based on difference between the VGG19 features."""
        preds = sigmoid(logits)

        pred_feature = self.feature_extractor(preds)
        target_feature = self.feature_extractor(targets)
        loss = self.p_criterion(pred_feature, target_feature)
        return loss
