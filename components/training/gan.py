import torch
import torch.nn as nn
from torch import Tensor
from collagen.data import DataProvider, ItemLoader
from collagen.data.samplers import GaussianNoiseSampler
from collagen.core import Module

from BoneEnhance.components.transforms import train_test_transforms
from BoneEnhance.components.models import WGAN_VGG_generator, WGAN_VGG_discriminator


class GANFakeImageSampler(ItemLoader):
    def __init__(self, g_network, batch_size, image_size, name='ganfake'):
        super().__init__(meta_data=None, parse_item_cb=None, name=name)
        self.batch_size = batch_size
        self.__image_size = image_size
        self.__g_network = g_network
        self.__name = name

    def sample(self, k=1):
        samples = []
        for _ in range(k):
            noise = torch.randn(self.batch_size, 3, self.__image_size[0], self.__image_size[1])
            noise_on_device = noise.to(next(self.__g_network.parameters()).device)
            fake: torch.Tensor = self.__g_network(noise_on_device)
            samples.append(
                {'name': self.__name, 'data': fake.detach(), 'target': torch.zeros(self.batch_size).to(fake.device),
                 'latent': noise})

        return samples

    def __len__(self):
        return 1


def init_model_gan(config, device='cuda', gpus=1):
    model_g = WGAN_VGG_generator()
    model_d = WGAN_VGG_discriminator(config.training.crop_small[0])

    if gpus > 1:
        model_g = nn.DataParallel(model_g)
        model_d = nn.DataParallel(model_d)

    return model_g.to(device), model_d.to(device)


def create_data_provider_gan(g_network, item_loaders, args, config, parser, metadata, mean, std, device):
    # Compile ItemLoaders
    item_loaders['real'] = ItemLoader(meta_data=metadata['train'],
                                      transform=train_test_transforms(config, mean, std)['train'],
                                      parse_item_cb=parser,
                                      batch_size=config.training.bs, num_workers=args.num_threads,
                                      shuffle=True)

    item_loaders['fake'] = GANFakeImageSampler(g_network=g_network,
                                          batch_size=config.training.bs,
                                          image_size=config.training.crop_small)

    item_loaders['noise'] = GaussianNoiseSampler(batch_size=config.training.bs,
                                                 latent_size=config.gan.latent_size,
                                                 device=device, n_classes=config.gan.classes)

    return DataProvider(item_loaders)


class DiscriminatorLoss(Module):
    """

    """
    def __init__(self, input_size):
        super(DiscriminatorLoss, self).__init__()
        self.generator = WGAN_VGG_generator()
        self.discriminator = WGAN_VGG_discriminator(input_size)

    def forward(self, img: Tensor, target: Tensor, gp=False, return_gp=False):
        fake = self.generator(img)
        d_real = self.discriminator(target)
        d_fake = self.discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(target, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss