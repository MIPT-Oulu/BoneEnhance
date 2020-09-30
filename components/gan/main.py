import torch
import torch.nn as nn
import pandas as pd
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collagen.data import DataProvider, ItemLoader
from collagen.data.samplers import GaussianNoiseSampler
from collagen.core import Module
from collagen.callbacks import ImagePairVisualizer, SimpleLRScheduler, \
    RandomImageVisualizer, ModelSaver

from BoneEnhance.components.transforms import train_test_transforms
from BoneEnhance.components.models import WGAN_VGG_generator, WGAN_VGG_discriminator, EnhanceNet, Vgg16, \
    Discriminator, ConvNet
from BoneEnhance.components.utilities.callbacks import ScalarMeterLogger, RunningAverageMeter


class GANFakeImageSampler(ItemLoader):
    def __init__(self, g_network, batch_size, image_size, name='ganfake') -> object:
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
    # Image size
    crop = (3, config.training.crop_small[0], config.training.crop_small[1])

    # Networks
    # model_g = WGAN_VGG_generator()
    model_g = EnhanceNet(config.training.crop_small, config.training.magnification,
                         activation=config.training.activation, upscale_input=config.training.upscale_input)
    ConvNet(config.training.magnification,
            activation=config.training.activation,
            upscale_input=config.training.upscale_input,
            n_blocks=config.training.n_blocks,
            normalization=config.training.normalization)
    # model_d = WGAN_VGG_discriminator(config.training.crop_small[0])
    model_d = Discriminator(crop)
    model_f = Vgg16()
    # Feature extractor does not need to be updated
    model_f.eval()

    if gpus > 1:
        model_g = nn.DataParallel(model_g)
        model_d = nn.DataParallel(model_d)
        model_f = nn.DataParallel(model_f)

    return model_g.to(device), model_d.to(device), model_f.to(device)


def create_data_provider_gan(g_network, item_loaders, args, config, parser, metadata, mean, std, device):
    # Compile ItemLoaders
    item_loaders['real'] = ItemLoader(meta_data=metadata['train'],
                                      transform=train_test_transforms(config, mean, std)['train'],
                                      parse_item_cb=parser,
                                      batch_size=config.training.bs, num_workers=args.num_threads,
                                      shuffle=True)

    #item_loaders['fake'] = GANFakeImageSampler(g_network=g_network,
    #                                           batch_size=config.training.bs,
    #                                           image_size=config.training.crop_small)
    item_loaders['fake'] = ItemLoader(meta_data=metadata['train'],
                                      transform=train_test_transforms(config, mean, std)['train'],
                                      parse_item_cb=parser,
                                      batch_size=config.training.bs, num_workers=args.num_threads,
                                      shuffle=True)

    item_loaders['noise'] = ItemLoader(meta_data=metadata['train'],
                                       transform=train_test_transforms(config, mean, std)['train'],
                                       parse_item_cb=parser,
                                       batch_size=config.training.bs, num_workers=args.num_threads,
                                       shuffle=True)

    #item_loaders['noise'] = GaussianNoiseSampler(batch_size=config.training.bs,
    #                                             latent_size=config.gan.latent_size,
    #                                             device=device, n_classes=config.gan.classes)

    return DataProvider(item_loaders)


def init_callbacks(fold_id, config, snapshots_dir, snapshot_name, model, optimizer, mean, std):
    # Snapshot directory
    current_snapshot_dir = snapshots_dir / snapshot_name
    crop = config.training.crop_small
    log_dir = current_snapshot_dir / f"fold_{fold_id}_log"

    # Tensorboard
    writer = SummaryWriter(comment='BoneEnhance', log_dir=log_dir, flush_secs=15, max_queue=1)
    prefix = f"{crop[0]}x{crop[1]}_fold_{fold_id}"

    # Callbacks
    train_cbs = (RunningAverageMeter(prefix="train", name="G_loss"),
                 RunningAverageMeter(prefix="train", name="D_loss"),
                 #RandomImageVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std),
                 ScalarMeterLogger(writer, comment='training', log_dir=str(log_dir)))

    val_cbs = (RunningAverageMeter(prefix="eval", name="G_loss"),
               RunningAverageMeter(prefix="eval", name="D_loss"),
               ImagePairVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std, scale=(0, 1)),
               #RandomImageVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std,
               #                      sigmoid=False),
               # Reduce LR on plateau
               SimpleLRScheduler('eval/loss', ReduceLROnPlateau(optimizer[0],
                                                                patience=int(config.training.patience),
                                                                factor=float(config.training.factor),
                                                                eps=float(config.training.eps))),
               SimpleLRScheduler('eval/loss', ReduceLROnPlateau(optimizer[1],
                                                                patience=int(config.training.patience),
                                                                factor=float(config.training.factor),
                                                                eps=float(config.training.eps))),
               ScalarMeterLogger(writer=writer, comment='validation', log_dir=log_dir))

    return train_cbs, val_cbs


class DiscriminatorLoss(Module):
    """

    """
    def __init__(self, config):
        super(DiscriminatorLoss, self).__init__()
        self.generator = EnhanceNet(config.training.crop_small, config.training.magnification)
        self.discriminator = WGAN_VGG_discriminator(config.training.crop_small[0])

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


class FeatureMatchingSampler(ItemLoader):
    def __init__(self, model: nn.Module, latent_size: int, data_key: str = "data",
                 meta_data: pd.DataFrame or None = None,
                 parse_item_cb: callable or None = None, name='fm',
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = False, timeout: int = 0):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last,
                         timeout=timeout)
        self.__model: nn.Module = model
        self.__latent_size: int = latent_size
        self.__data_key = data_key
        self.__name = name

    def sample(self, k=1):
        samples = []
        real_imgs_list = super().sample(k)
        for i in range(k):
            real_imgs = real_imgs_list[i][self.__data_key]
            features = self.__model.get_features(real_imgs)
            noise = torch.randn(self.batch_size, self.__latent_size)
            noise_on_device = noise.to(next(self.__model.parameters()).device)
            samples.append({'name': self.__name, 'real_features': features.detach(), 'real_data': real_imgs,
                            'latent': noise_on_device})
        return samples
