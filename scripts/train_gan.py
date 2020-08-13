from torch import optim, cuda, save, cat
from time import time
from copy import deepcopy
from torch.nn.functional import interpolate
import gc
from omegaconf import OmegaConf
import cv2
from functools import partial
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

from torch.nn import BCELoss, BCEWithLogitsLoss, L1Loss


from BoneEnhance.components.training.session import init_experiment, save_transforms, parse_grayscale, \
    create_data_provider
from BoneEnhance.components.splits import build_splits
from BoneEnhance.components.training.gan import init_model_gan, DiscriminatorLoss, init_callbacks
from BoneEnhance.components.inference.pipeline_components import inference_runner_oof, evaluation_runner



cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def _call_callbacks_by_name(cb_func_name, **kwargs):
    """
    _call_callbacks_by_name is a private function to be called inside the class. This function traverses all the
    sessions to check if they have the callback function to be called. If the callback is found it is called.
    Afterwards, it searches the self.__callback, a private variable holding manually provided callbacks, if the
    provided callback is found here, it is also called.

    Parameters
    ----------
    cb_func_name: str
        name of the call_back_function to be called
    kwargs: list or tuple
        argument for the callback function
    """

    for cb in callbacks:
        if hasattr(cb, cb_func_name):
            getattr(cb, cb_func_name)(strategy='', **kwargs)


if __name__ == "__main__":
    # Timing
    start = time()

    # Initialize experiment
    args_base, config_list, device = init_experiment(experiments='../experiments/run_gan')

    for experiment in range(len(config_list)):
        # Current experiment
        start_exp = time()
        args = deepcopy(args_base)  # Copy args so that they can be updated
        config = OmegaConf.create(config_list[experiment])

        # Update arguments according to the configuration file
        parser = partial(parse_grayscale, config=config)

        # Split training folds
        parser_debug = partial(parser, debug=True)  # Display figures
        splits_metadata = build_splits(args.data_location, args, config, parser_debug,
                                       args.snapshots_dir, config.training.snapshot)
        mean, std = splits_metadata['mean'], splits_metadata['std']

        # Save transforms list
        save_transforms(args.snapshots_dir / config.training.snapshot, config, args, mean, std)

        # Training for separate folds
        for fold in range(config.training.n_folds):
            print(f'\nTraining fold {fold}')

            # Initialize model and optimizer
            generator, discriminator, feature_extractor = init_model_gan(config, device, args.gpus)

            optimizer_d = optim.Adam(discriminator.parameters(), lr=config.training.lr, weight_decay=config.training.wd)
            optimizer_g = optim.Adam(generator.parameters(), lr=config.training.lr, weight_decay=config.training.wd)

            #loss_d = DiscriminatorLoss(config).to(device)
            #loss_d = BCELoss().to(device)
            #loss_g = GeneratorLoss(d_network=generator, d_loss=loss_d).to(device)

            criterion_GAN = BCEWithLogitsLoss().to(device)
            criterion_content = L1Loss().to(device)
            criterion_pixel = L1Loss().to(device)

            Tensor = cuda.FloatTensor

            # Initialize data provider
            dataloader = create_data_provider(args, config, parser, metadata=splits_metadata[f'fold_{fold}'],
                                                 mean=mean, std=std)

            callbacks = init_callbacks(fold, config, args.snapshots_dir, config.training.snapshot,
                                       (generator, discriminator), (optimizer_g, optimizer_d), mean, std)


            # Training loop
            pbar = tqdm(range(config.training.epochs))
            for epoch, _ in enumerate(pbar):
                dataloader.set_epoch(epoch)
                state = dataloader.state_dict()

                for stage, cb in zip(['train', 'eval'], callbacks):
                    n_batches = state[f'loader_{stage}']['total']

                    _call_callbacks_by_name(callbacks=cb, cb_func_name='on_epoch_begin', epoch=epoch, stage=stage,
                                             n_epochs=config.training.epochs)

                    for i in range(n_batches):

                        _call_callbacks_by_name(callbacks=cb, cb_func_name='on_batch_begin', epoch=epoch, stage=stage,
                                                n_epochs=config.training.epochs, batch_i=i, n_batches=n_batches,
                                                progress_bar=pbar)

                        imgs = dataloader.sample(**{f'loader_{stage}': 1})[0][0]
                        batches_done = epoch * state[f'loader_{stage}']['total'] + i

                        # Configure model input
                        imgs_lr = Variable(imgs['data']).to(device)
                        imgs_hr = Variable(imgs['target']).to(device)

                        # Adversarial ground truths
                        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.module.output_shape))),
                                         requires_grad=False)
                        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.module.output_shape))),
                                        requires_grad=False)

                        # ------------------
                        #  Train Generators
                        # ------------------

                        optimizer_g.zero_grad()

                        # Generate a high resolution image from low resolution input
                        gen_hr = generator(imgs_lr)

                        # Measure pixel-wise loss against ground truth
                        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

                        if batches_done < config.gan.warmup_batches:
                            # Warm-up (pixel-wise loss only)
                            loss_pixel.backward()
                            optimizer_g.step()
                            pbar.set_description(
                                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                                % (epoch, config.training.epochs, i, n_batches, loss_pixel.item())
                            )

                            _call_callbacks_by_name(callbacks=cb, cb_func_name='on_batch_end', epoch=epoch, stage=stage,
                                                    n_epochs=config.training.epochs, batch_i=i, n_batches=n_batches,
                                                    progress_bar=pbar)
                            continue

                        # Extract validity predictions from discriminator
                        pred_real = discriminator(imgs_hr).detach()
                        pred_fake = discriminator(gen_hr)

                        # Adversarial loss (relativistic average GAN)
                        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

                        # Content loss
                        gen_features = feature_extractor(gen_hr)
                        real_features = feature_extractor(imgs_hr)#.detach()
                        if isinstance(gen_features, dict):
                            loss_content = 0
                            for l in gen_features.keys():
                                loss_content += criterion_content(gen_features[l], real_features[l])
                        else:
                            loss_content = criterion_content(gen_features, real_features)

                        # Total generator loss
                        lambda_adv, lambda_pix = config.gan.lambda_adv, config.gan.lambda_pixel
                        loss_g = loss_content + lambda_adv * loss_GAN + lambda_pix * loss_pixel

                        loss_g.backward()
                        optimizer_g.step()

                        # ---------------------
                        #  Train Discriminator
                        # ---------------------

                        optimizer_d.zero_grad()

                        pred_real = discriminator(imgs_hr)
                        pred_fake = discriminator(gen_hr.detach())

                        # Adversarial loss for real and fake images (relativistic average GAN)
                        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
                        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

                        # Total loss
                        loss_d = (loss_real + loss_fake) / 2

                        loss_d.backward()
                        optimizer_d.step()

                        # --------------
                        #  Log Progress
                        # --------------

                        pbar.set_description(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                            % (
                                epoch,
                                config.training.epochs,
                                i,
                                n_batches,
                                loss_d.item(),
                                loss_g.item(),
                                loss_content.item(),
                                loss_GAN.item(),
                                loss_pixel.item(),
                            )
                        )

                        _call_callbacks_by_name(callbacks=cb, cb_func_name='on_batch_end', epoch=epoch, stage=stage,
                                                n_epochs=config.training.epochs, batch_i=i, n_batches=n_batches,
                                                progress_bar=pbar)
                        """
                        log_dir = args.snapshots_dir / config.training.snapshot / f"fold_{fold}_log"
                        log_dir.mkdir(exist_ok=True)
                        if batches_done % config.gan.sample_interval == 0:
                            # Save image grid with upsampled inputs and ESRGAN outputs
                            imgs_lr = interpolate(imgs_lr, scale_factor=4)
                            img_grid = cat((imgs_lr, gen_hr, imgs_hr), -1)
                            save_image(img_grid, str(log_dir) + "/%d.png" % batches_done, nrow=1, normalize=False)

                        model_dir = args.snapshots_dir / config.training.snapshot / 'saved_models'
                        model_dir.mkdir(exist_ok=True)
                        if batches_done % config.gan.sample_interval == 0:
                            # Save model checkpoints
                            save(generator.state_dict(), str(model_dir) + "generator_%d.pth" % epoch)
                            save(discriminator.state_dict(), str(model_dir) + "discriminator_%d.pth" % epoch)
                        """
                    _call_callbacks_by_name(callbacks=cb, cb_func_name='on_epoch_end', epoch=epoch, stage=stage,
                                            n_epochs=config.training.epochs, n_batches=n_batches)

        dur = time() - start_exp
        print(f'Model {experiment + 1} trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')

        if config.inference.calc_inference:
            save_dir = inference_runner_oof(args, config, splits_metadata, device)

            evaluation_runner(args, config, save_dir)

    dur = time() - start
    print(f'Models trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')

