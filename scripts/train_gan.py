from torch import optim
from time import time
from copy import deepcopy
from omegaconf import OmegaConf
import cv2
from functools import partial
from torch.nn import BCEWithLogitsLoss, L1Loss, BCELoss

from BoneEnhance.components.training.session import init_experiment, save_transforms, parse_grayscale, \
    create_data_provider, init_loss
from BoneEnhance.components.splits import build_splits
from BoneEnhance.components.gan import init_model_gan, init_callbacks, Trainer
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

            loss_g = init_loss('combined_layers', config, device=device)

            # Initialize data provider
            dataloader = create_data_provider(args, config, parser, metadata=splits_metadata[f'fold_{fold}'],
                                                 mean=mean, std=std)

            callbacks = init_callbacks(fold, config, args.snapshots_dir, config.training.snapshot,
                                       (generator, discriminator), (optimizer_g, optimizer_d), mean, std)
            callbacks = {'train': callbacks[0], 'eval': callbacks[1]}

            trainer = Trainer(
                model=[generator, discriminator],
                loaders=dataloader,
                criterion=[loss_g, criterion_GAN],
                opt=[optimizer_g, optimizer_g],
                device=device,
                callbacks=callbacks
            )
            trainer.run(num_epochs=config.training.epochs)

        dur = time() - start_exp
        print(f'Model {experiment + 1} trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')

        if config.inference.calc_inference:
            save_dir = inference_runner_oof(args, config, splits_metadata, device)

            evaluation_runner(args, config, save_dir)

    dur = time() - start
    print(f'Models trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')

