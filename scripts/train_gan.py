from torch import optim
from time import time
from copy import deepcopy
from omegaconf import OmegaConf
import cv2
from functools import partial
from torch.nn import BCEWithLogitsLoss, L1Loss, BCELoss, MSELoss

from bone_enhance.training.session import init_experiment, save_transforms, create_data_provider, init_loss
from bone_enhance.training import parse_grayscale
from bone_enhance.splits import build_splits
from bone_enhance.gan import init_model_gan, init_callbacks, Trainer
from bone_enhance.inference.pipeline_components import inference_runner_oof, evaluation_runner

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    # Timing
    start = time()

    # Initialize experiment
    args_base, config_list, config_paths, device = init_experiment(experiments='../experiments/run_gan')

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

        # Loss
        criterion_GAN = MSELoss().to(device)
        criterion_content = init_loss(config.training.loss, config, device=device)
        criterion_pixel = L1Loss().to(device)
        loss = {
            'content': criterion_content,
            'pixel': criterion_pixel,
            'adversarial': criterion_GAN
        }

        # Save transforms list
        save_transforms(args.snapshots_dir / config.training.snapshot, config, args, mean, std)

        # Training for separate folds
        for fold in range(config.training.n_folds):
            print(f'\nTraining fold {fold}')

            # Initialize model
            generator, discriminator, feature_extractor = init_model_gan(config, device, args.gpus)

            # Optimizers
            optimizer_d = optim.Adam(discriminator.parameters(), lr=config.training.lr, weight_decay=config.training.wd)
            optimizer_g = optim.Adam(generator.parameters(), lr=config.training.lr, weight_decay=config.training.wd)

            # Initialize data provider
            dataloader = create_data_provider(args, config, parser, metadata=splits_metadata[f'fold_{fold}'],
                                              mean=mean, std=std)

            # Combine callbacks into dictionary
            callbacks = init_callbacks(fold, config, args.snapshots_dir, config.training.snapshot,
                                       (generator, discriminator), (optimizer_g, optimizer_d), mean, std)
            callbacks = {'train': callbacks[0], 'eval': callbacks[1]}
            current_snapshot_dir = args.snapshots_dir / config.training.snapshot

            # Set up model training
            trainer = Trainer(
                model=[generator, discriminator],
                loaders=dataloader,
                criterion=loss,
                opt=[optimizer_g, optimizer_d],
                device=device,
                config=config,
                callbacks=callbacks,
                snapshot=current_snapshot_dir,
                prefix=f'fold_{fold}',
                mean=mean,
                std=std
            )
            trainer.run(num_epochs=config.training.epochs)

        dur = time() - start_exp
        print(f'Model {experiment + 1} trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')

        if config.inference.calc_inference:
            save_dir = inference_runner_oof(args, config, splits_metadata, device)

            evaluation_runner(args, config, save_dir)

    dur = time() - start
    print(f'Models trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')

