from torch import optim, cuda
from time import time
from copy import deepcopy
import gc
from omegaconf import OmegaConf
import cv2
from functools import partial
from torch.utils.tensorboard import SummaryWriter

from collagen.core import Session
from collagen.strategies import Strategy
from collagen.callbacks import SamplingFreezer, ScalarMeterLogger, ImageSamplingVisualizer, RunningAverageMeter, \
    BatchProcFreezer
from collagen.losses import GeneratorLoss

from BoneEnhance.components.training.session import init_experiment, save_transforms, parse_grayscale
from BoneEnhance.components.splits import build_splits
from BoneEnhance.components.gan.main import create_data_provider_gan, init_model_gan, DiscriminatorLoss
from BoneEnhance.components.inference.pipeline_components import inference_runner_oof, evaluation_runner

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


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
            model_g, model_d, model_f = init_model_gan(config, device, args.gpus)

            optimizer_d = optim.Adam(model_d.parameters(), lr=config.training.lr, weight_decay=config.training.wd)
            #loss_d = BCELoss().to(device)
            loss_d = DiscriminatorLoss(config).to(device)

            optimizer_g = optim.Adam(model_g.parameters(), lr=config.training.lr, weight_decay=config.training.wd)
            loss_g = GeneratorLoss(d_network=model_g, d_loss=loss_d).to(device)

            # Initialize data provider
            item_loaders = dict()
            data_provider = create_data_provider_gan(model_g, item_loaders, args, config, parser,
                                                     metadata=splits_metadata[f'fold_{fold}'],
                                                     mean=mean, std=std, device=device)

            # Setting up the callbacks
            log_dir = args.snapshots_dir / config.training.snapshot / f"fold_{fold}_log"
            summary_writer = SummaryWriter(comment='BoneEnhance', log_dir=log_dir, flush_secs=15, max_queue=1)
            st_callbacks = (SamplingFreezer([model_d, model_g]),
                            ScalarMeterLogger(writer=summary_writer),
                            ImageSamplingVisualizer(generator_sampler=item_loaders['fake'],
                                                    transform=lambda x: (x + 1.0) / 2.0,
                                                    writer=summary_writer,
                                                    grid_shape=tuple(config.training.crop_small)))

            # Initialize session
            sessions = dict()
            sessions['G'] = Session(data_provider=data_provider,
                                    train_loader_names=tuple(config.data_sampling.train.data_provider.G.keys()),
                                    val_loader_names=tuple(config.data_sampling.eval.data_provider.G.keys()),
                                    module=model_g, loss=loss_g, optimizer=optimizer_g,
                                    train_callbacks=(BatchProcFreezer(modules=model_d),
                                                     RunningAverageMeter(prefix="train/G", name="loss")),
                                    val_callbacks=RunningAverageMeter(prefix="eval/G", name="loss"),)

            sessions['D'] = Session(data_provider=data_provider,
                                    train_loader_names=tuple(config.data_sampling.train.data_provider.D.keys()),
                                    val_loader_names=None,
                                    module=model_d, loss=loss_d, optimizer=optimizer_d,
                                    train_callbacks=(BatchProcFreezer(modules=model_g),
                                                     RunningAverageMeter(prefix="train/D", name="loss")))

            # Run training
            strategy = Strategy(data_provider=data_provider,
                                data_sampling_config=config.data_sampling,
                                strategy_config=config.strategy,
                                sessions=sessions,
                                n_epochs=config.training.epochs,
                                callbacks=st_callbacks,
                                device=device)

            strategy.run()

            # Manage memory
            del strategy
            cuda.empty_cache()
            gc.collect()

        dur = time() - start_exp
        print(f'Model {experiment + 1} trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')

        if config.inference.calc_inference:
            save_dir = inference_runner_oof(args, config, splits_metadata, device)

            evaluation_runner(args, config, save_dir)

    dur = time() - start
    print(f'Models trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
