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


from BoneEnhance.components.training.session import create_data_provider, init_experiment, init_callbacks, \
    save_transforms, init_loss, init_model
from BoneEnhance.components.training import parse_grayscale, parse_3d, parse_3d_debug, parse_segmentation
from BoneEnhance.components.splits import build_splits
from BoneEnhance.components.inference.pipeline_components import inference_runner_oof, evaluation_runner

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    # Timing
    start = time()

    # Initialize experiment
    args_base, config_list, device = init_experiment()

    # Loop for experiment configurations
    for experiment in range(len(config_list)):
        # Time of the current experiment
        start_exp = time()
        args = deepcopy(args_base)  # Copy args so that they can be updated
        config = OmegaConf.create(config_list[experiment])

        # Update arguments according to the configuration file
        if len(config.training.crop_small) == 3:
            parser = partial(parse_3d, config=config)
        else:
            parser = partial(parse_grayscale, config=config)

        # Split training folds
        parser_debug = partial(parser, debug=True)  # Display figures
        splits_metadata = build_splits(args.data_location, args, config, parser_debug,
                                       args.snapshots_dir, config.training.snapshot)
        mean, std = splits_metadata['mean'], splits_metadata['std']

        # Loss
        loss_criterion = init_loss(config.training.loss, config, device=device, mean=mean, std=std, args=args)

        # Save transforms list
        save_transforms(args.snapshots_dir / config.training.snapshot, config, args, mean, std)

        # Training for separate folds
        for fold in range(config.training.n_folds):
            print(f'\nTraining fold {fold}')
            # Initialize data provider
            data_provider = create_data_provider(args, config, parser, metadata=splits_metadata[f'fold_{fold}'],
                                                 mean=mean, std=std)

            # Initialize model
            model = init_model(config, device, args.gpus, args=args)

            # Optimizer
            optimizer = optim.Adam(model.parameters(),
                                   lr=config.training.lr,
                                   weight_decay=config.training.wd)
            # Callbacks
            train_cbs, val_cbs = init_callbacks(fold, config, args.snapshots_dir,
                                                config.training.snapshot, model, optimizer, mean=mean, std=std)

            # Initialize session
            sessions = dict()
            sessions['SR'] = Session(data_provider=data_provider,
                                     train_loader_names=tuple(config.data_sampling.train.data_provider.SR.keys()),
                                     val_loader_names=tuple(config.data_sampling.eval.data_provider.SR.keys()),
                                     module=model, loss=loss_criterion, optimizer=optimizer,
                                     train_callbacks=train_cbs,
                                     val_callbacks=val_cbs)

            # Run training
            strategy = Strategy(data_provider=data_provider,
                                data_sampling_config=config.data_sampling,
                                strategy_config=config.strategy,
                                sessions=sessions,
                                n_epochs=config.training.epochs,
                                device=device)
            strategy.run()

            # Manage memory
            del strategy
            del model
            cuda.empty_cache()
            gc.collect()

        # Duration of the current experiment
        dur = time() - start_exp
        print(f'Model {experiment + 1} trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')

        # Calculate out-of-fold inference and evaluate metrics
        if config.inference.calc_inference:
            save_dir = inference_runner_oof(args, config, splits_metadata, device)

            evaluation_runner(args, config, save_dir)

    # Duration of the whole script
    dur = time() - start
    print(f'Models trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
