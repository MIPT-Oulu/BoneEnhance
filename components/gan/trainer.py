import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
import torch


class Trainer:
    """
    The class that runs training and evaluation of the model.
    """
    def __init__(self, model, loaders, criterion, opt, callbacks, device='cpu'):
        self.device = device
        self.model = model
        # Move model to selected device
        self.model_g = self.model[0].to(self.device)
        self.model_d = self.model[1].to(self.device)
        # Data
        self.loaders = loaders

        self.loss_g = criterion[0].to(self.device)
        self.loss_d = criterion[1].to(self.device)
        self.optimizer_g = opt[0]
        self.optimizer_d = opt[1]
        self.callbacks = callbacks
        #self.tb = logger
        self.num_epochs = None
        self.progress_bar = None

    def run(self, num_epochs=1):
        best_loss = np.inf
        self.num_epochs = num_epochs
        self.progress_bar = tqdm(range(num_epochs), desc=f'Training')

        losses_train, losses_val = [], []
        for epoch, _ in enumerate(self.progress_bar):
            log_train = self.run_epoch(stage='train', epoch=epoch)
            losses_train.append(log_train)
            log_val = self.run_epoch(stage='val', epoch=epoch)
            losses_val.append(log_val)

            if log_val < best_loss:
                # Here we typically save the best model
                best_loss = log_val
            # Update progress bar
            self.progress_bar.set_postfix({'': f'Validation loss: {log_val:.2f}'})

        # Plot result
        self.plot_lr_curve([losses_train, losses_val])

    def run_epoch(self, stage, epoch):
        self._call_callbacks_by_name(callbacks=self.callbacks[stage], cb_func_name='on_epoch_begin', epoch=epoch, stage=stage,
                                     n_epochs=self.num_epochs)

        # Set up dataloader
        self.loaders.set_epoch(epoch)
        state = self.loaders.state_dict()
        n_batches = state[f'loader_{stage}']['total']

        # Set whether model coefficients are updated
        if stage == 'train':
            self.model_g = self.model_g.train()
            self.model_d = self.model_d.train()
        else:
            self.model_g = self.model_g.eval()
            self.model_d = self.model_d.eval()

        # Iterate over minibatches
        running_loss = 0
        for batch_id in range(n_batches):

            self.run_batch(self.loaders, epoch, stage, batch_id, n_batches)

        self._call_callbacks_by_name(callbacks=self.callbacks[stage], cb_func_name='on_epoch_end', epoch=epoch, stage=stage,
                                n_epochs=self.num_epochs, n_batches=n_batches)

        # Scale to number of batches
        running_loss = running_loss / len(self.loaders)
        # Add loss value to Tensorboard
        #self.tb.add_scalar(f'Loss/{stage}', running_loss, epoch)
        return running_loss

    def run_batch(self, loader, epoch, stage, batch_i, n_batches):
        self._call_callbacks_by_name(callbacks=self.callbacks[stage], cb_func_name='on_batch_begin', epoch=epoch, stage=stage,
                                     n_epochs=self.num_epochs, batch_i=batch_i, n_batches=n_batches,
                                     progress_bar=self.progress_bar)

        imgs = loader.sample(**{f'loader_{stage}': 1})[0][0]

        # Configure model input
        imgs_lr = Variable(imgs['data']).to(self.device)
        imgs_hr = Variable(imgs['target']).to(self.device)

        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(np.ones((imgs_lr.size(0), *self.model_d.module.output_shape))),
                         requires_grad=False).to(self.device)
        fake = Variable(torch.FloatTensor(np.zeros((imgs_lr.size(0), *self.model_d.module.output_shape))),
                        requires_grad=False).to(self.device)

        # ------------------
        #  Train Generators
        # ------------------
        if stage == 'train':
            # Set the gradients of the optimizer to zero
            self.optimizer_g.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = self.model_g(imgs_lr)

        """
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
        """


        # Extract validity predictions from discriminator
        #pred_real = self.model_d(imgs_hr)
        #pred_fake = self.model_d(gen_hr)

        ## Adversarial loss (relativistic average GAN)
        #loss_GAN = self.loss_d(pred_fake - pred_real.mean(0, keepdim=True), valid)

        """
        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)  # .detach()
        if isinstance(gen_features, dict):
            loss_content = 0
            for l in gen_features.keys():
                loss_content += self.loss_g(gen_features[l], real_features[l])
        else:
            loss_content = self.loss_g(gen_features, real_features)

        # Total generator loss
        lambda_adv, lambda_pix = config.gan.lambda_adv, config.gan.lambda_pixel
        loss_g = loss_content + lambda_adv * loss_GAN + lambda_pix * loss_pixel
        """

        loss_g = self.loss_g(gen_hr, imgs_hr)
        if stage == 'train':
            loss_g.backward(retain_graph=True)
            self.optimizer_g.step()
            self.optimizer_d.zero_grad()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        pred_real = self.model_d(imgs_hr)
        pred_fake = self.model_d(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = self.loss_d(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.loss_d(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_d = loss_real + loss_fake

        if stage == 'train':
            loss_d.backward()
            self.optimizer_d.step()

        # --------------
        #  Log Progress
        # --------------

        self.progress_bar.set_description(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (
                epoch,
                self.num_epochs,
                batch_i,
                n_batches,
                loss_d.item(),
                loss_g.item(),
            )
        )


        self._call_callbacks_by_name(callbacks=self.callbacks[stage], cb_func_name='on_batch_end', epoch=epoch, stage=stage,
                                    n_epochs=self.num_epochs, batch_i=batch_i, n_batches=n_batches,
                                    progress_bar=self.progress_bar)

    def _call_callbacks_by_name(self, cb_func_name, **kwargs):
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

        for cb in self.callbacks:
            if hasattr(cb, cb_func_name):
                getattr(cb, cb_func_name)(strategy='', **kwargs)

    @staticmethod
    def plot_lr_curve(loss_lists):
        # Learning curve
        plt.plot(loss_lists[0], label='Training loss', color='b')
        plt.plot(loss_lists[1], label='Validation loss', color='r')
        plt.legend()
        f_size = 18
        plt.xticks(fontsize=f_size)
        plt.yticks(fontsize=f_size)
        plt.ylabel('Loss value', fontsize=f_size)
        plt.xlabel('Epoch', fontsize=f_size)
        plt.grid(True)
        plt.show()