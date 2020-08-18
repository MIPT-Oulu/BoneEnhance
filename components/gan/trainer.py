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
            # Training split
            log_train = self.run_epoch(stage='train', epoch=epoch)
            losses_train.append(log_train)
            # Validation split
            log_val = self.run_epoch(stage='eval', epoch=epoch)
            losses_val.append(log_val)

        # Plot resulting LR curve
        self.plot_lr_curve([losses_train, losses_val])

    def run_epoch(self, stage, epoch):
        self._call_callbacks_by_name(callbacks=self.callbacks[stage], cb_func_name='on_epoch_begin', epoch=epoch,
                                     stage=stage, s=stage, n_epochs=self.num_epochs)

        # Set up dataloader
        self.loaders.set_epoch(epoch)
        state = self.loaders.state_dict()
        n_batches = state[f'loader_{stage}']['total']
        # n_batches = 10  # For debugging

        # Set whether model coefficients are updated
        if stage == 'train':
            self.model_g = self.model_g.train()
            self.model_d = self.model_d.train()
        else:
            self.model_g = self.model_g.eval()
            self.model_d = self.model_d.eval()

        # Iterate over minibatches
        running_loss = []
        for batch_id in range(n_batches):
            losses = self.run_batch(self.loaders, epoch, stage, batch_id, n_batches)
            running_loss.append(losses)
        running_loss = np.mean(running_loss, axis=0)

        self._call_callbacks_by_name(callbacks=self.callbacks[stage], cb_func_name='on_epoch_end', epoch=epoch,
                                     s=stage, stage=stage, n_epochs=self.num_epochs, n_batches=n_batches, strategy=self)

        return running_loss

    def run_batch(self, loader, epoch, stage, batch_i, n_batches):
        self._call_callbacks_by_name(callbacks=self.callbacks[stage], cb_func_name='on_minibatch_begin', epoch=epoch,
                                     stage=stage, n_epochs=self.num_epochs, batch_i=batch_i, n_batches=n_batches,
                                     progress_bar=self.progress_bar, s=stage)

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

        # Generator loss
        loss_g = self.loss_g(gen_hr, imgs_hr)
        if stage == 'train':
            loss_g.backward()
            self.optimizer_g.step()
            self.optimizer_d.zero_grad()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Predict real and fake images
        pred_real = self.model_d(imgs_hr)
        pred_fake = self.model_d(gen_hr.detach())  # Detach is important to only update the discriminator

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
            "[Epoch %d/%d %s] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (
                epoch,
                self.num_epochs,
                stage,
                batch_i,
                n_batches,
                loss_d.item(),
                loss_g.item(),
            )
        )
        loss_list = {'G_loss': loss_g, 'D_loss': loss_d}
        # Batch end callbacks
        self._call_callbacks_by_name(callbacks=self.callbacks[stage], cb_func_name='on_minibatch_end', epoch=epoch,
                                     s=stage, stage=stage, n_epochs=self.num_epochs, batch_i=batch_i, batches_count=n_batches,
                                     progress_bar=self.progress_bar, loss_list=loss_list, loss=loss_g,
                                     input=imgs_lr.detach().cpu(), output=gen_hr.cpu(), target=imgs_hr.detach().cpu())

        return loss_g.detach().cpu().numpy(), loss_d.detach().cpu().numpy()

    def _call_callbacks_by_name(self, cb_func_name, s, **kwargs):
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

        for cb in self.callbacks[s]:
            if hasattr(cb, cb_func_name):
                getattr(cb, cb_func_name)(**kwargs)

    def get_callbacks_by_name(self, name, stage):
        """
        get_callbacks_by_name is a public function which only retrieves some callback but does not call it
        Parameter
        ---------
        name: str
            name of the sessions where the callback function would be searched
        stage: str
            name of the learning stage where the call back function name would be searched.
        """

        cbs = ()
        for cb in self.callbacks[stage]:
            if hasattr(cb, name):
                cbs += cb

        return cbs

    @staticmethod
    def plot_lr_curve(loss_lists):
        # Learning curve
        plt.plot(np.array(loss_lists[0])[:, 0], 'b', label='Training loss [G]')
        plt.plot(np.array(loss_lists[0])[:, 1], 'b--', label='Training loss [D]')
        plt.plot(np.array(loss_lists[1])[:, 0], 'r', label='Validation loss [G]')
        plt.plot(np.array(loss_lists[1])[:, 1], 'r--', label='Validation loss [D]')
        plt.legend()
        f_size = 18
        plt.xticks(fontsize=f_size)
        plt.yticks(fontsize=f_size)
        plt.ylabel('Loss value', fontsize=f_size)
        plt.xlabel('Epoch', fontsize=f_size)
        plt.grid(True)
        plt.show()
