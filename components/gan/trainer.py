import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer():
    """
    The class that runs training and evaluation of the model.
    """
    def __init__(self, model, loaders, criterion, opt, logger, device='cpu'):
        self.device = device
        self.model = model
        # Move model to selected device
        self.model = self.model.to(self.device)
        # Data
        self.loaders = loaders

        self.loss_func = criterion
        self.loss_func = self.loss_func.to(self.device)
        self.optimizer = opt
        self.tb = logger

    def run(self, num_epochs=1):
        best_loss = np.inf
        progress_bar = tqdm(range(num_epochs), desc=f'Training')

        losses_train, losses_val = [], []
        for epoch, _ in enumerate(progress_bar):
            log_train = self.run_epoch(self, mode='train', epoch_idx=epoch)
            losses_train.append(log_train)
            log_val = self.run_epoch(self, mode='val', epoch_idx=epoch)
            losses_val.append(log_val)

            if log_val < best_loss:
                # Here we typically save the best model
                best_loss = log_val
            # Update progress bar
            progress_bar.set_postfix({'': f'Validation loss: {log_val:.2f}'})

        # Plot result
        self.plot_lr_curve([losses_train, losses_val])

    @staticmethod
    def run_epoch(self, mode, epoch_idx):
        loader = self.loaders[f'loader_{mode}']

        # Set whether model coefficients are updated
        if mode == 'train':
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        # Iterate over minibatches
        running_loss = 0
        for batch_id, batch in enumerate(loader):

            # Extract the input and target from the batch
            x = batch['data'].to(self.device)
            y = batch['target'].to(self.device)

            if mode == 'train':
                # Set the gradients of the optimizer to zero
                self.optimizer.zero_grad()

            # Forward through the model
            predict = self.model(x).flatten()

            # Calculate loss
            loss = self.loss_func(predict, y)

            if mode == 'train':
                # Apply backpropagation and compute gradients
                loss.backward()

                # Update model weights based on the gradients
                self.optimizer.step()

            running_loss += loss.item()

        # Scale to number of batches
        running_loss = running_loss / len(loader)
        # Add loss value to Tensorboard
        self.tb.add_scalar(f'Loss/{mode}', running_loss, epoch_idx)
        return running_loss

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