import torch
import torch.nn.functional as F
from torch.nn.init import sparse_
from pytorch_lightning import LightningModule, Trainer
from optimizers.kfac import KFACOptimizer


class VAE(LightningModule):

    def __init__(self, config):
        super().__init__()

        self.optimizer_ = config['optimizer']
        self.is_kfac = 'kfac' in str(self.optimizer_).lower()
        if self.is_kfac:
            self.criterion1 = torch.nn.CrossEntropyLoss()
        self.optimizer_params = config['optimizer_params']

        # Encoder architecture
        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(28*28, 1000),
            torch.nn.Tanh(),
            torch.nn.Linear(1000, 500),
            torch.nn.Tanh(),
            torch.nn.Linear(500, 250),
            torch.nn.Tanh(),
            torch.nn.Linear(250, 30),
        )

        # Decoder architecture
        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(30, 250),
            torch.nn.Tanh(),
            torch.nn.Linear(250, 500),
            torch.nn.Tanh(),
            torch.nn.Linear(500, 1000),
            torch.nn.Tanh(),
            torch.nn.Linear(1000, 28*28),
        )

        # Iterate and initialize weights for both parts of network following sparse initialization (Martens 2010)
        with torch.no_grad():
            for l1, l2 in zip(self._encoder, self._decoder):

                # Skip activation layers
                if isinstance(l1, torch.nn.Tanh):
                    continue

                # Set bias to zero
                l1.bias.data.fill_(0)
                l2.bias.data.fill_(0)

                # Retrieve weights (take transpose so columns end up being output units)
                w1 = l1.weight.T
                w2 = l2.weight.T

                # Compute sparsity so we have D - 15 sparse weights for each column
                s1 = 1 - 15/w1.shape[0]
                s2 = 1 - 15/w2.shape[0]

                # Initialize weights with sparse initialization (Martens 2010)
                l1.weight.data = sparse_(w1, sparsity=s1).T
                l2.weight.data = sparse_(w2, sparsity=s2).T

    def forward(self, X):
        # Vectorize MNIST default format (channel x height x width -> vector)
        X = X.view(X.size(0), -1)
        X = self._encoder(X)
        return self._decoder(X)

    """def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False):
        # Update params
        optimizer.step(closure=optimizer_closure)

        # Custom step for KFAC
        # From https://github.com/alecwangcq/KFAC-Pytorch
        raise Exception"""

    def training_step(self, batch, batch_idx):
        X, y = batch
        X_pred = self(X)

        if self.is_kfac:

            optimizer = self.trainer.optimizers[0]
            if optimizer.steps % optimizer.TCov == 0:
                # compute true fisher
                optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(X_pred.data, dim=1), 1).squeeze() # Potentially problematic
                loss_sample = self.criterion1(X_pred, sampled_y)
                loss_sample.backward(retain_graph=True)
                optimizer.acc_stats = False
                optimizer.zero_grad()  # clear the gradient for computing true-fisher.

        loss = F.binary_cross_entropy_with_logits(X_pred, X.view(X.size(0), -1), reduction='mean')
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X_pred = self(X)
        loss = F.binary_cross_entropy_with_logits(X_pred, X.view(X.size(0), -1), reduction='mean')
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer_input = self if self.is_kfac else self.parameters()
        return self.optimizer_(optimizer_input, **self.optimizer_params)
