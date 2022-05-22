import torch
import torch.nn.functional as F
from torch.nn.init import sparse_
from pytorch_lightning import LightningModule, Trainer


class VAE(LightningModule):

    def __init__(self, config):
        super().__init__()

        self.optimizer = config['optimizer']
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

    def training_step(self, batch, batch_idx):
        X, y = batch
        X_pred = self(X)
        loss = F.binary_cross_entropy_with_logits(X_pred, X.view(X.size(0), -1), reduction='mean')
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X_pred = self(X)
        loss = F.binary_cross_entropy_with_logits(X_pred, X.view(X.size(0), -1), reduction='mean')
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optimizer_params)
