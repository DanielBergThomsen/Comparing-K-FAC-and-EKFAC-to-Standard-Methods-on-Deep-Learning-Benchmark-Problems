import torch
import torch.nn.functional as F
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
