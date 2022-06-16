"""
Contains the PyTorch Lightning implementations of all the network architectures used in our experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import sparse_
import torchvision

from pytorch_lightning import LightningModule


class KFACSuperModule(LightningModule):
    """
    Configures and implements the stuff that is common between the autoencoder and the ResNets during training.
    """
    def __init__(self, config):
        super().__init__()

        # Configure optimizer
        self.optimizer_ = config['optimizer']
        self.optimizer_params = config['optimizer_params']
        self.loss = config['loss_function']
        self.save_hyperparameters()

        # If using KFAC, set inner criterion as well
        self.is_kfac = 'kfac' in str(self.optimizer_).lower()
        if self.is_kfac:
            self.criterion1 = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        """
        Training step adapted for the implementations of KFAC and EKFAC we use in our experiments.
        """
        X, y = batch
        X_pred = self(X)
        loss = self.loss(X_pred, X, y)

        # This code is adapted from https://github.com/alecwangcq/KFAC-Pytorch
        # It corresponds to updating our running approximate of the inverse of the Fisher Information Matrix
        if self.is_kfac:

            optimizer = self.trainer.optimizers[0]
            if optimizer.steps % optimizer.TCov == 0:
                # compute true fisher
                optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(X_pred.data, dim=1), 1).squeeze()
                loss_sample = self.criterion1(X_pred, sampled_y)
                loss_sample.backward(retain_graph=True)
                optimizer.acc_stats = False
                optimizer.zero_grad()  # clear the gradient for computing true-fisher.

        # Compute and log training statistics
        _, pred = torch.max(X_pred, 1)
        num_correct = (pred == y).sum()
        training_acc = num_correct.item()/y.size(0)

        self.log("train_loss", loss.item())
        self.log("train_acc", training_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Standard validation step. In PyTorch Lightning, this will run at the end of every k epochs (as configured).
        """
        X, y = batch
        X_pred = self(X)

        loss = self.loss(X_pred, X, y)

        # Compute and log training statistics
        _, pred = torch.max(X_pred.data, 1)
        num_correct = (pred == y).sum()
        training_acc = num_correct.item()/y.size(0)

        self.log("val_loss", loss.item())
        self.log("val_acc", training_acc)

    def configure_optimizers(self):
        optimizer_input = self if self.is_kfac else self.parameters()  # Quick hack to get KFAC running
        return self.optimizer_(optimizer_input, **self.optimizer_params)


class ResNet(KFACSuperModule):
    """
    Implements ResNet-like architectures using configuration parameters. In particular, it is important to specify which
    architecture to use in the 'network_name' parameter of the provided config.
    """
    def __init__(self, config):
        super().__init__(config)

        # Configure ResNet architecture
        if 'architecture' in config:
            arch = config['architecture']
        else:
            raise ValueError('Architecture not specified.')
        if arch == 'ResNet18':
            self.model = torchvision.models.resnet18(pretrained=False, num_classes=10)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.maxpool = nn.Identity()

        elif arch == 'ResNet50':
            self.model = torchvision.models.resnet50(pretrained=False, num_classes=10)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.maxpool = nn.Identity()

        elif arch == 'ResNet34':
            self.model = torchvision.models.resnet34(pretrained=False, num_classes=10)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.maxpool = nn.Identity()

        elif arch == 'vgg_11_bn':
            self.model = torchvision.models.vgg11_bn(pretrained=False, num_classes=10)

        elif arch == 'vgg_11':
            self.model = torchvision.models.vgg11(pretrained=False, num_classes=10)

        elif arch == 'desnet':
            self.model = torchvision.models.densenet121(pretrained=False, num_classes=10)

        else:
            raise ValueError(f'"{arch}" is not a valid architecture configuration for the ResNet module.')

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)


class VAE(KFACSuperModule):

    def __init__(self, config):
        super().__init__(config)

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

