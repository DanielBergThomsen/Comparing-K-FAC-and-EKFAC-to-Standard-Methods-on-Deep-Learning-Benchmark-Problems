"""
This training file will reproduce part of the first experiment in the original paper of KFAC.
Two versions of KFAC will be tested for fixed mini-batch size.
The total number of iterations is 3000, which is given in the original paper.
Batch size can be selected from 2000, 4000, and 6000

To reproduce the plots in our report, please run this file three times with
BATCH_SIZE = 2000
BATCH_SIZE = 4000
BATCH_SIZE = 6000
and go to plot_experiment_1.ipynb to plot
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda, Compose
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from os.path import isdir
from networks import VAE
from optimizers.kfac_eigen import KFACOptimizer as KFACOptimizer_eigen
from optimizers.kfac_dia import KFACOptimizer as KFACOptimizer_dia
import logging

# Change batch size for different experiments: 2000, 4000, and 6000
BATCH_SIZE = 2000
'''
The number of epochs will change according to batch size to make sure that the total number of iterations equals to 3000
Max epoch = total_number_of_iterations / number_of_iterations_per_epoch
'''
MAX_EPOCHS = int(30000/(60000/BATCH_SIZE))

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)  # Silences internal Pytorch Lightning warnings

# Set up GPU (if available)
NUM_GPUS = 1 if torch.cuda.is_available() else 0
DATALOADER_WORKERS = 0 if torch.cuda.is_available() else 6


# Loss function for the VAE
def loss_function(X_pred, X, _):
    return F.binary_cross_entropy_with_logits(X_pred.view(X_pred.size(0), -1), X.view(X.size(0), -1), reduction='mean')


configs = [
    {
        'label': f'SGD_{BATCH_SIZE}',
        'loss_function': loss_function,
        'optimizer': torch.optim.SGD,
        'optimizer_params': {'lr': 1e-2,
                             'momentum': 0.99,
                             'nesterov': True,
                             'weight_decay': 1e-5}
    },
    {
        'label': f'Adam_{BATCH_SIZE}',
        'loss_function': loss_function,
        'optimizer': torch.optim.Adam,
        'optimizer_params': {'lr': 1e-2,
                             'weight_decay': 1e-5}
    },
    {
        'label': f'KFAC_dia_{BATCH_SIZE}',
        'loss_function': loss_function,
        'optimizer': KFACOptimizer_dia,
        'optimizer_params': {'lr': 1e-2,
                             'momentum': 0.9,
                             'stat_decay': 0.99,
                             'damping': 1e-3,
                             'kl_clip': 1e-2,
                             'weight_decay': 1e-5,
                             'TCov': 10,
                             'TInv': 100}
    },
    {
        'label': f'KFAC_eigen_{BATCH_SIZE}',
        'loss_function': loss_function,
        'optimizer': KFACOptimizer_eigen,
        'optimizer_params': {'lr': 1e-2,
                             'momentum': 0.9,
                             'stat_decay': 0.99,
                             'damping': 1e-3,
                             'kl_clip': 1e-2,
                             'weight_decay': 1e-5,
                             'TCov': 10,
                             'TInv': 100}
    }
]
# Load dataset
PATH_DATASETS = 'data/'
download = isdir(PATH_DATASETS + 'MNIST')
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=Compose([ToTensor(), Lambda(torch.flatten)]))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=DATALOADER_WORKERS)
val_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=Compose([ToTensor(), Lambda(torch.flatten)]))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=DATALOADER_WORKERS)


# Run and train for each config
for config in configs:

    # Config name
    config_name = config['label']

    # Initialize VAE with the chosen config
    vae = VAE(config)

    # Train network
    logger = CSVLogger(save_dir='logs/experiment_1/', version="V0", name=config_name)
    trainer = Trainer(
        enable_model_summary=False,
        gpus=NUM_GPUS,
        max_epochs=MAX_EPOCHS,
        logger=logger
    )

    # Train the model
    trainer.fit(vae, train_loader, val_loader)
