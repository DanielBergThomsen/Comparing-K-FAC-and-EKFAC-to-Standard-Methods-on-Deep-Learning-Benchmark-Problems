'''
This training file will reproduce our second experiment.
Two versions of KFAC will be tested for different networks.

To reproduce the plots in our report, please run this file three times with
different kfac_version and network_name
and go to plot_experiment_2.py to plot
'''
from os.path import isdir
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import logging
from networks import ResNet
from optimizers.kfac_eigen import KFACOptimizer as KFACOptimizer_eigen
from optimizers.kfac_dia import KFACOptimizer as KFACOptimizer_dia
from torch.utils.data import DataLoader

# Choose different networks
#network_name = 'ResNet18'
#network_name = 'desnet'
ARCHITECTURE = 'ResNet18'

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)  # Silences internal Pytorch Lightning warnings

# Set up GPU (if available)
NUM_GPUS = 1 if torch.cuda.is_available() else 0
DATALOADER_WORKERS = 0 if torch.cuda.is_available() else 6


# Loss function for the ResNets
loss_fn = torch.nn.CrossEntropyLoss()
def loss_function(X_pred, X, y):
    return loss_fn(X_pred, y)


configs = [
    {
        'label': 'KFAC_dia',
        'architecture': ARCHITECTURE,
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
        'label': 'KFAC_eigen',
        'architecture': ARCHITECTURE,
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

# Dataloader settings
MAX_EPOCHS = 100
BATCH_SIZE = 128

# Load dataset
PATH_DATASETS = 'data/'
train_ds = CIFAR10(PATH_DATASETS, train=True, download=True, transform=Compose([ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])]))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=DATALOADER_WORKERS, shuffle=True,)
val_ds = CIFAR10(PATH_DATASETS, train=False, download=True, transform=Compose([ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])]))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=DATALOADER_WORKERS, shuffle=False)


for config in configs:

    # Config name
    config_name = config['label']

    # Initialize ResNet with the chosen config
    model = ResNet(config)

    # Train network
    logger = CSVLogger(save_dir=f'logs/experiment_2/{ARCHITECTURE}/', version="V0", name=config_name)
    trainer = Trainer(
            enable_model_summary=False,
            gpus=NUM_GPUS,
            max_epochs=MAX_EPOCHS,
            logger=logger
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)
