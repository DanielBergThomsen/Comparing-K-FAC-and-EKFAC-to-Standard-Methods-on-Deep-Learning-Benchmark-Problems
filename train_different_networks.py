'''
This training file will reproduce our second experiment.
Two versions of KFAC will be tested for different networks.

To reproduce the plots in our report, please run this file three times with
different kfac_version and network_name
and go to plot_experiment_2.py to plot
'''

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from data import read_cifar10
import logging
from networks import ResNet
from optimizers.kfac_eigen import KFACOptimizer as KFACOptimizer_eigen
from optimizers.kfac_dia import KFACOptimizer as KFACOptimizer_dia

#Choose different networks
#network_name = 'ResNet18'
#network_name = 'desnet'
network_name = 'ResNet18'


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
NUM_GPUS = 1 if torch.cuda.is_available() else 0
DATALOADER_WORKERS = 0 if torch.cuda.is_available() else 6
MAX_EPOCHS = 100
BATCH_SIZE=128

configs = [
    {
        'label': 'KFAC_dia',
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
data_loader_train, data_loader_test = read_cifar10(BATCH_SIZE,PATH_DATASETS)

for config in configs:
    config_name = config['label']
    model = ResNet(config, network_name)

    logger = CSVLogger(save_dir=f'logs/experiment_2/{network_name}/', version="V0", name=config_name)
    trainer = Trainer(
            enable_model_summary=False,
            gpus=NUM_GPUS,
            max_epochs=MAX_EPOCHS,
            logger=logger
    )

    trainer.fit(model, data_loader_train, data_loader_test)



