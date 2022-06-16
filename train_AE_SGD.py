'''
This training file will reproduce part of the first experiment in the original paper of KFAC.
Two versions of KFAC will be tested for fixed mini-batch size.
The total number of iterations is 3000, which is given in the original paper.
Batch size can be selected from 2000, 4000, and 6000

To reproduce the plots in our report, please run this file three times with
BATCH_SIZE = 2000
BATCH_SIZE = 4000
BATCH_SIZE = 6000
and go to plot_experiment_1.py to plot
'''
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torchvision.datasets import MNIST
from torchvision import transforms
from os.path import isdir
from autoencoder import VAE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
import numpy as np
import torch
import logging

#Change batch size for different experiments: 2000, 4000, and 6000
BATCH_SIZE = 6000


'''
The number of epochs will change according to batch size to make sure that the total number of iterations equals to 3000
Max epoch = total_number_of_iterations / number_of_iterations_per_epoch
'''
MAX_EPOCHS = int(10000/(60000/BATCH_SIZE))

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
NUM_GPUS = 1 if torch.cuda.is_available() else 0
DATALOADER_WORKERS = 0 if torch.cuda.is_available() else 6

learning_rate = 1e-2
weight_decay_rate = 1e-5
momentum=0.99

#SGD_optimizer = optim.SGD(vae.parameters(),lr=learning_rate, weight_decay=weight_decay_rate, momentum =momentum, nesterov =True)
#Adam_optimizer = optim.Adam()

config ={
        'label': f'SGD_{BATCH_SIZE}',
        'optimizer': torch.optim.SGD,
        'optimizer_params': {'lr': learning_rate,
                             'momentum': momentum,
                             'nesterov': True,
                             'weight_decay': weight_decay_rate
                             }
    }

# Load dataset
PATH_DATASETS = 'data/'
download = isdir(PATH_DATASETS + 'MNIST')
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=DATALOADER_WORKERS)
val_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=DATALOADER_WORKERS)


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
'''
training_logs = []

    # Load training logs
train_df = pd.read_csv(f'logs/experiment_1/{config_name}/V0/metrics.csv')

matplotlib.use('TKAgg')
    # Add training losses
train_loss = train_df[~train_df.train_loss.isnull()][['train_loss', 'step']]
train_loss['Optimizer'] = config_name
training_logs += train_loss.T.to_dict().values()
training_df = pd.DataFrame(training_logs)
training_df['train_loss'] = np.log(training_df['train_loss'] )
plt.close()

sns.lineplot(data=training_df, x='step', y='train_loss', hue='Optimizer')
plt.show()
'''