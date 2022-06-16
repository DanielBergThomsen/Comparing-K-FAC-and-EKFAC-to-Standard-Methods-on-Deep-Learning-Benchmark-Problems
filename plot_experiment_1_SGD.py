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
BATCH_SIZE = 2000


'''
The number of epochs will change according to batch size to make sure that the total number of iterations equals to 3000
Max epoch = total_number_of_iterations / number_of_iterations_per_epoch
'''
MAX_EPOCHS = int(15000/(60000/BATCH_SIZE))

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
NUM_GPUS = 1 if torch.cuda.is_available() else 0
DATALOADER_WORKERS = 0 if torch.cuda.is_available() else 6


#SGD_optimizer = optim.SGD(vae.parameters(),lr=learning_rate, weight_decay=weight_decay_rate, momentum =momentum, nesterov =True)
#Adam_optimizer = optim.Adam()

configs =[{
        'label': 'SGD_2000'
    },
    {
        'label': 'SGD_4000'
    },
    {
        'label': 'SGD_6000'
    }
    ]

training_pf = []

for config in configs:
    # Config name
    config_name = config['label']
    training_logs = []

    # Load training logs
    train_df = pd.read_csv(f'logs/experiment_1/{config_name}/V0/metrics.csv')
    train_loss = train_df[~train_df.train_loss.isnull()][['train_loss', 'step']]
    train_loss['Optimizer'] = config_name
    training_logs += train_loss.T.to_dict().values()
    training_df = pd.DataFrame(training_logs)
    training_df['train_loss'] = np.log(training_df['train_loss'])
    training_pf.append(training_df)



matplotlib.use('TKAgg')
    # Add training losses

final_df = pd.merge(training_pf[0], training_pf[1], how='left', on='step')
final_df = pd.merge(final_df, training_pf[2], how='left', on='step')
final_df.set_index(['step'], inplace=True)

final_df.rename(columns = {'train_loss_x':'SGD_2000', 'train_loss_y':'SGD_4000', 'train_loss':'SGD_6000'}, inplace = True)
print(final_df)

plt.close()
sns.lineplot(data=final_df)
plt.show()
