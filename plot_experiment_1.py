import matplotlib
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torchvision.datasets import MNIST
from torchvision import transforms
from os.path import isdir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from autoencoder import VAE
import logging
from optimizers.kfac_dia import KFACOptimizer as KFACOptimizer_dia
from optimizers.kfac_eigen import KFACOptimizer as KFACOptimizer_eigen


#Choose K-FAC from eigendecomposition or digonal approximation
#kfac_version = 'eigen'
kfac_version = 'dia'

# Change if training with GPU
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
NUM_GPUS = 1 if torch.cuda.is_available() else 0
DATALOADER_WORKERS = 0 if torch.cuda.is_available() else 6

configs =[
    {
        'label': f'KFAC_{kfac_version}_2000'
    },
    {
        'label': f'KFAC_{kfac_version}_4000'
    },
    {
        'label': f'KFAC_{kfac_version}_6000'
    }]
DF = []

for config in configs:
    # Config name
    config_name = config['label']

    # Plot
    training_logs = []

    # Load training logs
    train_df = pd.read_csv(f'logs/experiment_1/{config_name}/V0/metrics.csv')

    matplotlib.use('TKAgg')
    # Add training losses
    train_loss = train_df[~train_df.train_loss.isnull()][['train_loss', 'step']]
    train_loss['Optimizer'] = config_name
    training_logs += train_loss.T.to_dict().values()
    training_df = pd.DataFrame(training_logs)
    DF.append(training_df)

final_df = pd.merge(DF[2], DF[1], how='left', on='step')
final_df = pd.merge(final_df, DF[0], how='left', on='step')
final_df.set_index(['step'], inplace=True)
cols = ['train_loss','train_loss_x', 'train_loss_y']
final_df[cols] = np.log(final_df[cols])
final_df.rename(columns = {'train_loss':'KFAC_2000', 'train_loss_x':'KFAC_6000', 'train_loss_y':'KFAC_4000'}, inplace = True)
print(final_df)
plt.close()

sns.lineplot(data=final_df)
plt.show()
plt.savefig(f'experiment_1_{kfac_version}.png')
