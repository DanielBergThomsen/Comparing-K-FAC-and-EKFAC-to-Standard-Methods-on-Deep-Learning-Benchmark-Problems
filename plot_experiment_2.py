import matplotlib
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Choose different networks
#network_name = 'ResNet50'
#network_name = 'desnet'
network_name = 'ResNet18'

configs = [
    {
        'label': f'KFAC_dia'
    },
    {
        'label': f'KFAC_eigen'
    }
]

# Plot
for config in configs:
    validation_logs = []
    config_name = config['label']

    # Load training logs
    train_df = pd.read_csv(f'logs/experiment_2/{network_name}/{config_name}/V0/metrics.csv')

    # Add validation accuracy
    valid_loss = train_df[~train_df.val_loss.isnull()][['val_acc', 'step', 'epoch']]
    valid_loss['Optimizer'] = config_name
    valid_loss= valid_loss[valid_loss['epoch']<100]
    validation_logs += valid_loss.T.to_dict().values()

    validation_df = pd.DataFrame(validation_logs)
    print(validation_df['val_acc'].max())

    plt.close()
    sns.lineplot(data=validation_df, x='epoch', y='val_acc', hue='Optimizer')
    plt.show()
    plt.savefig(f'experiment_2_{network_name}_{config_name}.png')
