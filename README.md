# Comparing-K-FAC-and-EKFAC-to-Standard-Methods-on-Deep-Learning-Benchmark-Problem
Repository for project in the course "Optimization for Machine Learning" (CS-439) at EPFL.

## Team member ##
Daniel Berg Thomsen: 331375

Xianjie Dai: 336675

Artur Stefaniuk: 274071

## Introduction ##
This project includes two experiments: 
  1. reproducing part of the first experiment in the original paper of KFAC
  2. testing the performance of KFAC on different networks

The first experiment is conducted on a simple auto-encoder with MNIST. The auto-encoder consists of an encoder of size (28*28)-1000-500-250-30 and a symmetric decoder. The first experiment tested the relationship between different mini-batch sizes and per-iteration progress. We followed the experiment and set mini-batch sizes to 2000, 4000, and 6000. Notice that Martens & Grosse 2015 did not provide any results for K-FAC with diagonal approximation because of the unsatisfying performance. We were curious about the actual performance of diagonal approximation and reproduced the omitted experiments. Our results are plotted using log-scale. We also choose the version of SGD with momentum based on Nesterov’s Accelerated as our baseline.

The second experiment is conducted on different networks with CIFAR-10. We selected three classic model: ResNet18, ResNet50, and DesNet121.

## How to use ##
Each experiments corrosponds to train_autoencoder.py and train_different_networks.py seperately.
To reproduce our experiemnts results, please run train_autoencoder.py and train_different_networks.py first to:
  1. download MNIST or CIFAR-10
  2. choose mini-batch size (and the network architecture to use in the second experiment)
  3. Training logs are automatically saved in the logs folder so finally, run [plot_experiment_1.ipynb](https://github.com/DanielBergThomsen/Comparing-K-FAC-and-EKFAC-to-Standard-Methods-on-Deep-Learning-Benchmark-Problems/blob/main/plot_experiment_1.ipynb), and [plot_experiment_2.py](https://github.com/DanielBergThomsen/Comparing-K-FAC-and-EKFAC-to-Standard-Methods-on-Deep-Learning-Benchmark-Problems/blob/main/plot_experiment_2.py) to reproduce our experiments results.

The reason why we seperate training and plotting in different files is that training is computationally expensive, which usually takes hours to train on a GPU, and plotting only takes seconds to finish. It would be much convenient to check the reproducibility by seperating training and plotting.

**We have already provided our trianing logs in log file, train_autoencoder.py and train_different_networks.py will clear and overwrite training logs.**

Here is our suggestion:
  1. to simply reproduce the plots in our report, please run [plot_experiment_1.ipynb](https://github.com/DanielBergThomsen/Comparing-K-FAC-and-EKFAC-to-Standard-Methods-on-Deep-Learning-Benchmark-Problems/blob/main/plot_experiment_1.ipynb), and [plot_experiment_2.py](https://github.com/DanielBergThomsen/Comparing-K-FAC-and-EKFAC-to-Standard-Methods-on-Deep-Learning-Benchmark-Problems/blob/main/plot_experiment_2.py) to reproduce our experiments results first.
  2. to reproduce the whole project, please run train_autoencoder.py and train_different_networks.py first, and use plot_experiment_1.py and plot_experiment_2.py to plot.

logs: all our experiment results can be found here

networks.py: different networks implementation

data.py: CIFAR-10 dataloader

autoencoder.py: auto-encoder implementation

optimizers: two different versions of KFAC, based on alecwangcq's implementation: https://github.com/alecwangcq/KFAC-Pytorch

## Acknowledgement ##
Our implementation of KFAC with diagonal approximation is based on alecwangcq's implementation, which can be found at https://github.com/alecwangcq/KFAC-Pytorch. It was inspired by gpauloski implementation (https://github.com/gpauloski/kfac-pytorch).
Also, please consider cite the original paper of KFAC:

@inproceedings{martens2015optimizing,

  title={Optimizing neural networks with kronecker-factored approximate curvature},
  
  author={Martens, James and Grosse, Roger},
  
  booktitle={International conference on machine learning},
  
  pages={2408--2417},
  
  year={2015}
  
}
