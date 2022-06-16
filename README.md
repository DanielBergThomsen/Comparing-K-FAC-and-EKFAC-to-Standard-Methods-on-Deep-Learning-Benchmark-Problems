# CS-439
Repository for project in the course "Optimization for Machine Learning" (CS-439) at EPFL.

## Team member ##
Daniel Berg Thomsen:
Xianjie Dai: 336675
Artur Stefaniuk:

## Introduction ##
This project includes two experiments: 
  1. reproducing part of the first experiment in the original paper of KFAC
  2. testing the performance of KFAC on different networks

The first experiment is conducted on a simple auto-encoder with MNIST. The auto-encoder consists of an encoder of size (28*28)-1000-500-250-30 and a symmetric decoder. The first experiment tested the relationship between different mini-batch sizes and per-iteration progress. We followed the experiment and set mini-batch sizes to 2000, 4000, and 6000. Notice that \cite{kfac_martens} did not provide any results for K-FAC with diagonal approximation because of the unsatisfying performance. We were curious about the actual performance of diagonal approximation and reproduced the omitted experiments. Our results are plotted using log-scale.

The second experiment is conducted on different networks with CIFAR-10. We selected three classic model: ResNet18, ResNet50, and DesNet121.

## How to use ##
Each experiments corrosponds to train_autoencoder.py and train_different_networks.py seperately.
To reproduce our experiemnts results, please run these two files first to:
  1. download MNIST or CIFAR-10
  2. choose KFAC version and mini-batch size or network names
  3. save training logs in log file
Finally, run plot_experiment_1.py and plot_experiment_2.py to reproduce our experiments results. 
**We have already provided our trianing logs in log file, train_autoencoder.py and train_different_networks.py will clear and overwrite training logs.**
Here is our suggestion:
  1. to simply reproduce the plots in our report, please run plot_experiment_1.py and plot_experiment_2.py to reproduce our experiments results first.
  2. to reproduce the whole project, please run train_autoencoder.py and train_different_networks.py first, and use plot_experiment_1.py and plot_experiment_2.py to plot.
