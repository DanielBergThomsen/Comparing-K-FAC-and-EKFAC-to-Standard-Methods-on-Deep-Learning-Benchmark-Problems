import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from autoencoder import AE
from tensorboardX import SummaryWriter

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

epochs = 200
bs = 500
learning_rate = 1e-2
weight_decay_rate = 1e-5
momentum=0.99
initial_loss = 0

# MNIST Dataset
'''
train_dataset = datasets.MNIST(root='./data/MNIST/', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]), download=True)#[-1,1]
test_dataset = datasets.MNIST(root='./data/MNIST/', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]), download=False)#[-1,1]
'''

train_dataset = datasets.MNIST(root='./data/MNIST/', train=True, transform=transforms.ToTensor(), download=True)#[0,1]
test_dataset = datasets.MNIST(root='./data/MNIST/', train=False, transform=transforms.ToTensor(), download=False)#[0,1]


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


# build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AE().to(device)
optimizer = optim.SGD(vae.parameters(),lr=learning_rate, weight_decay=weight_decay_rate, momentum =momentum, nesterov =True)
criterion = nn.BCELoss(reduction='mean').to(device)

# Tensorboard Writer
writer = SummaryWriter(log_dir='./runs/SGD_500')




def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        data = data.to(device)


        optimizer.zero_grad()
        output = vae(data)
        loss = criterion(output,data)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()*bs

        if batch_idx % 120 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))

    writer.add_scalar('trainning loss (iteration)', np.log10(train_loss / len(train_loader.dataset)), (epoch+1) * len(train_loader))
    print('====> Epoch: {} Total loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    print('====> Epoch: {} Log total loss: {:.4f}'.format(epoch, np.log10(train_loss / len(train_loader.dataset)), (epoch+1) * len(train_loader)))
    '''
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/ len(train_loader.dataset)))
    print('====> Epoch: {} Log average loss: {:.4f}'.format(epoch, np.log10(train_loss/ len(train_loader.dataset))))
    '''

    if epoch==(epochs-1):
        torch.save(vae.state_dict(), './model/SGD_500.pth')

    '''
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            data = data.to(device)
            output = vae(data)

            # sum up batch loss
            test_loss += criterion(output,data).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    '''
for batch_idx, (data, _) in enumerate(train_loader):
    data = data.view(data.size(0), -1)
    data = data.to(device)
    output = vae(data)
    loss = criterion(output, data)
    initial_loss += loss.item()*bs
writer.add_scalar('trainning loss (iteration)', np.log10(initial_loss / len(train_loader.dataset)), 0)
print('====> Epoch: {} Total loss: {:.4f}'.format(0, initial_loss / len(train_loader.dataset)))
print('====> Epoch: {} Log total loss: {:.4f}'.format(0, np.log10(initial_loss / len(train_loader.dataset))))

for epoch in range(0, epochs):
    train(epoch)
writer.close()

#tensorboard --logdir=E:\pythonProject\task2\runs\autoencoder

