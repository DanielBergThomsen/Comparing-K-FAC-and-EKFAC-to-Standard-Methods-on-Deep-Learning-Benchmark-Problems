import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from autoencoder import AE
from tensorboardX import SummaryWriter
from optimizers import KFACOptimizer
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

epochs = 200
bs = 500
learning_rate = 1e-2
weight_decay_rate = 1e-5
initial_loss=0

# MNIST Dataset
'''
train_dataset = datasets.MNIST(root='./data/MNIST/', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]), download=True)
test_dataset = datasets.MNIST(root='./data/MNIST/', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]), download=False)
'''

train_dataset = datasets.MNIST(root='./data/MNIST/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/MNIST/', train=False, transform=transforms.ToTensor(), download=False)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AE().to(device)
criterion1 = nn.CrossEntropyLoss().to(device)
criterion2 = nn.BCELoss(reduction='mean').to(device)
optimizer = KFACOptimizer(vae,
                          lr=learning_rate,
                          momentum=0,
                          stat_decay=0.99,
                          damping=1e-3,
                          kl_clip=1e-2,
                          weight_decay=weight_decay_rate,
                          TCov=10,
                          TInv=100)
#scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1)

# Tensorboard Writer
writer = SummaryWriter(log_dir='./runs/kfac_test_500_0.01')


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        data = data.to(device)

        vae.zero_grad()
        output = vae(data)


        if optimizer.steps % optimizer.TCov == 0:
            # compute true fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1),
                                              1).squeeze().cuda()
            loss_sample = criterion1(output, sampled_y)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true-fisher.

        loss = criterion2(output, data)
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


    if epoch==(epochs-1):
        torch.save(vae.state_dict(), './model/kfac_500.pth')

    #scheduler.step()
    '''
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1)
            data = data.to(device)
            output = vae(data)

            # sum up batch loss
            test_loss += criterion2(output,data).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    '''

for batch_idx, (data, _) in enumerate(train_loader):
    data = data.view(data.size(0), -1)
    data = data.to(device)
    output = vae(data)
    loss = criterion2(output, data)
    initial_loss += loss.item()*bs
writer.add_scalar('trainning loss (iteration)', np.log10(initial_loss / len(train_loader.dataset)), 0)
print('====> Epoch: {} Total loss: {:.4f}'.format(0, initial_loss / len(train_loader.dataset)))
print('====> Epoch: {} Log total loss: {:.4f}'.format(0, np.log10(initial_loss / len(train_loader.dataset))))
for epoch in range(0, epochs):
    train(epoch)
writer.close()

