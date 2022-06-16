import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pytorch_lightning import LightningModule


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class ResNet(LightningModule):
    def __init__(self, config, network_name):
        super().__init__()

        self.optimizer_ = config['optimizer']
        self.is_kfac = 'kfac' in str(self.optimizer_).lower()
        if self.is_kfac:
            self.criterion1 = torch.nn.CrossEntropyLoss()
        self.optimizer_params = config['optimizer_params']
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        if network_name == 'ResNet18':
            self.model = torchvision.models.resnet18(pretrained=False, num_classes=10)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.maxpool = nn.Identity()
        if network_name == 'ResNet50':
            self.model = torchvision.models.resnet50(pretrained=False, num_classes=10)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.maxpool = nn.Identity()
        if network_name == 'ResNet34':
            self.model = torchvision.models.resnet34(pretrained=False, num_classes=10)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.maxpool = nn.Identity()
        if network_name=='vgg_11_bn':
            self.model = torchvision.models.vgg11_bn(pretrained=False, num_classes=10)
        if network_name=='vgg_11':
            self.model = torchvision.models.vgg11(pretrained=False, num_classes=10)
        if network_name == 'desnet':
            self.model = torchvision.models.densenet121(pretrained=False, num_classes=10)


    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        X, y = batch
        X_pred = self(X)
        loss = self.loss(X_pred, y)

        if self.is_kfac:

            optimizer = self.trainer.optimizers[0]
            if optimizer.steps % optimizer.TCov == 0:
                # compute true fisher
                optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(X_pred.data, dim=1),
                                                  1).squeeze()  # Potentially problematic
                loss_sample = self.criterion1(X_pred, sampled_y)
                loss_sample.backward(retain_graph=True)
                optimizer.acc_stats = False
                optimizer.zero_grad()  # clear the gradient for computing true-fisher.

        _, pred = torch.max(X_pred, 1)  # 预测最大值所在位置标签
        num_correct = (pred == y).sum()
        training_acc = num_correct.item()/y.size(0)

        self.log("train_loss", loss.item())
        self.log("train_acc", training_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X_pred = self(X)

        loss = self.loss(X_pred, y)

        _, pred = torch.max(X_pred.data, 1)
        num_correct = (pred == y).sum()
        training_acc = num_correct.item()/y.size(0)

        self.log("val_loss", loss.item())
        self.log("val_acc", training_acc)


    def configure_optimizers(self):
        optimizer_input = self if self.is_kfac else self.parameters()
        return self.optimizer_(optimizer_input, **self.optimizer_params)