from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import pytorch_lightning as pl
from conv_modules import *
import numpy as np
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def conv(deform,**kwrds):
    if deform:
        # print(**kwrds)
        return DeformConvBlock(**kwrds)
    else:
        return ConvBlock(**kwrds)
    

class CNN(nn.Module):
    def __init__(self, channels, kernel_size, pooling=[2,2,2,2],sconv=True,num_classes=10,deform=False):
        super().__init__()
        # print(channels)
        self.inp = MultiResolutionMelSpec(n_mels=128)
        self.conv1 = ConvBlock(in_channels=3, out_channels=channels[0],kernel_size=kernel_size[0],pooling=pooling[0],sconv=sconv)
        self.conv2 = ConvBlock(in_channels=channels[0], out_channels=channels[1],kernel_size=kernel_size[1],pooling=pooling[1],sconv=sconv)
        self.deform = True if deform else False
        self.conv3 = conv(in_channels=channels[1], out_channels=channels[2],kernel_size=kernel_size[2],pooling=pooling[2],sconv=sconv,deform=deform)
        self.conv4 = conv(in_channels=channels[2], out_channels=channels[3],kernel_size=kernel_size[3],pooling=pooling[3],sconv=sconv,deform=deform)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(channels[3], 50)
        self.norm = nn.InstanceNorm2d(50)
        self.out_nl = nn.ReLU()
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(50, num_classes)
    
    def forward(self,x):
        """_summary_

        Args:
            x (torch.tensor): torch tensor of raw audio, (B, C, L) (3D)

        Returns:
            x: output prediction
        """
        x = self.inp(x)
        # print(x.shape)
        # plt.pcolormesh(x[0,0,:,:].detach().numpy(),cmap="magma")
        # plt.show()

        x = self.conv1(x)
        # print(x.shape)
        # plt.title("x1, size:{}".format(x.shape))
        # plt.pcolormesh(x[0,0,:,:].detach().numpy(),cmap="magma")
        # plt.show()

        x = self.conv2(x)
        # print(x.shape)
        # plt.title("x2, size:{}".format(x.shape))
        # plt.pcolormesh(x[0,0,:,:].detach().numpy(),cmap="magma")
        # plt.show()

        x = self.conv3(x)  
        # print(x.shape) 
        # plt.pcolormesh(x[0,0,:,:].detach().numpy(),cmap="magma")
        # plt.title("x3, size:{}".format(x.shape))
        # plt.show()

        x = self.conv4(x)
        # print(x.shape)
        # plt.pcolormesh(x[0,0,:,:].detach().numpy(),cmap="magma")
        # plt.title("x4, size:{}".format(x.shape))
        # plt.show()
        x = self.gap(x)
        x = x.squeeze(-1)
        x = x.permute(0,2,1)
        x = self.linear(x)
        feature = x
        x = self.out_nl(self.norm(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x.squeeze(), feature.squeeze()
    



class PlModel(pl.LightningModule):
    """
    Baseline model 
    borrowed from 
    https://github.com/bill317996/Singer-identification-in-artist20/blob/master
    """
    def __init__(self, param, classes_num, class_weights, retrain):
        super().__init__()
        self.lr = param.lr
        self.num_classes = classes_num
        self.softmax = nn.Softmax(dim=1)
        self.net = CNN(param.filters,kernel_size=param.kernel_size,pooling=param.pooling, sconv=param.sconv, num_classes=classes_num,deform=param.deform)

        self.class_weights = class_weights
        self.retrain = retrain
        self.train_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.val_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_acc = Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        self.test_top2 = Accuracy(num_classes=self.num_classes, average='macro', top_k=2, task='multiclass')
        self.test_top3 = Accuracy(num_classes=self.num_classes, average='macro', top_k=3, task='multiclass')
        self.test_f1 = F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        self.confusion = ConfusionMatrix(num_classes=self.num_classes, task='multiclass')

    def forward(self, x):
        x, emb = self.net(x)
        return x, emb

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # print(x.shape)
        out,_ = self(x)
        if self.retrain:
            class_weights = self.class_weights.to(DEVICE)
        else: 
            class_weights = None
        loss = F.cross_entropy(out, y, weight=class_weights)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc(out, y), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out,_ = self(x)
        class_weights=self.class_weights.to(DEVICE)
        loss = F.cross_entropy(out, y, weight=class_weights)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_acc', self.val_acc(out, y), on_step=False, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out,_ = self(x)
        self.log('test_accuracy', self.test_acc(out,y), on_epoch=True, on_step=False)
        self.log('test_f1', self.test_f1(out, y), on_epoch=True, on_step=False)
        self.log('test_top2_accuracy', self.test_top2(out, y), on_epoch=True, on_step=False)
        self.log('test_top3_accuracy', self.test_top3(out, y), on_epoch=True, on_step=False)
        self.log('test_confusion', self.confusion(out, y), on_epoch=False, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict(self, x):
        self.eval()
        out, _ = self.forward(x)
        print(out.shape)
        out = torch.argmax(out, dim=1)
        out = out.cpu().detach().numpy().copy()
        # out = np.squeeze(out)
        return out

    def predict_proba(self, x):
        """

        :param x: input tensor (torch.Tensor)
        :return: single output of model (numpy.array)
        """
        self.eval()
        out, _ = self.forward(x)
        out = torch.softmax(out, dim=1)  # assuming logits has the shape [batch_size, nb_classes]
        out = out.cpu().detach().numpy().copy()
        out = np.squeeze(out)
        return out