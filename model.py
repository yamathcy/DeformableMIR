from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import pytorch_lightning as pl
from conv_modules import *


class Basemodel(pl.LightningModule):
    @abstractmethod 
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        return super().forward(*args, **kwargs)
    

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
        self.conv3 = conv(in_channels=channels[1], out_channels=channels[2],kernel_size=kernel_size[2],pooling=pooling[2],sconv=sconv)
        self.conv4 = conv(in_channels=channels[2], out_channels=channels[3],kernel_size=kernel_size[3],pooling=pooling[3],sconv=sconv)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(channels[3], 50)
        self.norm = nn.InstanceNorm2d(50)
        self.out_nl = nn.ReLU()
        self.dropout = nn.Dropout()
        self.act = nn.Linear(50, num_classes)
    
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
        x = self.act(x)
        return x.squeeze(), feature.squeeze()



