
from multiprocessing import pool
import numpy as np
from scipy.misc import derivative
import torch
from torch import device
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import torchvision
from torchmetrics import F1Score, Accuracy, ConfusionMatrix

from types import SimpleNamespace
from torchinfo import summary
from abc import abstractmethod
import pytorch_lightning as pl
import torchmetrics 
from torch.optim import Adam, RMSprop
from util import DEVICE, SEED, Config
import mlflow
from transformers import AutoModel, AutoFeatureExtractor

class BaseModel(pl.LightningModule):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model # prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return super().__str__() + '\nTrainable parameters: {}'.format(params)


    def view_summary(self, input_size):
        summary(self, input_size=input_size)

    def get_feature(self, z):
        z = z.to(DEVICE)
        _,feature =  self.forward(z)
        if type(feature) == tuple:
            feature = feature[0]
        return feature
    
    def predict(self, x):
        """

        :param x: input tensor (torch.Tensor)
        :return: single output of model (numpy.array)
        """

        self.eval()
        out,_ = self.forward(x)
        out = torch.argmax(out, dim=1)
        out = out.cpu().detach().numpy().copy()
        #out = np.squeeze(out)
        return out

    def predict_proba(self, x):
        """

        :param x: input tensor (torch.Tensor)
        :return: single output of model (numpy.array)
        """
        self.eval()
        out,_ = self.forward(x)
        out = F.softmax(out, dim=1) # assuming logits has the shape [batch_size, nb_classes]
        out = out.cpu().detach().numpy().copy()
        out = np.squeeze(out)
        return out
    


class Conv_Oblong(nn.Module):
    # oblong shape kernel
    def __init__(self, input_channels, output_channels, vertical_shape, horizontal_shape, pool_size, drop_out=0.3, deform=False):
        super(Conv_Oblong, self).__init__()
        if deform:
            self.conv = DeformableConv2d(input_channels, output_channels, kernel_size=(vertical_shape, horizontal_shape),
                              padding=[vertical_shape//2, horizontal_shape // 2])
             
        else:
            self.conv = nn.Conv2d(input_channels, output_channels, (vertical_shape, horizontal_shape),
                              padding=[vertical_shape//2, horizontal_shape // 2])
            

        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.pool_size=pool_size
        if pool_size is None:
            self.maxpool = nn.Identity()
        else:
            self.maxpool= nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        freq = x.size(2)
        
        out = self.maxpool(x)
        out = self.dropout(out)
        out = out.squeeze(2)
        return out


class Conv_V(nn.Module):
    # vertical convolution
    def __init__(self, input_channels, output_channels, filter_shape):
        super(Conv_V, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, filter_shape,
                              padding=(0, filter_shape[1]//2))
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        freq = x.size(2)
        out = nn.MaxPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        return out


class Conv_H(nn.Module):
    # horizontal convolution
    def __init__(self, input_channels, output_channels, filter_length):
        super(Conv_H, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, filter_length,
                              padding=filter_length//2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        freq = x.size(2)
        out = nn.AvgPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        out = self.relu(self.bn(self.conv(out)))
        return out

class Conv1DBlock(nn.Module):
    def __init__(self, input_channels, output_channels, filter_length, mp_kernel=64, mp_stride=8):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, filter_length, 
                              padding=0, stride=1)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=mp_kernel, stride=mp_stride)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x
    


class MultiResolutionSpec(nn.Module):
    def __init__(self, win_length_list=[2048, 1024, 512], hop_size=512, power=1.0,
                 normalized=False):
        self.stacked_channels = len(win_length_list)
        self.n_fft_list = win_length_list
        self.win_length_list = win_length_list
        self.hop_size = hop_size
        self.power = power
        self.normalized = normalized

        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "m" and "l" specifying the output of the dimensionality reducing s convolutions
            c_out - Dictionary with keys "s", "m", "l", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super(MultiResolutionSpec, self).__init__()
        self.layer_list = []
        self.n_fft = max(self.win_length_list)
        self.first_layer = nn.Sequential()
        self.first_layer.add_module('spec', torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.win_length_list[0],
                                                  hop_length=hop_size, pad=0,
                                                  window_fn=torch.hann_window,
                                                  power=self.power, normalized=self.normalized, wkwargs=None),)
        self.first_layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        #self.first_layer.add_module('channel', GenerateChannnelDim())

        self.second_layer = nn.Sequential()
        self.second_layer.add_module('spec', torchaudio.transforms.Spectrogram(n_fft=self.n_fft,
                                                                              win_length=self.win_length_list[1],
                                                                              hop_length=hop_size, pad=0,
                                                                              window_fn=torch.hann_window,
                                                                              power=self.power,
                                                                              normalized=self.normalized,
                                                                              wkwargs=None), )
        self.second_layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        #self.second_layer.add_module('channel', GenerateChannnelDim())

        self.third_layer = nn.Sequential()
        self.third_layer.add_module('spec', torchaudio.transforms.Spectrogram(n_fft=self.n_fft,
                                                                              win_length=self.win_length_list[2],
                                                                              hop_length=hop_size, pad=0,
                                                                              window_fn=torch.hann_window,
                                                                              power=self.power,
                                                                              normalized=self.normalized,
                                                                              wkwargs=None), )
        self.third_layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        #self.third_layer.add_module('channel', GenerateChannnelDim())

        # for win_length in self.win_length_list:
        #     if win_length == self.n_fft: continue
        #     self.layer_list.append(
        #         torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=win_length,
        #                                           hop_length=None, pad=0,
        #                                           window_fn=torch.hann_window,
        #                                           power=self.power, normalized=self.normalized, wkwargs=None),
        #     )

    def forward(self,x):
        spec1 = self.first_layer(x)
        spec2 = self.second_layer(x)
        spec3 = self.third_layer(x)
        # print(spec1.shape, spec2.shape, spec3.shape)
        stacked = torch.cat([spec1, spec2, spec3], 1)
        return stacked

def frontend_interface(mode, param) -> nn.Module:
    # allow ['square', 'cascade-narrow', 'cascade-wide', 'bibranch-trans', 'bibranch-cis', 'musicnn', 'dense', 'multi-dense', 'inception']
    frontend_list = ['stft', 'mel', 'multi']
    if not mode in frontend_list: raise NotImplementedError
    n_fft = param.n_fft
    win_length = param.win_length
    power = 1.0
    normalized = False
    sr = param.sr
    hop_length = param.hop_length
    f_min = 0.0
    f_max = param.f_max
    n_mels = param.n_mels
    layer = nn.Sequential()
    if mode == 'stft':
        layer.add_module('spec', torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length,
                                                      hop_length=None,
                                                      window_fn=torch.hann_window,
                                                      power=power, normalized=normalized, wkwargs=None))

        layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        #layer.add_module('channel', GenerateChannnelDim())
        # layer.add_module('bn', nn.BatchNorm2d(1))
        output_dim = 1

    elif mode == 'mel':
        layer.add_module('spec',torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels))

        layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        #layer.add_module('channel', GenerateChannnelDim())
        # layer.add_module('bn', nn.BatchNorm2d(1))
        output_dim = 1

    elif mode == 'multi':
        # todo: implement hyper parameter input
        # win_length_list = [2048, 1024, 512], hop_siaze=512
        layer.add_module('spec', MultiResolutionSpec(win_length_list=[2048, 1024, 512], hop_size=512,normalized=True))
        # layer.add_module('bn', nn.BatchNorm2d(3))
        output_dim = 3

    return layer, output_dim

class GenerateChannnelDim(nn.Module):
    def __init__(self, *args):
        super(GenerateChannnelDim, self).__init__()
        self.shape = args

    def forward(self, x):
        x = x.unsqueeze(1)
        return x


class CascadeCNN(BaseModel):
    def __init__(self,freq:int,time:int,channel:int, weights:dict or list, param:Config, kernel_list=[(4,1), (16,1), (1,4), (1,16)], pool_list=[(4,4),(4,4),(3,3),(2,2)], dense_dim=50, feature_dim=30, class_num=10, aggr='gap', batch_size=64, deform=False):

        super(CascadeCNN, self).__init__()
        self.dense_dim=dense_dim
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.is_fit = False
        self.batch_size =batch_size
        self.param = param
        self.aggr = param.aggr
        self.pool_list = pool_list
        # input_shape = (freq, time, channel)

        # deform
        if param.deform:
            if param.deform_list == 'all':
                deform = [True, True, True, True]
                self.pool_list = [None, None, None, None]

            elif param.deform_list == 'last':
                deform = [False, False, False, True]
                self.pool_list = [self.pool_list[0],self.pool_list[1],self.pool_list[2],None]

            elif param.deform_list == 'temporal':
                deform = [False, False, True, True]
                self.pool_list = [self.pool_list[0], self.pool_list[1], None, None]

            elif param.deform_list == 'timbral':
                deform = [True, True, False, False]
                self.pool_list = [None, None, self.pool_list[2], self.pool_list[3]]

            else:
                raise NotImplementedError
        else:
            deform = [False, False, False, False]

        # deform = deform if type(deform) == list else [deform, deform, deform, deform]

        self.spec,_ = frontend_interface(mode=param.input_feature, param=param)
        self.conv_layer1 = Conv_Oblong(channel, 32, kernel_list[0][0], kernel_list[0][1],pool_size=self.pool_list[0], deform=deform[0])
        self.conv_layer2 = Conv_Oblong(32,64, kernel_list[1][0],kernel_list[1][1],pool_size=self.pool_list[1], deform=deform[1])
        self.conv_layer3 = Conv_Oblong(64,128, kernel_list[2][0],kernel_list[2][1],pool_size=self.pool_list[2], deform=deform[2])
        self.conv_layer4 = Conv_Oblong(128,128, kernel_list[3][0],kernel_list[3][1],pool_size=self.pool_list[3], deform=deform[3])

        # # aggregation
        # if aggr == 'gap':
        #     self.aggr = nn.Sequential(nn.AdaptiveAvgPool2d(128), nn.Flatten())
        # elif aggr == 'gmp':
        #     self.aggr = nn.Sequential(nn.AdaptiveMaxPool2d(128), nn.Flatten())
        self.flat=nn.Flatten()
        if self.aggr == 'flatten':
            self.dense = nn.Linear(3840, dense_dim)
        else: 
            self.dense = nn.Linear(128, dense_dim)
        self.bn = nn.BatchNorm1d(dense_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.feature = nn.Linear(dense_dim, feature_dim)
        self.output = nn.Linear(feature_dim, class_num)

        self.train_acc = torchmetrics.Accuracy(num_classes=self.param.class_num,average='macro')
        self.f1= torchmetrics.F1()
        class_weights = [float(x) for x in weights.values()]
        self.class_weights = torch.from_numpy(np.array(class_weights)).float()

        
    def forward(self, x):
        out = self.spec(x)
        out = self.conv_layer1(out)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        # # aggregation
        if self.aggr == 'gap':
            out = out.mean(dim=(-2, -1))
        elif self.aggr == 'gmp':
            out,_ = out.max(dim=(-1,-2))
        else: 
            out = self.flat(out)
        #     self.aggr = nn.Sequential(nn.AdaptiveMaxPool2d(128), nn.Flatten())
        # out = out.view(-1, 128)
        # out = self.aggr(out)
        # print(out.shape)
        out = self.relu(self.bn(self.dense(out)))
        out = self.dropout(out)
        out = self.feature(out)
        feature = out
        out = self.output(out)
        return out, feature

    def training_step(self, batch, batch_idx):
        x, t, _ = batch 
        y,_ = self(x)
        y = y.to(DEVICE)
        t = t.to(DEVICE)
        class_weights=self.class_weights.to(DEVICE)
        loss = F.cross_entropy(y,t,weight=class_weights)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc(y,t), on_step=False, on_epoch=True)
        self.log('train_f1', self.f1(y,t), on_epoch=True) 
        return loss

    def configure_optimizers(self, lr=1e-3):
        optimizer = Adam(self.parameters(), lr)
        return optimizer


class Musicnn(BaseModel):
    '''
    Pons et al. 2017
    End-to-end learning for music audio tagging at scale.
    This is the updated implementation of the original paper. Referred to the Musicnn code.
    https://github.com/jordipons/musicnn
    '''
    def __init__(self,
                 param,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                dense_channel=200,
                n_class=10):
        super(Musicnn, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # Pons front-end
        m1 = Conv_V(1, 204, (int(0.7*n_mels), 7))
        m2 = Conv_V(1, 204, (int(0.4*n_mels), 7))
        m3 = Conv_H(1, 51, 129)
        m4 = Conv_H(1, 51, 65)
        m5 = Conv_H(1, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # Pons back-end
        # backend_channel= 512 if dataset=='msd' else 64
        backend_channel = 64
        self.layer1 = nn.Conv1d(561, backend_channel, 7, 1, 1)
        self.layer2 = nn.Conv1d(backend_channel, backend_channel, 7, 1, 1)
        self.layer3 = nn.Conv1d(backend_channel, backend_channel, 7, 1, 1)

        # Dense
        # dense_channel = 500 if dataset=='msd' else 200
        self.dense1 = nn.Linear((561+(backend_channel*3))*2, dense_channel)
        self.bn = nn.BatchNorm1d(dense_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_channel, n_class)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        # x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # Pons front-end
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)

        # Pons back-end
        length = out.size(2)
        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)

        out = self.relu(self.bn(self.dense1(out)))
        out = self.dropout(out)
        feature = out
        out = self.dense2(out)

        return out, feature
    
    def training_step(self, batch, batch_idx):
        x, t, _ = batch 
        y,_ = self(x)
        y = y.to(DEVICE)
        t = t.to(DEVICE)
        class_weights=self.class_weights.to(DEVICE)
        loss = F.cross_entropy(y,t,weight=class_weights)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc(y,t), on_step=False, on_epoch=True)
        self.log('train_f1', self.f1(y,t), on_epoch=True) 
        return loss

    def configure_optimizers(self, lr=1e-3):
        optimizer = Adam(self.parameters(), lr)
        return optimizer


class RawWaveNet(BaseModel):
    def __init__(self, weights:dict or list, param:Config, class_num=10):

        super(RawWaveNet, self).__init__()
        self.class_num = class_num
        self.is_fit = False
        self.batch_size = 64
        self.param = param
        self.aggr = param.aggr
        # input_shape = (freq, time, channel)

        
        # deform = deform if type(deform) == list else [deform, deform, deform, deform]

        self.conv_layer1 = Conv1DBlock(1,16,128)
        self.conv_layer2 = Conv1DBlock(16,8,64)
        self.conv_layer3 = Conv1DBlock(8,32,256)
        self.dense1 = nn.Linear(6944,32)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(32,class_num)
        # # aggregation
        # if aggr == 'gap':
        #     self.aggr = nn.Sequential(nn.AdaptiveAvgPool2d(128), nn.Flatten())
        # elif aggr == 'gmp':
        #     self.aggr = nn.Sequential(nn.AdaptiveMaxPool2d(128), nn.Flatten())
       
        self.train_acc = torchmetrics.Accuracy(num_classes=self.param.class_num,average='macro')
        self.f1= torchmetrics.F1()
        class_weights = [float(x) for x in weights.values()]
        self.class_weights = torch.from_numpy(np.array(class_weights)).float()

        
    def forward(self, x):
        out = self.conv_layer1(x)
        #print(out.shape)
        out = self.conv_layer2(out)
        #print(out.shape)
        out = self.conv_layer3(out)
        #print(out.shape)
        #     self.aggr = nn.Sequential(nn.AdaptiveMaxPool2d(128), nn.Flatten())
        # out = out.view(-1, 128)
        # out = self.aggr(out)
        out = nn.Flatten()(out)
        #print(out.shape)
        out = self.relu((self.dense1(out)))
        out = nn.Dropout(0.4)(out)
        #print(out.shape)
        out = self.dense2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, t, _ = batch 
        y = self(x)
        y = y.to(DEVICE)
        t = t.to(DEVICE)
        class_weights=self.class_weights.to(DEVICE)
        loss = F.cross_entropy(y,t,weight=class_weights)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', self.train_acc(y,t), on_step=False, on_epoch=True)
        self.log('train_f1', self.f1(y,t), on_epoch=True) 
        return loss

    def configure_optimizers(self, lr=1e-4):
        optimizer = RMSprop(self.parameters(), lr, weight_decay=1e-3)
        return optimizer
    
    def predict(self, x):
        """

        :param x: input tensor (torch.Tensor)
        :return: single output of model (numpy.array)
        """

        self.eval()
        out = self.forward(x)
        out = torch.argmax(out, dim=1)
        out = out.cpu().detach().numpy().copy()
        #out = np.squeeze(out)
        return out

    def predict_proba(self, x):
        """

        :param x: input tensor (torch.Tensor)
        :return: single output of model (numpy.array)
        """
        self.eval()
        out = self.forward(x)
        out = F.softmax(out, dim=1) # assuming logits has the shape [batch_size, nb_classes]
        out = out.cpu().detach().numpy().copy()
        out = np.squeeze(out)
        return out

