import torch
import torch.nn as nn
from input_modules import *
from collections import OrderedDict
import torchvision


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding=padding,bias=bias,groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=bias)
    def forward(self,x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out
    
    
class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 sconv=True,
                 bias=False,
                 mod=True):
        super().__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.sconv = sconv
        self.mod = mod

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        if sconv:
            self.depthwise_conv = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding=padding,bias=bias,groups=in_channels)
            self.pointwise_conv = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=bias)
        else:
            self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        #   dilation=self.dilation,
                                        bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        if self.mod:
            modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        else: 
            modulator = None
        # op = (n - (k * d - 1) + 2p / s)
        if self.sconv:
            # depthwise deformable convolution
            x = torchvision.ops.deform_conv2d(input=x,
                                            offset=offset,
                                            weight=self.depthwise_conv.weight,
                                            bias=self.depthwise_conv.bias,
                                            padding=self.padding,
                                            mask=modulator,
                                            stride=self.stride,
                                            dilation=self.dilation)
            x = self.pointwise_conv(x)
        else:
            x = torchvision.ops.deform_conv2d(input=x,
                                            offset=offset,
                                            weight=self.regular_conv.weight,
                                            bias=self.regular_conv.bias,
                                            padding=self.padding,
                                            mask=modulator,
                                            stride=self.stride,
                                            dilation=self.dilation)

        return x
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling=2, padding=1, sconv=False) -> None:
        super().__init__()
        if sconv:
            self.conv = SeparableConv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(pooling)

    def forward(self,x):
        x = self.pool(self.act(self.norm(self.conv(x))))
        return x


class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling=2, modulation=True, sconv=False) -> None:
        super().__init__()
        self.conv = DeformableConv2d(in_channels,out_channels,kernel_size=kernel_size, sconv=sconv)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(pooling)

    def forward(self,x):
        x = self.pool(self.act(self.norm(self.conv(x))))
        return x


class MultiResolutionMelSpec(nn.Module):
    def __init__(self, win_length_list=[2048, 1024, 512], n_mels=160, hop_size=512, power=1.0,
                 normalized=False):
        self.stacked_channels = len(win_length_list)
        self.n_fft_list = win_length_list
        self.win_length_list = win_length_list
        self.hop_size = hop_size
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels

        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "m" and "l" specifying the output of the dimensionality reducing s convolutions
            c_out - Dictionary with keys "s", "m", "l", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super(MultiResolutionMelSpec, self).__init__()
        self.layer_list = []
        self.n_fft = max(self.win_length_list)
        self.first_layer = nn.Sequential()
        self.first_layer.add_module('spec', torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,
                                                                                 win_length=self.win_length_list[0],
                                                                                 hop_length=hop_size, pad=0,
                                                                                 window_fn=torch.hann_window,
                                                                                 n_mels=self.n_mels,
                                                                                 power=self.power,
                                                                                 normalized=self.normalized,
                                                                                 wkwargs=None), )
        self.first_layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        # self.first_layer.add_module('channel', GenerateChannnelDim())

        self.second_layer = nn.Sequential()
        self.second_layer.add_module('spec', torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,
                                                                                  win_length=self.win_length_list[1],
                                                                                  hop_length=hop_size, pad=0,
                                                                                  window_fn=torch.hann_window,
                                                                                  n_mels=self.n_mels,
                                                                                  power=self.power,
                                                                                  normalized=self.normalized,
                                                                                  wkwargs=None), )
        self.second_layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        # self.second_layer.add_module('channel', GenerateChannnelDim())

        self.third_layer = nn.Sequential()
        self.third_layer.add_module('spec', torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,
                                                                                 win_length=self.win_length_list[2],
                                                                                 hop_length=hop_size, pad=0,
                                                                                 n_mels=self.n_mels,
                                                                                 window_fn=torch.hann_window,
                                                                                 power=self.power,
                                                                                 normalized=self.normalized,
                                                                                 wkwargs=None), )
        self.third_layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        # self.third_layer.add_module('channel', GenerateChannnelDim())

        # for win_length in self.win_length_list:
        #     if win_length == self.n_fft: continue
        #     self.layer_list.append(
        #         torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=win_length,
        #                                           hop_length=None, pad=0,
        #                                           window_fn=torch.hann_window,
        #                                           power=self.power, normalized=self.normalized, wkwargs=None),
        #     )

    def forward(self, x):
        spec1 = self.first_layer(x)
        spec2 = self.second_layer(x)
        spec3 = self.third_layer(x)
        # print(spec1.shape, spec2.shape, spec3.shape)
        stacked = torch.cat([spec1, spec2, spec3], 1)
        return stacked

