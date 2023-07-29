import torch
import torch.nn as nn
import torchaudio

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class GenerateChannnelDim(nn.Module):
    def __init__(self, *args):
        super(GenerateChannnelDim, self).__init__()
        self.shape = args

    def forward(self, x):
        x = x.unsqueeze(1)
        return x


def aggregation_interface(mode) -> nn.Module:
    layer = nn.Sequential()
    if mode == 'flatten':
        layer.add_module('aggregate', nn.Flatten())
    elif mode == 'gap':
        layer.add_module('aggregate', nn.AdaptiveAvgPool2d((1, 1)))
    elif mode == 'gmp':
        layer.add_module('aggregate', nn.AdaptiveMaxPool2d(1, 1))
    return layer


def gen_input(param, n_mels=160, multi_res=False):
    # allow ['square', 'cascade-narrow', 'cascade-wide', 'bibranch-trans', 'bibranch-cis', 'musicnn', 'dense', 'multi-dense', 'inception']
    n_fft = param.n_fft
    win_length = param.win_length
    power = 1.0
    normalized = False
    sr = param.sr
    hop_length = param.hop_length
    f_min = 0.0
    f_max = param.f_max
    layer = nn.Sequential()
    output_dim = 0
    if param.multi_res == 3:
        output_dim += 3
        print("computing multi_resolution...")
        layer.add_module('spec', MultiResolutionMelSpec(n_mels=n_mels, hop_size=param.hop_length))
        # layer.add_module('bn', nn.BatchNorm2d(3))
    elif param.multi_res == 2:
        layer.add_module('spec', BiResolutionMelSpec(n_mels=n_mels, hop_size=param.hop_length))
        output_dim += 2
    elif param.delta:
        print("computing delta...")
        output_dim += 3
        if param.model_arch == "mfcc":
            layer.add_module('spec', MFCCDelta(sr=param.sr))
        else:
            layer.add_module('spec', MelSpecDelta(n_mels=n_mels))

        # layer.add_module('bn', nn.BatchNorm2d(3))

    else:
        print("not computing delta.")
        output_dim += 1
        if param.model_arch == "mfcc":
            layer.add_module('spec', torchaudio.transforms.MFCC(sample_rate=sr,
                                                                n_mfcc=40,
                                                                melkwargs={
                                                                "n_fft":n_fft,
                                                                  'hop_length':hop_length,
                                                                  'f_min':f_min,
                                                                  'f_max':f_max}))
        else:
            layer.add_module('spec', torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                          n_fft=n_fft,
                                                                          hop_length=hop_length,
                                                                          f_min=f_min,
                                                                          f_max=f_max,
                                                                          n_mels=n_mels))

        layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        # layer.add_module('channel', GenerateChannnelDim())
        # layer.add_module('bn', nn.BatchNorm2d(1))


    return layer, output_dim



class MelSpecDelta(nn.Module):
    def __init__(self, n_mels=128, hop_size=441, power=2.0, delta_win=5, fmax=8000,
                 normalized=False):
        self.hop_size = hop_size
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels
        self.delta_win=delta_win
        self.fmax=fmax

        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "m" and "l" specifying the output of the dimensionality reducing s convolutions
            c_out - Dictionary with keys "s", "m", "l", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        self.layer_list = []
        self.first_layer = nn.Sequential()
        self.first_layer.add_module('spec', torchaudio.transforms.MelSpectrogram(n_fft=2048,
                                                                                 hop_length=hop_size,
                                                                                 window_fn=torch.hann_window,
                                                                                 n_mels=self.n_mels,
                                                                                 power=self.power,
                                                                                 normalized=self.normalized,
                                                                                 f_max=self.fmax,
                                                                                 wkwargs=None), )
        self.first_layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        # self.first_layer.add_module('channel', GenerateChannnelDim())


    def forward(self, x):
        spec1 = self.first_layer(x)
        spec2 = torchaudio.transforms.ComputeDeltas(win_length=self.delta_win)(spec1)
        spec3 = torchaudio.transforms.ComputeDeltas(win_length=self.delta_win)(spec2)
        # print(spec1.shape, spec2.shape, spec3.shape)
        stacked = torch.cat([spec1, spec2, spec3], 1)
        return stacked


class MFCCDelta(nn.Module):
    def __init__(self, sr, n_mfcc=40, n_fft=2048, hop_size=441, power=2.0, delta_win=5,
                 normalized=False):

        self.hop_size = hop_size
        self.power = power
        self.normalized = normalized
        self.delta_win=delta_win

        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "m" and "l" specifying the output of the dimensionality reducing s convolutions
            c_out - Dictionary with keys "s", "m", "l", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        self.layer_list = []
        self.first_layer = nn.Sequential()
        self.first_layer.add_module('spec', torchaudio.transforms.MFCC(sample_rate=sr,n_mfcc=n_mfcc,
                                                                       melkwargs={"n_fft": n_fft,
                                                                       
                                                                                  "hop_length":hop_size,
                                                                                  }))
        self.first_layer.add_module('to_db', torchaudio.transforms.AmplitudeToDB())
        # self.first_layer.add_module('channel', GenerateChannnelDim())


    def forward(self, x):
        spec1 = self.first_layer(x)
        spec2 = torchaudio.transforms.ComputeDeltas(win_length=self.delta_win)(spec1)
        spec3 = torchaudio.transforms.ComputeDeltas(win_length=self.delta_win)(spec2)
        # print(spec1.shape, spec2.shape, spec3.shape)
        stacked = torch.cat([spec1, spec2, spec3], 1)
        return stacked

class Stack2Deltas(nn.Module):
    def __init__(self, delta_win=5,
                 normalized=False):
        super().__init__()
        self.normalized = normalized
        self.delta_win=delta_win

    def forward(self, x):
        delta = torchaudio.transforms.ComputeDeltas(win_length=self.delta_win)(x)
        delta2 = torchaudio.transforms.ComputeDeltas(win_length=self.delta_win)(x)
        # print(spec1.shape, spec2.shape, spec3.shape)
        stacked = torch.cat([x, delta, delta2], 1)
        return stacked


class BiResolutionMelSpec(nn.Module):
    def __init__(self, win_length_list=[1024, 512], n_mels=64, hop_size=441, power=1.0,
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
        super().__init__()
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
        # print(spec1.shape, spec2.shape, spec3.shape)
        stacked = torch.cat([spec1, spec2], 1)
        return stacked


class MultiResolutionMelSpec(nn.Module):
    def __init__(self, win_length_list=[2048, 1024, 512], n_mels=64, hop_size=512, power=1.0,
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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    


