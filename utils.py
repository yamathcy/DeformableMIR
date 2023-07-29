import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import hydra
from dataclasses import dataclass
import dataclasses
import pandas as pd
import librosa
import mlflow
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, ListConfig, DictConfig

@dataclass
class Config:
    data_dir: str = '/home/ubuntu/dataset/cosian'
    experiment_name: str = 'singing technique identification'
    length: float = 3.0
    chunking_hop: float = 1.0
    n_mels:int = 160
    f_max:int = 8000
    n_fft:int = 2048  # 0.5sec
    hop_length:int = 441  # 0.25sec
    win_length:list = dataclasses.field(default_factory=list)
    sr:int = 44100
    normalize: bool = True
    without_valid: bool = False
    input_feature:str = 'mel'  # allow['mel', 'stft']
    model_arch: str = 'imoto'
    epoch:int = 100
    class_weight:bool =  False
    dropout: float = 0.0
    batch_size: int = 16
    optimizer: str = 'adam'
    lr: float = 1e-3
    random_seed: int =15213
    audio_path:str = '/home/ubuntu/dataset/cosian/audio'
    meta_path:str = '/home/ubuntu/dataset/cosian/cosian_st_annotation_v8.csv'
    model_path: str = './models'
    data_path: str = '/home/ubuntu/dataset/cosian'
    normalization: str = "instance"
    detection_threshold:float = 0.5
    pitch_shift:bool = False
    early_stop:bool = True
    loss:str = 'bce'
    focal_loss_gamma:float = 1.33
    focal_loss_alpha:float = 0.87
    technique:str = 'common' #allow['common', 'semi-common', 'all']
    test_fold:int = 7
    do_cv:bool = True
    aux_pitch:str = 'none'
    melody_path:str= '/home/ubuntu/dataset/cosian/pitch_10ms'
    time_resolution:float = 0.1
    delta:bool =  False
    elimination:float = 0.02
    layer_num:int = 6 
    activation:str = "lrelu"
    conv_dropout:float = 0.1
    multi_res:int = 1
    kernel_size:list = dataclasses.field(default_factory=list) 
    stride:list = dataclasses.field(default_factory=list) 
    filters:list = dataclasses.field(default_factory=list) 
    pooling:list = dataclasses.field(default_factory=list) 
    is_single_label:bool = False
    seblock:bool = False
    sereduction:int = 16
    autoth:bool = False
    blur:bool = False
    patience:float = 10
    freeze_all:bool=False
    apex:bool=False
    url:str="facebook/wav2vec2-base"
    retrain:int=0

ALL_MODE = ['square', 'cascade-narrow', 'cascade-wide', 'bibranch-trans', 'bibranch-cis', 'musicnn', 'dense', 'multi-dense', 'inception']
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
SR = 44100
HOP_LENGTH = 512
WINDOW_LENGTH = 2048
N_MFCC = 20
NUM_RFC= 50
F_MIN = 0
F_MAX = 8000
N_MELS = 128
eps = 1e-7
VOCALSET_LENGTH = 3
FEATURE_DIM = 22
CLASS_NUM=10
target_class = {'belt': 0, 'breathy': 1, 'inhaled': 2, 'liptrill': 3, 'straight': 4, 'trill': 5,
                             'trillo': 6, 'spoken': 7, 'vibrato': 8, 'vocalfry': 9}
target_class_inv = {v:k for k, v in target_class.items()}


# singer_label_map = {label:idx for label, idx in enumerate(all_singer)}

singer_class = {'m1': 0, 'm2': 1, 'm3': 2, 'm4': 3, 'm5': 4, 'm6': 5, 'm7': 6, 'm8': 7, 'm9': 8, 'm10': 9, 'm11': 10, 'f1': 11,'f2': 12, 'f3': 13, 'f4': 14, 'f5': 15, 'f6': 16, 'f7': 17, 'f8': 18, 'f9': 19}
singer_class_inv = {v:k for k, v in singer_class.items()}


# Visualization
def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img


import numpy as np
from omegaconf import ListConfig, DictConfig
import mlflow

train_singer = ["aiko","ayaka","chara","flumpool","fukuyama",
                "gackt","hamasaki","hirahara","hirai","hitoto","ikimono",
                "jam","koda","koyanagi","larc","makihara","matsutouya","moriyama",
                "matsuura","nakjima",
                "oda","onitsuka","sazan","tamaki","utada","yamazaki","yoasobi","yonezu"]
valid_singer = ["fuse","lisa","tmr","yamaguchi"]
test_singer = ["aimyon","creephyp","higedan","mrchildren","ootsuka","ozaki"]


def get_label_class(techique_type):
    '''

    :param techique_type: (list)
    :return:
    '''

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


def load_desc_file(_desc_file, target_labels, sr:int, is_dict=False):
    '''
    desc_file
    '''
    _desc_dict = dict()
    _desc_list_dcase = []
    for num, line in enumerate(open(_desc_file)):
        # ignore header
        if num == 0: continue

        # load description  ['start', 'end', 'label', 'path of audio']
        words = line.strip().split(',')

        # review: フルパスの必要はないかも
        file_name = words[3].split('/')[-1]
        # print(file_name)
        # name = words[0]  # フルパスじゃない場合
        if file_name not in _desc_dict:
            _desc_dict[file_name] = list()
        # desc_dict : {'file_name': [start_time(float), end_time(float), label(int)]
        # }
        print('filename: {}, Add label start{:.3f} end{:.3f} label{}'.format(file_name, float(words[0]), float(words[1]), words[2]))
        try:

            _desc_dict[file_name].append([float(words[0]), float(words[1]), target_labels[words[2]]])
        except KeyError:
            pass

        if is_dict == True:
            data = {
                'file': file_name,
                'event_label': words[2],
                'event_onset': float(words[0]),
                'event_offset': float(words[1])
                }
            print(data)
            _desc_list_dcase.append(data)

    return _desc_dict if is_dict is False else _desc_list_dcase

def load_desc_file_pd(_desc_file: pd.DataFrame, target_labels, sr:int, is_dict=False):
    '''
    desc_file
    '''
    _desc_dict = dict()
    _desc_list_dcase = []
    # print(_desc_file.head())
    # print(_desc_file['file'].unique())
    for i in _desc_file['file'].unique():
        _desc_dict[i] = []
    for num, line in enumerate(_desc_file.iterrows()):
        line = line[1]

        if line['label'] in target_labels:
            # ignore header
            if num == 0: continue

            # load description  ['start', 'end', 'label', 'path of audio']
            # words = line.strip().split(',')

            # review: フルパスの必要はないかも
            # file_name = words[3].split('/')[-1]
            # print(line)
            # print(type(line))
            # print(line[1]['file'])

            file_name = str(line['file']).replace(' ','')

            # print(file_name)
            # name = words[0]  # フルパスじゃない場合
            if file_name not in _desc_dict:
                _desc_dict[file_name] = list()
            # desc_dict : {'file_name': [start_time(float), end_time(float), label(int)]
            # }

            # print('filename: {}, Add label start{:.3f} end{:.3f} label{}'.format(file_name, float(words[0]), float(words[1]), words[2]))
            try:
                # _desc_dict[file_name].append([float(words[0]), float(words[1]), target_labels[words[2]]])
                _desc_dict[file_name].append([float(line['start']), float(line['end']), target_labels[line['label']]])
                # print([float(line['start']), float(line['end']), target_labels[line['label']]])
            except:
                pass

            if is_dict == True:
                data = {
                    'file': file_name,
                    'event_label': line['label'],
                    'event_onset': float(line['start']),
                    'event_offset': float(line['end'])
                    }
                # print(data)
                _desc_list_dcase.append(data)

    # print(_desc_dict)

    return _desc_dict if is_dict is False else _desc_list_dcase


def get_melody_contour(melody, n_mels=None, generate_newaxis=True, delta=False, blur=False):
    # quantize
    freq_bin=librosa.core.mel_frequencies(n_mels=n_mels)
    m_contr=np.zeros((freq_bin.shape[0], len(melody))).astype('float32')
    for idx in range(len(melody)):
        p=np.where(freq_bin<=melody[idx])[0][-1]
        m_contr[p, idx]=1.0
        
    for idx in range(len(melody)):
        if blur:
            p=np.where(freq_bin<=melody[idx])[0][-1]
            freq_diff = (melody[idx]-freq_bin[p]) / (freq_bin[p+1]-freq_bin[p])
            m_contr[p, idx]=(1-freq_diff)
            m_contr[p+1, idx]=freq_diff
        else:
            p=np.where(freq_bin<=melody[idx])[0][-1]
            m_contr[p, idx]=1.0

    assert all(m_contr.sum(axis=0))
    # if delta:
    #     m_contr_delta = librosa.feature.delta(m_contr,width=5,axis=-1)
    #     m_contr_delta_delta = librosa.feature.delta(m_contr_delta, width=5, axis=-1)
    #     m_contr = np.concatenate([m_contr,m_contr_delta, m_contr_delta_delta])
    if generate_newaxis:
        m_contr = m_contr[np.newaxis,...]
    return m_contr


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


def chunk_target_length(mat, target_length):
    # print("mat{}, target_length{}".format(type(mat),type(target_length)))
    # print(mat.shape)
    if type(mat) ==  tuple:
        print(mat)
        print(len(mat))
        return 
    # FIXME: Implicitly expected the last axis of the input should be time
    if type(target_length) != int:
        target_length=int(target_length)
    start = 0
    mat_length = mat.shape[-1]
    chunk_list = []

    # chunk
    while (start + target_length) < mat_length:
        chunk = mat[...,start:start+target_length]
        chunk_list.append(chunk)
        start += target_length

    # pad
    padded = pad_along_axis(mat[...,start:], target_length=target_length, axis=-1)  
    chunk_list.append(padded)

    return chunk_list

def frames_to_time(t, sr=None, hop_length=None, mode="wav2vec"):
    if mode in ["wavlm", "mert", "maitai"]:
        frame_length = 1/75
        return t * frame_length
    else:
        return librosa.frames_to_time(t, sr, hop_length)