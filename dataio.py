import os
import torchaudio
import numpy as np
from torch.utils import data
import pandas as pd
import torch
import torch.nn.functional as F
import random

from transformers import Wav2Vec2FeatureExtractor
random.seed(2023)

def inverse_frequency_dict(lst,alpha=1):
    frequency_dict = {}
    for num in lst:
        frequency_dict[num] = frequency_dict.get(num, 0) + 1

    inverse_frequency_dict = {num: 1 / freq ** alpha for num, freq in frequency_dict.items()}
    
    return inverse_frequency_dict


def crop_audio(
    waveform, 
    sample_rate, 
    crop_to_length_in_sec=None, 
    crop_to_length_in_sample_points=None, 
    crop_randomly=False, 
    pad=False,
):
    """Crop waveform to specified length in seconds or sample points.
    Supports random cropping and padding.

    Args:
        waveform (torch.Tensor): waveform of shape (1, n_sample)
        sample_rate (int): sample rate of waveform
        crop_to_length_in_sec (float, optional): crop to specified length in seconds. Defaults to None.
        crop_to_length_in_sample_points (int, optional): crop to specified length in sample points. Defaults to None.
        crop_randomly (bool, optional): crop randomly. Defaults to False.
        pad (bool, optional): pad to specified length if waveform is shorter than specified length. Defaults to False.

    Returns:
        torch.Tensor: cropped waveform
        int: start index of cropped waveform in original waveform
    """
    assert crop_to_length_in_sec is None or crop_to_length_in_sample_points is None, \
    "Only one of crop_to_length_in_sec and crop_to_length_in_sample_points can be specified"

    # convert crop length to sample points
    crop_duration_in_sample = None
    if crop_to_length_in_sec:
        crop_duration_in_sample = int(sample_rate * crop_to_length_in_sec)
    elif crop_to_length_in_sample_points:
        crop_duration_in_sample = crop_to_length_in_sample_points

    # crop
    start = 0
    if crop_duration_in_sample:
        if waveform.shape[-1] > crop_duration_in_sample:
            if crop_randomly:
                start = random.randint(0, waveform.shape[-1] - crop_duration_in_sample)
            waveform = waveform[..., start:start + crop_duration_in_sample]

        elif waveform.shape[-1] < crop_duration_in_sample:
            if pad:
                waveform = torch.nn.functional.pad(waveform, (0, crop_duration_in_sample - waveform.shape[-1]))
    
    return waveform, start


def chunk_audio(
    wav, 
    sample_rate, 
    sliding_window_size_in_sec, 
    sliding_window_overlap_in_percent, 
    last_chunk_align='overlap'
):
    '''
    given a wav torch tensor in [1, n_sample],
    output a list of wav chunks => [n_chunks, n_chunk_smaples], n_chunk_smaples must be the same
    last_chunk_align strategy: 
        'overlap' for filling with content in the previous chunk;
        'pad' to fill with 0 as postfix
    '''
    assert sliding_window_size_in_sec > 0, "sliding_window_size_in_sec must be positive"

    # print(f'wav shape: {wav.shape}') # torch.Size([1, 485936]) 
    overlap_in_sec = sliding_window_size_in_sec * sliding_window_overlap_in_percent / 100
    wavs = []
    for i in range(0, wav.shape[-1], int(sample_rate * (sliding_window_size_in_sec - overlap_in_sec))):
        wavs.append(wav[:, i : i + int(sample_rate * sliding_window_size_in_sec)])

    # assert last_chunk_align in ['overlap', 'pad', 'discard']
    # deal with the last chunk if it is shorter than the window size
    last_diff = sample_rate * sliding_window_size_in_sec - wavs[-1].shape[-1]
    if last_diff > 0:
        if last_chunk_align == 'overlap':
            z = torch.zeros_like(wavs[0]) # [1, sr * window]
            z[:, :last_diff] = wavs[-2][:,-last_diff:]
            z[:, last_diff:] = wavs[-1][:,:]
            wavs[-1] = z
        elif last_chunk_align == 'discard':
            wavs = wavs[:-1]
        elif last_chunk_align == 'pad':
            z = torch.zeros_like(wavs[0]) # [1, sr * window]
            z[:, :wavs[-1].shape[-1]] = wavs[-1][:,:] # assign
            wavs[-1] = z
        else:
            raise NotImplementedError
    
    assert torch.all(wavs[-1] <  2**31)

    # print(f'chunk wavs[0] shape: {wavs[0].shape}') torch.Size([1, 80000])
    # wavs = torch.stack(wavs, dim=0) # 
    # print(f'wavs tensor shape: {wavs.shape}') torch.Size([7, 1, 80000])
    return wavs


def find_audios(
    parent_dir, 
    exts=['.wav', '.mp3', '.flac', '.webm', '.mp4']
):
    audio_files = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if os.path.splitext(file)[1] in exts:
                audio_files.append(os.path.join(root, file))
    return audio_files


def load_audio(
    file_path, 
    target_sr, 
    is_mono=True, 
    is_normalize=False,
    crop_to_length_in_sec=None, 
    crop_to_length_in_sample_points=None, 
    crop_randomly=False, 
    pad=False,
    return_start=False,
    device=torch.device('cpu')
):
    """Load audio file and convert to target sample rate.
    Supports cropping and padding.

    Args:
        file_path (str): path to audio file
        target_sr (int): target sample rate, if not equal to sample rate of audio file, resample to target_sr
        is_mono (bool, optional): convert to mono. Defaults to True.
        is_normalize (bool, optional): normalize to [-1, 1]. Defaults to False.
        crop_to_length_in_sec (float, optional): crop to specified length in seconds. Defaults to None.
        crop_to_length_in_sample_points (int, optional): crop to specified length in sample points. Defaults to None. Note that the crop length in sample points is calculated before resampling.
        crop_randomly (bool, optional): crop randomly. Defaults to False.
        pad (bool, optional): pad to specified length if waveform is shorter than specified length. Defaults to False.
        device (torch.device, optional): device to use for resampling. Defaults to torch.device('cpu').
    
    Returns:
        torch.Tensor: waveform of shape (1, n_sample)
    """
    # TODO: deal with target_depth
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        if is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if is_normalize:
        waveform = waveform / waveform.abs().max()
    
    waveform, start = crop_audio(
        waveform, 
        sample_rate, 
        crop_to_length_in_sec=crop_to_length_in_sec, 
        crop_to_length_in_sample_points=crop_to_length_in_sample_points, 
        crop_randomly=crop_randomly, 
        pad=pad,
    )
    
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = waveform.to(device)
        resampler = resampler.to(device)
        waveform = resampler(waveform)
    
    if return_start:
        return waveform, start
    return waveform


class AudioDataset(data.Dataset):
    def __init__(self, audio_dir, metadata_dir, split, sample_duration=3, return_audio_path=False, sample_rate=44100, ssl=False): # Singer trim?
        # self.cfg = cfg
        self.metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}_t.txt'), 
                                    names = ['audio_path'])
        self.audio_dir = audio_dir
        self.class2id = {'belt': 0, 'breathy': 1, 'inhaled': 2, 'lip_trill': 3, 'spoken': 4, 'straight': 5, 'trill': 6, 'trillo': 7, 'vibrato': 8, 'vocal_fry': 9}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        self.sample_rate = sample_rate
        self.ssl = ssl
        self.sample_duration = sample_duration * self.sample_rate if sample_duration else None
        # extractor
        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sample_rate,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=True,
        )

    def process_wav(self, waveform):
        # return the same shape
        return self.processor(
            waveform,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
            padding=True).input_values[0]


    def __getitem__(self, index):
        audio_path = self.metadata.iloc[index][0]
        audio = load_audio(os.path.join(self.audio_dir, audio_path), 
            target_sr = self.sample_rate,
            is_mono = True,
            is_normalize =  True,
        )

        # sample a duration of audio from random start
        if self.sample_duration is not None:  
            # if audio is shorter than sample_duration, pad it with zeros
            if audio.shape[1] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[1]), 'constant', 0)
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[:, random_start:random_start+self.sample_duration]
        
        # preprocess and reshaping
        if self.ssl:
            audio_features = self.process_wav(audio)
        else:
            audio_features = audio
        # # convert
        # audio_features = self.processor(audio, return_tensors="pt", sampling_rate=self.cfg.target_sr, padding=True).input_values[0]
        
        label = self.class2id[audio_path.split('/')[0]]
        if self.return_audio_path:
            return audio_features, label, audio_path
        return audio_features, label

    def __len__(self):
        return len(self.metadata)
    
    def get_class_weights(self,alpha):
        df = self.metadata.copy()
        df['class'] = df['audio_path'].apply(lambda x: self.class2id[x.split('_')[-3]])
        list_class = df['class'].values.tolist()
        result_dict = inverse_frequency_dict(list_class,alpha)
        return result_dict
