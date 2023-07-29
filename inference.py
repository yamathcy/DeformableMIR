import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl
import os, sys
import glob 
import numpy as np
import matplotlib.pylab as plt
import argparse


def main(args):
    """_summary_
    

    Args:
        args (_type_): _description_
    """
    
    # check the model load
    if not os.path.exists(args.model):
        raise FileNotFoundError 
    
    model = pl.LightningModule.load_from_checkpoint(args.model)
    model.eval()
    wav, sr = torchaudio.load(filepath=args.audiofile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The path to the saved model")
    parser.add_argument("--offset", type=bool, help="Outputs the plot of spectrogram of learned offset or not", default=False)
    parser.add_argument("--audiofile", type=str, help="The path to the audio file that wants to be processed")
    args = parser.parse_args()
    main(args)