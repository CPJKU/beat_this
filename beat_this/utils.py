import io
import base64
import struct
from pathlib import Path
import zipfile
import wave
import scipy.io.wavfile
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from rotary_embedding_torch import RotaryEmbedding
import copy
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.swa_utils import AveragedModel

PAD_TOKEN = 1024


def index_to_framewise(index, length):
    """Convert an index to a framewise sequence"""
    sequence = np.zeros(length, dtype=bool)
    sequence[index] = True
    return sequence

def get_spect_len(spect_path):
    return np.load(spect_path, mmap_mode='r').shape[-1]

def filename_to_augmentation(filename):
    """Convert a filename to an augmentation factor."""
    stem = Path(filename).stem
    if len(stem.split('_')) == 2: # only pitch shift, e.g. track_ps-1
        return {"pitch": int(stem.split('_')[1].replace('ps', '')), "stretch": 0}
    elif len(stem.split('_')) == 3: # pitch shift and time stretch, e.g. track_ps-1_ts12
        return {"pitch": int(stem.split('_')[1].replace('ps', '')), "stretch": int(stem.split('_')[2].replace('ts', ''))}
    else:
        raise ValueError(f"Unsupported filename: {filename}")
    
def load_spect(file_path, start=None, stop=None):
    """
    Load a spectrogram npy file. 
    Optionally returns an excerpt with positions given in samples.
    """
    # load full file as memory map
    data = np.load(file_path, mmap_mode='r+')
    # pick excerpt
    if start is not None or stop is not None:
        data = data[:,start:stop]
    return data.T