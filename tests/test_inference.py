import pytest
import numpy as np
import torch
from pathlib import Path

from beat_this.inference import Audio2Beats, Audio2Frames

def test_Audio2Beat():
    a2b = Audio2Beats()
    audio_path = Path("tests/Puttin On The Ritz - Kings of Swing.mp3")
    beat, downbeat = a2b(audio_path)
    assert isinstance(beat, np.ndarray)
    assert isinstance(downbeat, np.ndarray)
    
def test_Audio2Frame():
    a2f = Audio2Frames()
    audio_path = Path("tests/Puttin On The Ritz - Kings of Swing.mp3")
    beat, downbeat = a2f(audio_path)
    assert isinstance(beat, torch.Tensor)
    assert isinstance(downbeat, torch.Tensor)
    