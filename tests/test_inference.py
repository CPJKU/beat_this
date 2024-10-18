import soundfile as sf
import numpy as np
import torch
from pathlib import Path

from beat_this.inference import File2Beats, Audio2Frames


def test_File2Beat():
    f2b = File2Beats()
    audio_path = Path("tests/It Don't Mean A Thing - Kings of Swing.mp3")
    beat, downbeat = f2b(audio_path)
    assert isinstance(beat, np.ndarray)
    assert isinstance(downbeat, np.ndarray)


def test_Audio2Frames():
    a2f = Audio2Frames()
    audio_path = Path("tests/It Don't Mean A Thing - Kings of Swing.mp3")
    # load audio
    audio, sr = sf.read(audio_path)
    beat, downbeat = a2f(audio, sr)
    assert isinstance(beat, torch.Tensor)
    assert isinstance(downbeat, torch.Tensor)
