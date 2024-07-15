import torch
import librosa
import numpy as np
import torch.nn as nn
from typing import Tuple

from beat_this.preprocessing import load_audio, LogMelSpect
from beat_this.utils import split_predict_aggregate
from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import Postprocessor

CHECKPOINT_URL = {
    "final0": "https://cloud.cp.jku.at/index.php/s/Dbtd47JqzDxWoks/download/final0.ckpt",
    "final1": "https://cloud.cp.jku.at/index.php/s/DCm9YLkTBAEc4y3/download/final1.ckpt",
    "final2": "https://cloud.cp.jku.at/index.php/s/E8A3McdxpwSGGwJ/download/final2.ckpt",
    "fold0": "https://cloud.cp.jku.at/index.php/s/oZrBck4nCZLkkQw/download/fold0.ckpt",
    "fold1": "https://cloud.cp.jku.at/index.php/s/rDaS9YtiYE6Qyrn/download/fold1.ckpt",
    "fold2": "https://cloud.cp.jku.at/index.php/s/Z4PHTqD58x3C5dt/download/fold2.ckpt",
    "fold3": "https://cloud.cp.jku.at/index.php/s/Cmc5wT6KEoHE4mP/download/fold3.ckpt",
    "fold4": "https://cloud.cp.jku.at/index.php/s/tXz5KsmGrJNkPog/download/fold4.ckpt",
    "fold5": "https://cloud.cp.jku.at/index.php/s/Mb95SoY2GtMEA3H/download/fold5.ckpt",
    "fold6": "https://cloud.cp.jku.at/index.php/s/ADxyETzQQ5iGEj9/download/fold6.ckpt",
    "fold7": "https://cloud.cp.jku.at/index.php/s/jPXq6HqJeeezcqH/download/fold7.ckpt",
}


def lightning_to_torch(checkpoint: dict):
    """
    Convert a PyTorch Lightning checkpoint to a PyTorch checkpoint.

    Args:
        checkpoint (dict): The PyTorch Lightning checkpoint.

    Returns:
        dict: The PyTorch checkpoint.

    """
    # modify the checkpoint to remove the prefix "model.", so we can load a lightning module checkpoint in pure pytorch
    # allow loading from the PLBeatThis lightning checkpoint
    for key in list(
        checkpoint["state_dict"].keys()
    ):  # use list to take a snapshot of the keys
        if "model." in key:
            checkpoint["state_dict"][key.replace("model.", "")] = checkpoint[
                "state_dict"
            ].pop(key)
    return checkpoint


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load a BeatThis model from a checkpoint.

    Args:
        checkpoint_path (str): The path to the checkpoint. Can be a local path, a URL, or a key in MODELS_URL.
        device (torch.device): The device to load the model on.

    Returns:
        BeatThis: The loaded model.

    """
    model = BeatThis()
    if checkpoint_path is not None:
        try:
            if checkpoint_path in CHECKPOINT_URL:
                checkpoint_path = CHECKPOINT_URL[checkpoint_path]
            if str(checkpoint_path).startswith("https://"):
                checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise ValueError(
                "Could not load the checkpoint given the provided name", checkpoint_path
            )
        # modify the checkpoint to remove the prefix "model.", so we can load a lightning module checkpoint in pure pytorch
        checkpoint = lightning_to_torch(checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def audio2beat(
    audio_path,
    model_checkpoint: str = "final0",
    dbn: bool = False,
    device: torch.device = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract beat and downbeat positions (in seconds) from an audio file.

    Args:
        audio_path (str): The path to the audio file.
        model (str): The model checkpoint to use. Can be a local path, a URL, or a key in MODELS_URL. Default is "final0".
        dbn (bool): Whether to use the DBN postprocessor. Default is False.
        device (torch.device): The device to use for inference. Default is "cpu".

    Returns:
        tuple: A tuple containing two np.ndarray with beat and downbeat positions in seconds.

    """
    a2b = Audio2Beat(model_checkpoint, device, dbn)
    return a2b(audio_path)


class Spect2Frames:
    """
    Class for extracting framewise beat and downbeat predictions (logits) from a spectrogram.
    """

    def __init__(self, model_checkpoint_path="final0", device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.model = load_model(model_checkpoint_path, self.device)
        self.model.eval()

    def __call__(self, spect):
        with torch.no_grad():
            model_prediction = split_predict_aggregate(
                spect=spect,
                chunk_size=1500,
                overlap_mode="keep_first",
                border_size=6,
                model=self.model,
            )
        return model_prediction["beat"], model_prediction["downbeat"]


class Audio2Frames(Spect2Frames):
    """
    Class for extracting framewise beat and downbeat predictions (logits) from an audio file.
    """

    def __init__(self, model_checkpoint_path="final0", device="cpu"):
        super().__init__(model_checkpoint_path, device)
        self.device = torch.device(device)

    def __call__(self, audio_path):
        waveform, audio_sr = load_audio(audio_path)
        if waveform.ndim != 1:
            waveform = np.mean(waveform, axis=1)
        if audio_sr != 22050:
            waveform = librosa.resample(waveform, orig_sr=audio_sr, target_sr=22050)
        waveform = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        spect = LogMelSpect(device=self.device)(waveform)
        return super().__call__(spect)


class Audio2Beat(Audio2Frames):
    """
    Class for extracting beat and downbeat positions (in seconds) from an audio files.

    Args:
        model_checkpoint_path (str): Path to the model checkpoint file. It can be a local path, a URL, or a key from the CHECKPOINT_URL dictionary. Default is "final0", which will load the model trained on all data except GTZAN with seed 0.
        device (str): Device to use for inference. Default is "cpu".
        dbn (bool): Whether to use the madmom DBN for post-processing. Default is False.
    """

    def __init__(self, model_checkpoint_path="final0", device="cpu", dbn=False):
        super().__init__(model_checkpoint_path, device)
        self.device = torch.device(device)
        self.postprocessor = Postprocessor(type="dbn" if dbn else "minimal")

    def __call__(self, audio_path: str):
        beat, downbeat = super().__call__(audio_path)
        return self.postprocessor(beat, downbeat)
