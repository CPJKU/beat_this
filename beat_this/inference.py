import torch
import torchaudio
import librosa
import numpy as np

from beat_this.preprocessing import load_audio, LogMelSpect
from beat_this.utils import split_predict_aggregate 
from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import Postprocessor

def lightning_to_torch(checkpoint : dict):
    """
    Convert a PyTorch Lightning checkpoint to a PyTorch checkpoint.

    Args:
        checkpoint (dict): The PyTorch Lightning checkpoint.

    Returns:
        dict: The PyTorch checkpoint.

    """
    # modify the checkpoint to remove the prefix "model.", so we can load a lightning module checkpoint in pure pytorch
    # allow loading from the PLBeatThis lightning checkpoint
    for key in list(checkpoint['state_dict'].keys()):  # use list to take a snapshot of the keys
        if "model." in key:
            checkpoint['state_dict'][key.replace("model.", "")] = checkpoint['state_dict'].pop(key)
    return checkpoint

def load_model(checkpoint_path : str, device : torch.device):
    model = BeatThis()
    if checkpoint_path is not None:
        if str(checkpoint_path).startswith("https://"):
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # modify the checkpoint to remove the prefix "model.", so we can load a lightning module checkpoint in pure pytorch
        checkpoint = lightning_to_torch(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    return model.to(device)


def predict_piece(audio_path, model, dbn, device ):
    print("Loading audio...")
    waveform, audio_sr = load_audio(audio_path)
    if waveform.ndim != 1: # if stereo, convert to mono
        waveform = np.mean(waveform, axis=1)
    if audio_sr != 22050: # resample to 22050 if necessary
        waveform = librosa.resample(waveform, orig_sr=audio_sr, target_sr=22050)
    waveform = torch.tensor(waveform, dtype=torch.float32, device=device)
    spect = LogMelSpect(device=device)(waveform)

    # Predict the beats and downbeats
    print("Predicting beats and downbeats...")
    model.eval()
    with torch.no_grad():
        model_prediction = split_predict_aggregate(spect=spect, chunk_size=1500, overlap_mode="keep_first", border_size=6, model=model)
    # postprocess the predictions
    postprocessor = Postprocessor(type="dbn" if dbn else "minimal", fps=50)
    model_prediction = postprocessor(model_prediction)
    return model_prediction["postp_beat"][0], model_prediction["postp_downbeat"][0]