dependencies = ['torch', 'torchaudio', 'numpy']

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

from beat_this.inference import predict_piece, load_model

class Audio2Beat(nn.Module):
    def __init__(self, model_checkpoint_path, device):
        super().__init__()
        self.device = torch.device(device)
        self.model = load_model(model_checkpoint_path, self.device)
        self.model.eval()

    def forward(self, audio_path):
        beat, downbeat = predict_piece(audio_path, self.model, False, self.device)
        return beat, downbeat
    

def beat_this_audio2beat(pretrained=True, device='cuda') -> Audio2Beat:
    """ Load pretrained audio to beat model. """
    if pretrained:
        model_checkpoint_path = "https://cloud.cp.jku.at/index.php/s/Dbtd47JqzDxWoks/download/final0.ckpt"
    else:
        model_checkpoint_path = None
    device = torch.device(device)
    return Audio2Beat(model_checkpoint_path, device)

if __name__ == "__main__":
    model = beat_this_audio2beat()
    audio_path = "data/preprocessed/mono_tracks/jaah/jaah_006-jelly_roll_mortons_red_hot_peppers-black_bottom_stomp/track_ps0.wav"
    beat, downbeat = model(audio_path)
    print(beat[:10], downbeat[:10])
    print("Done!")

