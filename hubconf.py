dependencies = ['torch', 'torchaudio', 'numpy']

import torch

from beat_this.inference import Audio2Beat


    

def beat_this_audio2beat(pretrained=True, device='cuda') -> Audio2Beat:
    """ Load pretrained audio to beat model. """
    if pretrained:
        model_checkpoint_path = "final0"
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

