from beat_this.preprocessing.preprocess_audio import load_audio
from beat_this.model.beat_tracker import BeatThis
from beat_this.model.pl_module import split_predict_aggregate
from beat_this.model.postprocessor import Postprocessor
import os
import librosa
from pathlib import Path

import torchaudio
import numpy as np
import argparse
import torch

# this is necessary to avoid a bug which cause pytorch to not see any GPU in some systems
os.environ['CUDA_VISIBLE_DEVICES']='0'

def main(audio_path, modelfile, dbn, outpath,  gpu):
    device = torch.device("cuda:" + str(gpu) if gpu >= 0 else "cpu")

    # Load the audio and convert to spectrogram
    waveform, audio_sr = load_audio(audio_path)
    if audio_sr != 22050: # resample to 22050 if necessary
        waveform = librosa.resample(waveform, orig_sr=audio_sr, target_sr=22050)
    if waveform.ndim != 1: # if stereo, convert to mono
        waveform = np.mean(waveform, axis=1)
    waveform = torch.tensor(waveform, dtype=torch.float32, device=device)
    mel_args = dict(n_fft=1024, hop_length=441, f_min=30, f_max=11000,
                    n_mels=128, mel_scale='slaney', normalized='frame_length', power=1)
    spect_class = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, **mel_args).to(device)
    spect = spect_class(waveform.unsqueeze(0)).squeeze(0).T
    # scale the values with log(1 + 1000 * x)
    spect = torch.log1p(1000*spect)

    # Load the model
    model = BeatThis()
    checkpoint = torch.load(modelfile, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # Predict the beats and downbeats
    model.eval()
    with torch.no_grad():
        model_prediction = split_predict_aggregate(spect=spect, chunk_size=1500, overlap_mode="keep_first", border_size=6, model=model)
    # postprocess the predictions
    postprocessor = Postprocessor(type="dbn" if dbn else "minimal", fps=50)
    model_prediction = postprocessor(model_prediction)
    save_beat_csv(model_prediction["postp_beat"][0], model_prediction["postp_downbeat"][0], outpath)

    


def save_beat_csv(beats,downbeats, outpath):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)

    # handle pickup measure, by considering the beat number of the first full measure
    # find the number of beats between the first two downbeats
    if len(downbeats) > 2:
        beat_in_first_measure = beats[(beats < downbeats[1]) & (beats >= downbeats[0])].shape[0]
        # find the number of pickup beats
        pickup_beats = beats[beats < downbeats[0]].shape[0]
        if pickup_beats < beat_in_first_measure:
            start_counter = beat_in_first_measure - pickup_beats
        else: 
            print("WARNING: There are more pickup beats than beats in the first measure. This should not happen. The pickup measure will be considered as a normal measure.")
            pickup_beats = 0
            beat_in_first_measure = 0
            counter = 0
    else:
        print("WARNING: There are less than two downbeats in the predictions. Something may be wrong. No pickup measure will be considered.")
        start_counter = 0

    
    counter = start_counter
    # write the beat file
    with open(outpath, "w") as f:
        for beat in beats:
            if beat in downbeats:
                counter = 1
            else:
                counter += 1
            f.write(str(beat) + "\t" + str(counter) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes predictions for a given model and a given audio file.")
    parser.add_argument("--audio-path", type=str,
                        required=True,
                        help="Path to the audio file to process")
    parser.add_argument("--model", type=str,
                        required=True,
                        help="Local checkpoint files to use")
    parser.add_argument("--output_path", type=str,
                        default="test_output.beat",
                        help="where to save the .beat file containing beat and downbeat predictions")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which gpu to use, if any. -1 for cpu. Default is 0.")
    parser.add_argument(
        "--dbn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="override the option to use madmom postprocessing dbn",
    )

    args = parser.parse_args()

    main(args.audio_path, args.model, args.dbn, args.output_path,  args.gpu)