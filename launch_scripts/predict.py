import os
import librosa
from pathlib import Path
import torchaudio
import numpy as np
import argparse
import torch

from beat_this.utils import save_beat_csv
from beat_this.inference import predict_piece, load_model


def main(audio_path, modelfile, dbn, outpath, gpu):
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            gpu
        )  # this is necessary to avoid a bug which causes pytorch to not see any GPU in some systems
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = load_model(modelfile, device)

    beat, downbeat = predict_piece(audio_path, model, dbn, device)
    print("Saving predictions...")
    save_beat_csv(beat, downbeat, outpath)
    print(f"Done, saved in {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes predictions for a given model and a given audio file."
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the audio file to process",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Local checkpoint files to use"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="test_output.beat",
        help="where to save the .beat file containing beat and downbeat predictions",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="which gpu to use, if any. -1 for cpu. Default is 0.",
    )
    parser.add_argument(
        "--dbn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="override the option to use madmom postprocessing dbn",
    )

    args = parser.parse_args()

    main(args.audio_path, args.model, args.dbn, args.output_path, args.gpu)
