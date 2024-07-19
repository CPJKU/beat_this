from pathlib import Path
from itertools import chain
import torch
import torch.nn.functional as F
import numpy as np


def index_to_framewise(index, length):
    """Convert an index to a framewise sequence"""
    sequence = np.zeros(length, dtype=bool)
    sequence[index] = True
    return sequence


def filename_to_augmentation(filename):
    """Convert a filename to an augmentation factor."""
    stem = Path(filename).stem
    if len(stem.split("_")) == 2:  # only pitch shift, e.g. track_ps-1
        return {"pitch": int(stem.split("_")[1].replace("ps", "")), "stretch": 0}
    elif (
        len(stem.split("_")) == 3
    ):  # pitch shift and time stretch, e.g. track_ps-1_ts12
        return {
            "pitch": int(stem.split("_")[1].replace("ps", "")),
            "stretch": int(stem.split("_")[2].replace("ts", "")),
        }
    else:
        raise ValueError(f"Unsupported filename: {filename}")


def load_spect(file_path, start=None, stop=None):
    """
    Load a spectrogram npy file.
    Optionally returns an excerpt with positions given in samples.
    """
    # load full file as memory map
    data = np.load(file_path, mmap_mode="c")
    # pick excerpt
    if start is not None or stop is not None:
        data = data[start:stop]
    return data


def save_beat_tsv(beats: np.ndarray, downbeats: np.ndarray, outpath: str) -> None:
    """
    Save beat information to a tab-separated file in the standard .beats format:
    each line has a time in seconds, a tab, and a beat number (1 = downbeat).
    The function requires that all downbeats are also listed as beats.

    Args:
        beats (numpy.ndarray): Array of beat positions in seconds (including downbeats).
        downbeats (numpy.ndarray): Array of downbeat positions in seconds.
        outpath (str): Path to the output TSV file.

    Returns:
        None
    """
    # check if all downbeats are beats
    if not np.all(np.isin(downbeats, beats)):
        raise ValueError("Not all downbeats are beats.")

    # handle pickup measure, by considering the beat count of the first full measure
    if len(downbeats) >= 2:
        # find the number of beats between the first two downbeats
        first_downbeat, second_downbeat = np.searchsorted(beats, downbeats[:2])
        beats_in_first_measure = second_downbeat - first_downbeat
        # find the number of beats before the first downbeat
        pickup_beats = first_downbeat
        # derive where to start counting
        if pickup_beats < beats_in_first_measure:
            start_counter = beats_in_first_measure - pickup_beats
        else:
            print(
                "WARNING: There are more pickup beats than beats in the first measure. This should not happen. The pickup measure will be considered as a normal measure."
            )
            start_counter = 0
    else:
        print(
            "WARNING: There are less than two downbeats in the predictions. Something may be wrong. No pickup measure will be considered."
        )
        start_counter = 0

    # write the beat file
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    counter = start_counter
    downbeats = chain(downbeats, [-1])
    next_downbeat = next(downbeats)
    try:
        with open(outpath, "w") as f:
            for beat in beats:
                if beat == next_downbeat:
                    counter = 1
                    next_downbeat = next(downbeats)
                else:
                    counter += 1
                f.write(f"{beat}\t{counter}\n")
    except KeyboardInterrupt:
        outpath.unlink()  # avoid half-written files
