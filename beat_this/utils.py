from itertools import chain
from pathlib import Path

import numpy as np


def index_to_framewise(index, length):
    """Convert an index to a framewise sequence"""
    sequence = np.zeros(length, dtype=bool)
    sequence[index] = True
    return sequence


def filename_to_augmentation(filename):
    """Convert a filename to an augmentation factor."""
    parts = Path(filename).stem.split("_")
    augmentations = {}
    for part in parts[1:]:
        if part.startswith("ps"):
            augmentations["shift"] = int(part[2:])
        elif part.startswith("ts"):
            augmentations["stretch"] = int(part[2:])
    return augmentations


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
                "WARNING: There are more beats in the pickup measure than in the first measure. The beat count will start from 2 without trying to estimate the length of the pickup measure."
            )
            start_counter = 1
    else:
        print(
            "WARNING: There are less than two downbeats in the predictions. Something may be wrong. The beat count will start from 2 without trying to estimate the length of the pickup measure."
        )
        start_counter = 1

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


def replace_state_dict_key(state_dict: dict, old: str, new: str):
    """Replaces `old` in all keys of `state_dict` with `new`."""
    keys = list(state_dict.keys())  # take snapshot of the keys
    for key in keys:
        if old in key:
            state_dict[key.replace(old, new)] = state_dict.pop(key)
    return state_dict
