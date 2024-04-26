import numpy as np
import torch
from jbt.utils import PAD_TOKEN
from pathlib import Path


def apply_maskings(maskings, audio, fps):
    """Apply the given masking operations to the audio of the given type."""
    for params in maskings:
        kind, prob, min_count, max_count, min_len, max_len, *params = params
        if np.random.rand() <= prob:
            count = np.random.randint(min_count, max_count + 1)
            min_len = int(min_len * fps)
            max_len = int(max_len * fps)
            for _ in range(count):
                length = np.random.randint(min_len, max_len + 1)
                start = np.random.randint(0, len(audio) - length)
                apply_mask_(audio[start:start + length], kind, params)
    return audio


def apply_mask_(excerpt, kind, params):
    """Apply a mask operation of the given kind in-place to the given tensor."""
    if kind == 'permute':
        num_parts = params[0] if len(params) == 1 else np.random.randint(params[0], params[1])
        choices = len(excerpt)
        num_parts = min(num_parts, choices + 1)
        positions = np.random.choice(choices, num_parts - 1, replace=False)
        positions.sort()
        if isinstance(excerpt, np.ndarray):
            parts = np.split(excerpt, positions)
        else:
            parts = ([excerpt[:positions[0]]] +
                     [excerpt[a:b] for a, b in zip(positions[:-1], positions[1:])] +
                     [excerpt[positions[-1]:]])
        parts = [parts[idx] for idx in np.random.permutation(num_parts)]
        if isinstance(excerpt, np.ndarray):
            excerpt[:] = np.concatenate(parts)
        else:
            excerpt[:] = torch.cat(parts)
    elif kind == 'zero':
        excerpt[:] = 0
    elif kind == 'constant':
        excerpt[:] = params[0]
    else:
        raise ValueError(f"Unsupported mask operation: {kind}")


def select_augmentation(item, augmentations):
    """Return a randomly chosen augmentation of the item.
    Handle precomputed augmentations if available.
    Mask augmentation is handled in the data loader."""

    def augment_pitch(item, pitch_params):
        """Apply pitch shifting to the item."""
        semitones = np.random.randint(pitch_params["min"], pitch_params["max"] + 1)
        item = shift_filename(item, semitones)
        item = shift_annotations(item, semitones)
        item["spect_length"] = item["spect_lengths"][0]
        return item

    def augment_time(item, time_params):
        """Apply time stretching to the item."""
        min = time_params["min"]
        max = time_params["max"]
        stride = time_params["stride"] if "stride" in time_params else 1
        percentage = np.random.choice(np.arange(min, max + 1, stride))
        item = stretch_filename(item, percentage)
        item = stretch_annotations(item, percentage)
        # select the spect length from the precomputed values
        item["spect_length"] = item["spect_lengths"][percentage]
        return item

    if 'pitch' in augmentations and 'time' in augmentations:
        # if both pitch and time are enabled, pick one of them
        if np.random.randint(2) == 0:
            # pitch
            item = augment_pitch(item, augmentations["pitch"])
        else:
            # tempo
            item = augment_time(item, augmentations["time"])
    elif 'pitch' in augmentations:
        item = augment_pitch(item, augmentations["pitch"])
    elif 'tempo' in augmentations:
        item = augment_time(item, augmentations["time"])
    return item

def stretch_annotations(item, percentage):
    """Apply time stretching to the item's annotations."""
    if not percentage:
        return item
    # percentage is the amount by which the *tempo* changes
    factor = 1.0 + percentage / 100
    item = dict(item)
    item["beat_time"] = item["beat_time"] / factor
    return item

def shift_annotations(item, semitones):
    """Apply pitch shifting to the item's annotations."""
    return item


def stretch_filename(item, percentage):
    """Derive filename of precomputed time stretched version."""
    filestem = "track_ps0"
    if percentage:
        filestem = filestem + f"_ts{percentage}"
    spect_path = Path(item["spect_folder"]) / f"{filestem}.npy"
    return {**item, "spect_path": spect_path}


def shift_filename(item, semitones):
    """Derive filename of precomputed pitch shifted version."""
    spect_path = Path(item["spect_folder"]) / f"track_ps{semitones}.npy"
    return {**item, "spect_path": spect_path}


def number_of_precomputed_augmentations(augmentations):
    """Return the number of augmentations that correspond to a precomputed file."""
    counter = 1
    for method, params in augmentations.values():
        if method in ('pitch'):
            counter += params["max"] - params["min"]
        elif method in ('tempo'):
            counter += ((params["max"] - params["min"]) // params["stride"])
    return counter


def precomputed_augmentation_filenames(augmentations):
    """Return the filenames of the precomputed augmentations."""
    filenames = ["track_ps0.npy"]
    for method, params in augmentations.items():
        if method == 'pitch':
            for semitones in range(params["min"], params["max"] + 1):
                filenames.append(f"track_ps{semitones}.npy")
        elif method == 'tempo':
            for percentage in range(params["min"], params["max"] + 1, params["stride"]):
                filenames.append(f"track_ps0_ts{percentage}.npy")
    return filenames

