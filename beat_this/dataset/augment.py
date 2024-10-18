import numpy as np
import torch


def augment_pitchtempo(item, augmentations):
    """
    Apply a randomly chosen pitch or tempo augmentation to the item.

    Parameters:
    item: dict
        A dictionary representing the item to be augmented. It should contain the following keys:
        - 'spect_path': The path to the the unaugmented spectrogram file.
        If pitch or tempo augmentation is applied, the 'spect_path' key will be updated.

    augmentations: dict
        A dictionary containing the augmentations to be applied. It can contain either or both of the following keys:
        - 'pitch': A dictionary with 'min' and 'max' keys specifying the range of pitch shifting in semitones.
        - 'tempo': A dictionary with 'min' and 'max' keys specifying the range of time stretching factors.

    Returns:
    item: dict
        The item after applying the augmentation. If a pitch or tempo augmentation was applied, the 'spect_path' key
        and the annotations will be updated.
    """
    # Handle pitch and tempo augmentations
    if "pitch" in augmentations and "tempo" in augmentations:
        # if both pitch and tempo are enabled, pick one of them
        if np.random.randint(2) == 0:
            # pitch
            item = augment_pitch(item, augmentations["pitch"])
        else:
            # tempo
            item = augment_tempo(item, augmentations["tempo"])
    elif "pitch" in augmentations:
        item = augment_pitch(item, augmentations["pitch"])
    elif "tempo" in augmentations:
        item = augment_tempo(item, augmentations["tempo"])

    return item


def augment_pitch(item, pitch_params):
    """Apply pitch shifting to the item."""
    semitones = np.random.randint(pitch_params["min"], pitch_params["max"] + 1)
    item = shift_filename(item, semitones)
    item = shift_annotations(item, semitones)
    return item


def augment_tempo(item, tempo_params):
    """Apply time stretching to the item."""
    percentage = np.random.choice(
        np.arange(tempo_params["min"], tempo_params["max"] + 1, tempo_params["stride"])
    )
    item = stretch_filename(item, percentage)
    item = stretch_annotations(item, percentage)
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
    spect_path = item["spect_path"]
    if percentage:
        stem = spect_path.stem + f"_ts{percentage}"
        spect_path = spect_path.with_stem(stem)
    return {**item, "spect_path": spect_path}


def shift_filename(item, semitones):
    """Derive filename of precomputed pitch shifted version."""
    spect_path = item["spect_path"]
    if semitones:
        stem = spect_path.stem + f"_ps{semitones}"
        spect_path = spect_path.with_stem(stem)
    return {**item, "spect_path": spect_path}


def number_of_precomputed_augmentations(augmentations):
    """Return the number of augmentations that correspond to a precomputed file."""
    counter = 1
    for method, params in augmentations.values():
        if method in ("pitch"):
            counter += params["max"] - params["min"]
        elif method in ("tempo"):
            counter += (params["max"] - params["min"]) // params["stride"]
    return counter


def precomputed_augmentation_filenames(augmentations, ext="npy"):
    """Return the filenames of the precomputed augmentations.

    Parameters:
    augmentations: dict
        A dictionary containing the augmentations to be applied. It can contain either or both of the following keys:
        - 'pitch': A dictionary with 'min' and 'max' keys specifying the range (including boundaries) of pitch shifting in semitones.
        - 'tempo': A dictionary with 'min' and 'max' keys specifying the range (including boundaries) of time stretching factors; and a 'stride' key specifying the step size.
    """
    filenames = [f"track.{ext}"]
    for method, params in augmentations.items():
        if method == "pitch":
            for semitones in range(params["min"], params["max"] + 1):
                if semitones == 0:
                    continue
                filenames.append(f"track_ps{semitones}.{ext}")
        elif method == "tempo":
            for percentage in range(params["min"], params["max"] + 1, params["stride"]):
                if percentage == 0:
                    continue
                filenames.append(f"track_ts{percentage}.{ext}")
    return filenames


def augment_mask_(spect, augmentations: dict, fps: int):
    """
    Apply the given masking operations to the spectrogram. The spectrogram is modified in place.

    Parameters:
    spect: ndarray
        The input spectrogram to which the mask will be applied. It is a 2D array where the first dimension
        represents time frames and the second dimension represents frequency bins.

    augmentations: dict
        A dictionary containing all the augmentations. If there is no "mask" key, this function returns the
        unmodified spectrogram. If "mask" key is present, the value is another dictionary which must include
        the following keys:
        - 'kind': The type of mask to apply. Choices: 'permute' and 'zero'.
        - 'min_count' and 'max_count': The minimum and maximum number of times the mask should be applied.
        - 'min_len' and 'max_len': The minimum and maximum length of the mask, expressed in seconds.
        - 'min_parts' and 'max_parts': The minimum and maximum number of parts in which each masked section is segmented.
            These are then randomly reordered. If 'kind'='permute' this parameter is not used.

    fps: int
        The frames per second of the audio. This is used to convert 'min_len' and 'max_len' from seconds to frames.

    Returns:
    spect: ndarray
        The spectrogram after applying the mask.

    """
    if "mask" in augmentations:
        mask_params = augmentations["mask"]
        count = np.random.randint(
            mask_params["min_count"], mask_params["max_count"] + 1
        )
        # convert min_len and max_len in frames
        min_len = int(mask_params["min_len"] * fps)
        max_len = int(mask_params["max_len"] * fps)
        # apply the masking a number of time specified by count
        for _ in range(count):
            length = np.random.randint(min_len, max_len + 1)
            start = np.random.randint(0, len(spect) - length)
            apply_mask_excerpt(
                spect[start : start + length],
                mask_params["kind"],
                mask_params["min_parts"],
                mask_params["max_parts"],
            )
    return spect


def apply_mask_excerpt(excerpt, kind, min_parts, max_parts):
    """Apply a mask operation of the given kind in-place to the given tensor."""
    if kind == "permute":
        num_parts = np.random.randint(min_parts, max_parts + 1)
        choices = len(excerpt)
        num_parts = min(num_parts, choices + 1)
        positions = np.random.choice(choices, num_parts - 1, replace=False)
        positions.sort()
        if isinstance(excerpt, np.ndarray):
            parts = np.split(excerpt, positions)
        else:
            parts = (
                [excerpt[: positions[0]]]
                + [excerpt[a:b] for a, b in zip(positions[:-1], positions[1:])]
                + [excerpt[positions[-1] :]]
            )
        parts = [parts[idx] for idx in np.random.permutation(num_parts)]
        if isinstance(excerpt, np.ndarray):
            excerpt[:] = np.concatenate(parts)
        else:
            excerpt[:] = torch.cat(parts)
    elif kind == "zero":
        excerpt[:] = 0
    else:
        raise ValueError(f"Unsupported mask operation: {kind}")
