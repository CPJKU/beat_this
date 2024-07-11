from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import madmom
import librosa


def index_to_framewise(index, length):
    """Convert an index to a framewise sequence"""
    sequence = np.zeros(length, dtype=bool)
    sequence[index] = True
    return sequence

def filename_to_augmentation(filename):
    """Convert a filename to an augmentation factor."""
    stem = Path(filename).stem
    if len(stem.split('_')) == 2: # only pitch shift, e.g. track_ps-1
        return {"pitch": int(stem.split('_')[1].replace('ps', '')), "stretch": 0}
    elif len(stem.split('_')) == 3: # pitch shift and time stretch, e.g. track_ps-1_ts12
        return {"pitch": int(stem.split('_')[1].replace('ps', '')), "stretch": int(stem.split('_')[2].replace('ts', ''))}
    else:
        raise ValueError(f"Unsupported filename: {filename}")
    
def load_spect(file_path, start=None, stop=None):
    """
    Load a spectrogram npy file. 
    Optionally returns an excerpt with positions given in samples.
    """
    # load full file as memory map
    data = np.load(file_path, mmap_mode='c')
    # pick excerpt
    if start is not None or stop is not None:
        data = data[start:stop]
    return data

def split_piece(spect : torch.Tensor, chunk_size : int, border_size : int=6, avoid_short_end : bool=True):
    """
    Split a tensor spectrogram matrix of shape (time x bins) into time chunks of `chunk_size` and return the chunks and starting positions.
    Consecutive chunks overlap by `border_size`, which is assumed to be discarded in the predictions, since the model
     is not really trained on this part due to the max-pool in the loss.
    If `avoid_short_end` is true, the last chunk start is shifted left to ends at the end of the piece, therefore the last chunk can potentially overlap with previous chunks more than border_size, otherwise it will be a shorter segment.
    If the piece is shorter than `chunk_size`, avoid_short_end is ignored and the piece is returned as a single shorter chunk.

    Args:
        spect (torch.Tensor): The input spectrogram tensor of shape (time x bins).
        chunk_size (int): The size of the chunks to produce.
        border_size (int, optional): The size of the border to overlap between chunks. Defaults to 6.
        avoid_short_end (bool, optional): If True, the last chunk is shifted left to end at the end of the piece. Defaults to True.
    """
    # generate the start and end indices 
    starts = np.arange(-border_size, len(spect)- 2*border_size, chunk_size - 2 * border_size)
    if avoid_short_end and len(spect) > chunk_size - border_size:
        # if we avoid short ends, move the last index to the end of the piece - (chunk_size - 2 *border_size)
        starts[-1] = len(spect) - (chunk_size - border_size)
    # generate the chunks
    chunks = [spect[max(start,0):min(start+chunk_size,len(spect))] for start in starts]
    # pad the first and last chunk in the time dimension to account for the border
    chunks[0] = F.pad(chunks[0], (0, 0, border_size, 0), "constant", 0)
    chunks[-1] = F.pad(chunks[-1], (0, 0, 0, border_size), "constant", 0)
    return chunks, starts

def aggregate_prediction(pred_chunks, starts, full_size, chunk_size, border_size, overlap_mode, device):
    """
    Aggregates the predictions for the whole piece based on the given prediction chunks.

    Args:
        pred_chunks (list): List of prediction chunks, where each chunk is a dictionary containing 'beat' and 'downbeat' predictions.
        starts (list): List of start positions for each prediction chunk.
        full_size (int): Size of the full piece.
        chunk_size (int): Size of each prediction chunk.
        border_size (int): Size of the border to be discarded from each prediction chunk.
        overlap_mode (str): Mode for handling overlapping predictions. Can be 'keep_first' or 'keep_last'.
        device (torch.device): Device to be used for the predictions.

    Returns:
        tuple: A tuple containing the aggregated beat predictions and downbeat predictions as torch tensors for the whole piece.
    """
    # cut the predictions to discard the border
    pred_chunks = [{'beat': pchunk['beat'][border_size:-border_size], 'downbeat': pchunk['downbeat'][border_size:-border_size]} for pchunk in pred_chunks]
    # aggregate the predictions for the whole piece
    piece_prediction_beat = torch.full((full_size,), -1000., device=device)
    piece_prediction_downbeat = torch.full((full_size,), -1000., device=device)
    if overlap_mode == 'keep_first':
        # process in reverse order, so predictions of earlier excerpts overwrite later ones
        pred_chunks = reversed(list(pred_chunks))
        starts = reversed(list(starts))
    for start, pchunk in zip(starts, pred_chunks):
        piece_prediction_beat[start + border_size:start + chunk_size - border_size] = pchunk["beat"]
        piece_prediction_downbeat[start + border_size:start + chunk_size - border_size] = pchunk["downbeat"]
    return piece_prediction_beat, piece_prediction_downbeat


def split_predict_aggregate(spect: torch.Tensor, chunk_size: int, border_size: int, overlap_mode: str, model: torch.nn.Module):
    """
    Function for pieces that are longer than the training length of the model.
    Split the input piece into chunks, run the model on them, and aggregate the predictions.

    Args:
        spect (torch.Tensor): the input piece
        chunk_size (int): the length of the chunks
        border_size (int): the size of the border that is discarded from the predictions
        overlap_mode (str): how to handle overlaps between chunks
        model (torch.nn.Module): the model to run

    Returns:
        dict: the model framewise predictions for the hole piece as a dictionary containing 'beat' and 'downbeat' predictions.
    """
    # split the piece into chunks
    chunks, starts = split_piece(spect, chunk_size, border_size= border_size, avoid_short_end=True)
    # run the model
    pred_chunks = [model(chunk.unsqueeze(0)) for chunk in chunks]
    # remove the extra dimension in beat and downbeat prediction due to batch size 1
    pred_chunks = [{"beat": p["beat"][0], "downbeat": p["downbeat"][0]} for p in pred_chunks]
    piece_prediction_beat, piece_prediction_downbeat = aggregate_prediction(pred_chunks, starts, spect.shape[0], chunk_size, border_size, overlap_mode, spect.device)
    # save it to model_prediction
    return {"beat": piece_prediction_beat, "downbeat": piece_prediction_downbeat}

def save_beat_csv(beats, downbeats, outpath):
    """
    Save beat information to a csv file in the standard .beat format.
    The class assume that all downbeats are also beats.

    Args:
        beats (numpy.ndarray): Array of beat positions in seconds.
        downbeats (numpy.ndarray): Array of downbeat positions in seconds.
        outpath (str): Path to the output CSV file.

    Returns:
        None
    """
    # check if all downbeats are beats
    if not np.all(np.isin(downbeats, beats)):
        raise ValueError("Not all downbeats are beats.")

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
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for beat in beats:
            if beat in downbeats:
                counter = 1
            else:
                counter += 1
            f.write(str(beat) + "\t" + str(counter) + "\n")
