from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import madmom
import librosa

def load_audio(path):
    try:
        return sf.read(path, dtype='float64')
    except Exception:
        # some files are not readable by soundfile, try madmom
        return madmom.io.load_audio_file(str(path), dtype="float64")


def save_audio(path, waveform, samplerate, resample_from=None):
    if resample_from and resample_from != samplerate:
        waveform = librosa.resample(waveform,
                                    orig_sr=resample_from,
                                    target_sr=samplerate)
    try:
        waveform = np.asarray(waveform, dtype=np.float64)
        sf.write(path, waveform, samplerate=samplerate)
    except KeyboardInterrupt:
        path.unlink()  # avoid half-written files
        raise

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
    model_prediction = {}
    model_prediction["beat"] = piece_prediction_beat.unsqueeze(0)
    model_prediction["downbeat"] = piece_prediction_downbeat.unsqueeze(0)
    return model_prediction