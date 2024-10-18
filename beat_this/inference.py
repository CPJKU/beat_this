import inspect

import numpy as np
import soxr
import torch
import torch.nn.functional as F

from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect, load_audio
from beat_this.utils import replace_state_dict_key, save_beat_tsv

CHECKPOINT_URL = "https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp"


def load_checkpoint(checkpoint_path: str, device: str | torch.device = "cpu") -> dict:
    """
    Load a BeatThis checkpoint as a dictionary.

    Args:
        checkpoint_path (str, optional): The path to the checkpoint. Can be a local path, a URL, or a shortname.
        device (torch.device or str): The device to load the model on.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    try:
        # try interpreting as local file name
        return torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        try:
            if not (
                str(checkpoint_path).startswith("https://")
                or str(checkpoint_path).startswith("http://")
            ):
                # interpret it as a name of one of our checkpoints
                checkpoint_url = (
                    f"{CHECKPOINT_URL}/download?path=%2F&files={checkpoint_path}.ckpt"
                )
                file_name = f"beat_this-{checkpoint_path}.ckpt"
            else:
                # try interpreting as a URL
                checkpoint_url = checkpoint_path
                file_name = None
            return torch.hub.load_state_dict_from_url(
                checkpoint_url,
                file_name=file_name,
                map_location=device,
            )
        except Exception:
            raise ValueError(
                "Could not load the checkpoint given the provided name",
                checkpoint_path,
            )


def load_model(
    checkpoint_path: str | None = "final0", device: str | torch.device = "cpu"
) -> BeatThis:
    """
    Load a BeatThis model from a checkpoint.

    Args:
        checkpoint_path (str, optional): The path to the checkpoint. Can be a local path, a URL, or a shortname.
        device (torch.device or str): The device to load the model on.

    Returns:
        BeatThis: The loaded model.
    """
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, device)
        # Retrieve the model hyperparameters as it could be the small model
        hparams = checkpoint["hyper_parameters"]
        # Filter only those hyperparameters that apply to the model itself
        hparams = {
            k: v
            for k, v in hparams.items()
            if k in set(inspect.signature(BeatThis).parameters)
        }
        # Create the uninitialized model
        model = BeatThis(**hparams)
        # The PLBeatThis (LightningModule) state_dict contains the BeatThis
        # state_dict under the "model." prefix; remove the prefix to load it
        state_dict = replace_state_dict_key(checkpoint["state_dict"], "model.", "")
        model.load_state_dict(state_dict)
    else:
        model = BeatThis()
    return model.to(device).eval()


def zeropad(spect: torch.Tensor, left: int = 0, right: int = 0):
    """
    Pads a tensor spectrogram matrix of shape (time x bins) with `left` frames in the beginning and `right` frames in the end.
    """
    if left == 0 and right == 0:
        return spect
    else:
        return F.pad(spect, (0, 0, left, right), "constant", 0)


def split_piece(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int = 6,
    avoid_short_end: bool = True,
):
    """
    Split a tensor spectrogram matrix of shape (time x bins) into time chunks of `chunk_size` and return the chunks and starting positions.
    The `border_size` is the number of frames assumed to be discarded in the predictions on either side (since the model was not trained on the input edges due to the max-pool in the loss).
    To cater for this, the first and last chunk are padded by `border_size` on the beginning and end, respectively, and consecutive chunks overlap by `border_size`.
    If `avoid_short_end` is true, the last chunk start is shifted left to ends at the end of the piece, therefore the last chunk can potentially overlap with previous chunks more than border_size, otherwise it will be a shorter segment.
    If the piece is shorter than `chunk_size`, avoid_short_end is ignored and the piece is returned as a single shorter chunk.

    Args:
        spect (torch.Tensor): The input spectrogram tensor of shape (time x bins).
        chunk_size (int): The size of the chunks to produce.
        border_size (int, optional): The size of the border to overlap between chunks. Defaults to 6.
        avoid_short_end (bool, optional): If True, the last chunk is shifted left to end at the end of the piece. Defaults to True.
    """
    # generate the start and end indices
    starts = np.arange(
        -border_size, len(spect) - border_size, chunk_size - 2 * border_size
    )
    if avoid_short_end and len(spect) > chunk_size - 2 * border_size:
        # if we avoid short ends, move the last index to the end of the piece - (chunk_size - border_size)
        starts[-1] = len(spect) - (chunk_size - border_size)
    # generate the chunks
    chunks = [
        zeropad(
            spect[max(start, 0) : min(start + chunk_size, len(spect))],
            left=max(0, -start),
            right=max(0, min(border_size, start + chunk_size - len(spect))),
        )
        for start in starts
    ]
    return chunks, starts


def aggregate_prediction(
    pred_chunks: list,
    starts: list,
    full_size: int,
    chunk_size: int,
    border_size: int,
    overlap_mode: str,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    if border_size > 0:
        # cut the predictions to discard the border
        pred_chunks = [
            {
                "beat": pchunk["beat"][border_size:-border_size],
                "downbeat": pchunk["downbeat"][border_size:-border_size],
            }
            for pchunk in pred_chunks
        ]
    # aggregate the predictions for the whole piece
    piece_prediction_beat = torch.full((full_size,), -1000.0, device=device)
    piece_prediction_downbeat = torch.full((full_size,), -1000.0, device=device)
    if overlap_mode == "keep_first":
        # process in reverse order, so predictions of earlier excerpts overwrite later ones
        pred_chunks = reversed(list(pred_chunks))
        starts = reversed(list(starts))
    for start, pchunk in zip(starts, pred_chunks):
        piece_prediction_beat[
            start + border_size : start + chunk_size - border_size
        ] = pchunk["beat"]
        piece_prediction_downbeat[
            start + border_size : start + chunk_size - border_size
        ] = pchunk["downbeat"]
    return piece_prediction_beat, piece_prediction_downbeat


def split_predict_aggregate(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int,
    overlap_mode: str,
    model: torch.nn.Module,
) -> dict:
    """
    Function for pieces that are longer than the training length of the model.
    Split the input piece into chunks, run the model on them, and aggregate the predictions.
    The spect is supposed to be a torch tensor of shape (time x bins), i.e., unbatched, and the output is also unbatched.

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
    chunks, starts = split_piece(
        spect, chunk_size, border_size=border_size, avoid_short_end=True
    )
    # run the model
    pred_chunks = [model(chunk.unsqueeze(0)) for chunk in chunks]
    # remove the extra dimension in beat and downbeat prediction due to batch size 1
    pred_chunks = [
        {"beat": p["beat"][0], "downbeat": p["downbeat"][0]} for p in pred_chunks
    ]
    piece_prediction_beat, piece_prediction_downbeat = aggregate_prediction(
        pred_chunks,
        starts,
        spect.shape[0],
        chunk_size,
        border_size,
        overlap_mode,
        spect.device,
    )
    # save it to model_prediction
    return {"beat": piece_prediction_beat, "downbeat": piece_prediction_downbeat}


class Spect2Frames:
    """
    Class for extracting framewise beat and downbeat predictions (logits) from a spectrogram.
    """

    def __init__(self, checkpoint_path="final0", device="cpu", float16=False):
        super().__init__()
        self.device = torch.device(device)
        self.float16 = float16
        self.model = load_model(checkpoint_path, self.device)

    def spect2frames(self, spect):
        with torch.inference_mode():
            with torch.autocast(enabled=self.float16, device_type=self.device.type):
                model_prediction = split_predict_aggregate(
                    spect=spect,
                    chunk_size=1500,
                    overlap_mode="keep_first",
                    border_size=6,
                    model=self.model,
                )
        return model_prediction["beat"].float(), model_prediction["downbeat"].float()

    def __call__(self, spect):
        return self.spect2frames(spect)


class Audio2Frames(Spect2Frames):
    """
    Class for extracting framewise beat and downbeat predictions (logits) from an audio tensor.
    """

    def __init__(self, checkpoint_path="final0", device="cpu", float16=False):
        super().__init__(checkpoint_path, device, float16)
        self.spect = LogMelSpect(device=self.device)

    def signal2spect(self, signal, sr):
        if signal.ndim == 2:
            signal = signal.mean(1)
        elif signal.ndim != 1:
            raise ValueError(f"Expected 1D or 2D signal, got shape {signal.shape}")
        if sr != 22050:
            signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        return self.spect(signal)

    def __call__(self, signal, sr):
        spect = self.signal2spect(signal, sr)
        return self.spect2frames(spect)


class Audio2Beats(Audio2Frames):
    """
    Class for extracting beat and downbeat positions (in seconds) from an audio tensor.

    Args:
        checkpoint_path (str): Path to the model checkpoint file. It can be a local path, a URL, or a key from the CHECKPOINT_URL dictionary. Default is "final0", which will load the model trained on all data except GTZAN with seed 0.
        device (str): Device to use for inference. Default is "cpu".
        float16 (bool): Whether to use half precision floating point arithmetic. Default is False.
        dbn (bool): Whether to use the madmom DBN for post-processing. Default is False.
    """

    def __init__(
        self, checkpoint_path="final0", device="cpu", float16=False, dbn=False
    ):
        super().__init__(checkpoint_path, device, float16)
        self.frames2beats = Postprocessor(type="dbn" if dbn else "minimal")

    def __call__(self, signal, sr):
        beat_logits, downbeat_logits = super().__call__(signal, sr)
        return self.frames2beats(beat_logits, downbeat_logits)


class File2Beats(Audio2Beats):
    def __call__(self, audio_path):
        signal, sr = load_audio(audio_path)
        return super().__call__(signal, sr)


class File2File(File2Beats):
    def __call__(self, audio_path, output_path):
        downbeats, beats = super().__call__(audio_path)
        save_beat_tsv(downbeats, beats, output_path)
