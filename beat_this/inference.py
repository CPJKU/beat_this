import torch
import soxr
import numpy as np

from beat_this.preprocessing import load_audio, LogMelSpect
from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import Postprocessor
from beat_this.utils import save_beat_tsv

CHECKPOINT_URL = "https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp"


def lightning_to_torch(checkpoint: dict) -> dict:
    """
    Convert a PyTorch Lightning checkpoint to a PyTorch checkpoint.

    Args:
        checkpoint (dict): The PyTorch Lightning checkpoint.

    Returns:
        dict: The PyTorch checkpoint.
    """
    # modify the checkpoint to remove the prefix "model.", so we can load the
    # PLBeatThis lightning module checkpoint in pure pytorch
    for key in list(
        checkpoint["state_dict"].keys()
    ):  # use list to take a snapshot of the keys
        if "model." in key:
            checkpoint["state_dict"][key.replace("model.", "")] = checkpoint[
                "state_dict"
            ].pop(key)
    return checkpoint


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
    model = BeatThis()
    if checkpoint_path is not None:
        try:
            # try interpreting as local file name
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except FileNotFoundError:
            try:
                if not (
                    str(checkpoint_path).startswith("https://")
                    or str(checkpoint_path).startswith("http://")
                ):
                    # interpret it as a name of one of our checkpoints
                    checkpoint_path = f"{CHECKPOINT_URL}/download?path=%2F&files={checkpoint_path}.ckpt"
                checkpoint = torch.hub.load_state_dict_from_url(
                    checkpoint_path, map_location=device
                )
            except Exception as e:
                raise ValueError(
                    "Could not load the checkpoint given the provided name",
                    checkpoint_path,
                )
        # modify the checkpoint to remove the prefix "model.", so we can load a lightning module checkpoint in pure pytorch
        checkpoint = lightning_to_torch(checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def split_piece(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int = 6,
    avoid_short_end: bool = True,
):
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
    starts = np.arange(
        -border_size, len(spect) - 2 * border_size, chunk_size - 2 * border_size
    )
    if avoid_short_end and len(spect) > chunk_size - border_size:
        # if we avoid short ends, move the last index to the end of the piece - (chunk_size - 2 *border_size)
        starts[-1] = len(spect) - (chunk_size - border_size)
    # generate the chunks
    chunks = [
        spect[max(start, 0) : min(start + chunk_size, len(spect))] for start in starts
    ]
    # pad the first and last chunk in the time dimension to account for the border
    chunks[0] = F.pad(chunks[0], (0, 0, border_size, 0), "constant", 0)
    chunks[-1] = F.pad(chunks[-1], (0, 0, 0, border_size), "constant", 0)
    return chunks, starts


def aggregate_prediction(
    pred_chunks: list, starts: list, full_size: int, chunk_size: int, border_size: int, overlap_mode: str, device: str | torch.device
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

    def __init__(self, checkpoint_path="final0", device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.model = load_model(checkpoint_path, self.device)

    def spect2frames(self, spect):
        with torch.inference_mode():
            model_prediction = split_predict_aggregate(
                spect=spect,
                chunk_size=1500,
                overlap_mode="keep_first",
                border_size=6,
                model=self.model,
            )
        return model_prediction["beat"], model_prediction["downbeat"]

    def __call__(self, spect):
        return spect2frames(spect)


class Audio2Frames(Spect2Frames):
    """
    Class for extracting framewise beat and downbeat predictions (logits) from an audio tensor.
    """

    def __init__(self, checkpoint_path="final0", device="cpu"):
        super().__init__(checkpoint_path, device)
        self.spect = LogMelSpect(device=self.device)

    def signal2spect(self, signal, sr):
        if signal.ndim != 1:
            signal = signal.mean(range(1, signal.ndim))
        if ar != 22050:
            signal = soxr.resample(signal, in_rate=audio_sr, out_rate=22050)
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        return self.spect(signal)

    def __call__(self, signal, sr):
        spect = super()(signal, sr)
        return self.spect2frames(spect)


class Audio2Beats(Audio2Frames):
    """
    Class for extracting beat and downbeat positions (in seconds) from an audio tensor.

    Args:
        checkpoint_path (str): Path to the model checkpoint file. It can be a local path, a URL, or a key from the CHECKPOINT_URL dictionary. Default is "final0", which will load the model trained on all data except GTZAN with seed 0.
        device (str): Device to use for inference. Default is "cpu".
        dbn (bool): Whether to use the madmom DBN for post-processing. Default is False.
    """

    def __init__(self, checkpoint_path="final0", device="cpu", dbn=False):
        super().__init__(checkpoint_path, device)
        self.frames2beats = Postprocessor(type="dbn" if dbn else "minimal")

    def __call__(self, signal, sr):
        beat_logits, downbeat_logits = super()(signal, sr)
        return self.frames2beats(beat_logits, downbeat_logits)


class File2Beats(Audio2Beats):
    def __call__(self, audio_path):
        return super()(load_audio(audio_path))


class File2File(File2Beats):
    def __call__(self, audio_path, output_path):
        downbeats, beats = super()(audio_path)
        save_beat_tsv(downbeats, beats, output_path)
