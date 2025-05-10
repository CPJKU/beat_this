from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from beat_this.postprocessing_interface import Postprocessor as PostprocessorInterface


class MinimalPostprocessor(PostprocessorInterface):
    """Minimal postprocessor for beat and downbeat predictions.
    
    This postprocessor applies a simple algorithm that:
    1. Finds peaks in the beat and downbeat logits
    2. Removes adjacent peaks
    3. Converts frame indices to timestamps
    4. Aligns downbeats to the nearest beats
    
    Args:
        fps (int): Frames per second of the model predictions. Default is 50.
    """

    def __init__(self, fps: int = 50):
        self.fps = fps
    
    def __call__(
        self,
        beat_logits: torch.Tensor,
        downbeat_logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Apply minimal postprocessing to the input beat and downbeat logits.
        Works with batched and unbatched inputs.
        
        Args:
            beat_logits (torch.Tensor): Beat prediction logits
            downbeat_logits (torch.Tensor): Downbeat prediction logits
            padding_mask (torch.Tensor, optional): Padding mask tensor. Defaults to None.
            
        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: Processed beat and downbeat times in seconds
        """
        was_batched = beat_logits.ndim > 1
        if padding_mask is None:
            padding_mask = torch.ones_like(beat_logits, dtype=torch.bool)

        # if inputs are 1D tensors, add a batch dimension
        if not was_batched:
            beat_logits = beat_logits.unsqueeze(0)
            downbeat_logits = downbeat_logits.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        # run the main processing
        postp_beat, postp_downbeat = self._process_batch(
            beat_logits, downbeat_logits, padding_mask
        )

        # if input wasn't batched, we still need to return lists
        if not was_batched:
            # Convert to lists with a single element
            postp_beat = [postp_beat[0]]
            postp_downbeat = [postp_downbeat[0]]

        return postp_beat, postp_downbeat
    
    def _process_batch(self, beat_logits, downbeat_logits, padding_mask):
        # concatenate beat and downbeat in the same tensor of shape (B, T, 2)
        packed_pred = rearrange(
            [beat_logits, downbeat_logits], "c b t -> b t c", b=beat_logits.shape[0], t=beat_logits.shape[1], c=2
        )
        # set padded elements to -1000 (= probability zero even in float64) so they don't influence the maxpool
        pred_logits = packed_pred.masked_fill(~padding_mask.unsqueeze(-1), -1000)
        # reshape to (2*B, T) to apply max pooling
        pred_logits = rearrange(pred_logits, "b t c -> (c b) t")
        # pick maxima within +/- 70ms
        pred_peaks = pred_logits.masked_fill(
            pred_logits != F.max_pool1d(pred_logits, 7, 1, 3), -1000
        )
        # keep maxima with over 0.5 probability (logit > 0)
        pred_peaks = pred_peaks > 0
        #  rearrange back to two tensors of shape (B, T)
        beat_peaks, downbeat_peaks = rearrange(
            pred_peaks, "(c b) t -> c b t", b=beat_logits.shape[0], t=beat_logits.shape[1], c=2
        )
        # run the piecewise operations
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._process_single_item, beat_peaks, downbeat_peaks, padding_mask
                )
            )
        return list(postp_beat), list(postp_downbeat)

    def _process_single_item(self, padded_beat_peaks, padded_downbeat_peaks, mask):
        """Process a single item in the batch."""
        # unpad the predictions by truncating the padding positions
        beat_peaks = padded_beat_peaks[mask]
        downbeat_peaks = padded_downbeat_peaks[mask]
        # pass from a boolean array to a list of times in frames.
        beat_frame = torch.nonzero(beat_peaks).cpu().numpy()[:, 0]
        downbeat_frame = torch.nonzero(downbeat_peaks).cpu().numpy()[:, 0]
        # remove adjacent peaks
        beat_frame = deduplicate_peaks(beat_frame, width=1)
        downbeat_frame = deduplicate_peaks(downbeat_frame, width=1)
        # convert from frame to seconds
        beat_time = beat_frame / self.fps
        downbeat_time = downbeat_frame / self.fps
        # move the downbeat to the nearest beat
        if (
            len(beat_time) > 0
        ):  # skip if there are no beats, like in the first training steps
            for i, d_time in enumerate(downbeat_time):
                beat_idx = np.argmin(np.abs(beat_time - d_time))
                downbeat_time[i] = beat_time[beat_idx]
        # remove duplicate downbeat times (if some db were moved to the same position)
        downbeat_time = np.unique(downbeat_time)
        return beat_time, downbeat_time


def deduplicate_peaks(peaks, width=1) -> np.ndarray:
    """
    Replaces groups of adjacent peak frame indices that are each not more
    than `width` frames apart by the average of the frame indices.
    """
    result = []
    peaks = map(int, peaks)  # ensure we get ordinary Python int objects
    try:
        p = next(peaks)
    except StopIteration:
        return np.array(result)
    c = 1
    for p2 in peaks:
        if p2 - p <= width:
            c += 1
            p += (p2 - p) / c  # update mean
        else:
            result.append(p)
            p = p2
            c = 1
    result.append(p)
    return np.array(result)
