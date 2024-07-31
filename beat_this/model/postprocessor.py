from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class Postprocessor:
    """ Postprocessor for the beat and downbeat predictions of the model.
    The postprocessor takes the (framewise) model predictions (beat and downbeats) and the padding mask, 
    and returns the postprocessed beat and downbeat as list of times in seconds.
    The beats and downbeats can be 1D arrays (for only 1 piece) or 2D arrays, if a batch of pieces is considered.
    The output dimensionality is the same as the input dimensionality.
    Two types of postprocessing are implemented:
        - minimal: a simple postprocessing that takes the maximum of the framewise predictions,
        and removes adjacent peaks.
        - dbn: a postprocessing based on the Dynamic Bayesian Network proposed by Böck et al.
    Args:
        type (str): the type of postprocessing to apply. Either "minimal" or "dbn". Default is "minimal".
        fps (int): the frames per second of the model framewise predictions. Default is 50.
    """
    def __init__(self, type: str = "minimal", fps: int = 50):
        assert type in ["minimal", "dbn"]
        self.type = type
        self.fps = fps
        if type == "dbn":
            from madmom.features.downbeats import DBNDownBeatTrackingProcessor
            self.dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=self.fps, transition_lambda=100, )


    def __call__(self, beat : torch.Tensor, downbeat: torch.Tensor, padding_mask: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply postprocessing to the input beat and downbeat tensors. Works with batched and unbatched inputs.
        The output is a list of times in seconds, or a list of lists of times in seconds, if the input is batched.

        Args:
            beat (torch.Tensor): The input beat tensor.
            downbeat (torch.Tensor): The input downbeat tensor.
            padding_mask (torch.Tensor, optional): The padding mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The postprocessed beat tensor.
            torch.Tensor: The postprocessed downbeat tensor.
        """
        batched = False if beat.ndim == 1 else True
        if padding_mask is None:
            padding_mask = torch.ones_like(beat, dtype=torch.bool)

        # if beat and downbeat are 1D tensors, add a batch dimension
        if not batched:
            beat = beat.unsqueeze(0)
            downbeat = downbeat.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        if self.type == "minimal":
            postp_beat, postp_downbeat = self.postp_minimal(beat, downbeat, padding_mask)
        elif self.type == "dbn":
            postp_beat, postp_downbeat = self.postp_dbn(beat, downbeat, padding_mask)
        else:
            raise ValueError("Invalid postprocessing type")

        # remove the batch dimension if it was added
        if not batched:
            postp_beat = postp_beat[0]
            postp_downbeat = postp_downbeat[0]

        # update the model prediction dict
        return postp_beat, postp_downbeat

    def postp_minimal(self, beat, downbeat, padding_mask):
        # concatenate beat and downbeat in the same tensor of shape (B, T, 2)
        packed_pred = rearrange([beat, downbeat], "c b t -> b t c", b=beat.shape[0], t=beat.shape[1], c=2)
        # set padded elements to -1000 (= probability zero even in float64) so they don't influence the maxpool
        pred_logits = packed_pred.masked_fill(~padding_mask.unsqueeze(-1), -1000)
        # reshape to (2*B, T) to apply max pooling
        pred_logits = rearrange(pred_logits, "b t c -> (c b) t")
        # pick maxima within +/- 70ms
        pred_peaks = pred_logits.masked_fill(
            pred_logits != F.max_pool1d(pred_logits, 7, 1, 3), -1000)
        # keep maxima with over 0.5 probability (logit > 0)
        pred_peaks = (pred_peaks > 0)
        #  rearrange back to two tensors of shape (B, T)
        beat_peaks, downbeat_peaks = rearrange(pred_peaks, "(c b) t -> c b t", b=beat.shape[0], t=beat.shape[1], c=2)
        # run the piecewise operations
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(*executor.map(self._postp_minimal_item, beat_peaks, downbeat_peaks, padding_mask))
        return postp_beat, postp_downbeat

    def _postp_minimal_item(self, padded_beat_peaks, padded_downbeat_peaks, mask):
        """ Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        beat_peaks = padded_beat_peaks[mask]
        downbeat_peaks = padded_downbeat_peaks[mask]
        # pass from a boolean array to a list of times in frames.
        beat_frame = torch.nonzero(beat_peaks).cpu().numpy()[:,0]
        downbeat_frame = torch.nonzero(downbeat_peaks).cpu().numpy()[:,0]
        # remove adjacent peaks
        beat_frame = deduplicate_peaks(beat_frame, width=1)
        downbeat_frame = deduplicate_peaks(downbeat_frame, width=1)
        # convert from frame to seconds
        beat_time = beat_frame / self.fps
        downbeat_time = downbeat_frame / self.fps
        # move the downbeat to the nearest beat
        if len(beat_time) > 0: # skip if there are no beats, like in the first training steps
            for i, d_time in enumerate(downbeat_time):
                beat_idx = np.argmin(np.abs(beat_time - d_time))
                downbeat_time[i] = beat_time[beat_idx]
        # remove duplicate downbeat times (if some db were moved to the same position)
        downbeat_time = np.unique(downbeat_time)
        return beat_time, downbeat_time

    def postp_dbn(self, beat, downbeat, padding_mask):
        beat_prob = beat.double().sigmoid()
        downbeat_prob = downbeat.double().sigmoid()
        # limit lower and upper bound, since 0 and 1 create problems in the DBN
        epsilon = 1e-5
        beat_prob = beat_prob * (1-epsilon) + epsilon/2
        downbeat_prob = downbeat_prob * (1-epsilon) + epsilon/2
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(*executor.map(self._postp_dbn_item, beat_prob, downbeat_prob, padding_mask))
        return postp_beat, postp_downbeat

    def _postp_dbn_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        """ Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        beat_prob = padded_beat_prob[mask]
        downbeat_prob = padded_downbeat_prob[mask]
        # build an artificial multiclass prediction, as suggested by Böck et al.
        # again we limit the lower bound to avoid problems with the DBN
        epsilon = 1e-5
        combined_act = np.vstack((np.maximum(beat_prob.cpu().numpy() - downbeat_prob.cpu().numpy(), epsilon/2), downbeat_prob.cpu().numpy())).T
        # run the DBN
        dbn_out = self.dbn(combined_act)
        postp_beat = dbn_out[:, 0]
        postp_downbeat = dbn_out[dbn_out[:,1] == 1][:,0]
        return postp_beat, postp_downbeat



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
