from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from beat_this.postprocessing_interface import Postprocessor as PostprocessorInterface

class DbnPostprocessor(PostprocessorInterface):
    """DBN-based postprocessor for beat and downbeat predictions.
    
    This postprocessor uses the Dynamic Bayesian Network from madmom to process
    beat and downbeat logits into beat and downbeat times.
    
    Args:
        fps (int): Frames per second of the model predictions. Default is 50.
        beats_per_bar (tuple): Possible number of beats per bar. Default is (3, 4).
        min_bpm (float): Minimum tempo in BPM. Default is 55.0.
        max_bpm (float): Maximum tempo in BPM. Default is 215.0.
        transition_lambda (int): Transition lambda parameter for DBN. Default is 100.
    """
    
    def __init__(
        self, 
        fps: int = 50, 
        beats_per_bar=(3, 4), 
        min_bpm=55.0, 
        max_bpm=215.0, 
        transition_lambda=100
    ):
        self.fps = fps
        self.beats_per_bar = beats_per_bar
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.transition_lambda = transition_lambda
        
        # Store parameters for lazy initialization
        self._dbn = None
    
    @property
    def dbn(self):
        """Lazy initialization of the DBN processor.
        
        The madmom import is deferred until the first time the DBN is needed.
        """
        if self._dbn is None:
            try:
                # Import here to avoid making madmom a required dependency for the package
                from madmom.features.downbeats import DBNDownBeatTrackingProcessor
                
                # Initialize madmom DBN
                self._dbn = DBNDownBeatTrackingProcessor(
                    fps=self.fps,
                    beats_per_bar=self.beats_per_bar,
                    min_bpm=self.min_bpm,
                    max_bpm=self.max_bpm,
                    transition_lambda=self.transition_lambda
                )
            except ImportError:
                raise ImportError(
                    "The madmom library is required for DbnPostprocessor. "
                    "Please install it using 'pip install madmom'."
                )
        return self._dbn
    
    def __call__(
        self,
        beat_logits: torch.Tensor,
        downbeat_logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Apply DBN-based postprocessing to the input beat and downbeat logits.
        Works with batched and unbatched inputs.
        
        Args:
            beat_logits (torch.Tensor): Beat prediction logits
            downbeat_logits (torch.Tensor): Downbeat prediction logits
            padding_mask (torch.Tensor, optional): Padding mask tensor. Defaults to None.
            
        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: Processed beat and downbeat times in seconds
        """
        batched = False if beat_logits.ndim == 1 else True
        if padding_mask is None:
            padding_mask = torch.ones_like(beat_logits, dtype=torch.bool)

        # if inputs are 1D tensors, add a batch dimension
        if not batched:
            beat_logits = beat_logits.unsqueeze(0)
            downbeat_logits = downbeat_logits.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
            
        # Convert logits to probabilities
        beat_prob = beat_logits.double().sigmoid()
        downbeat_prob = downbeat_logits.double().sigmoid()
        
        # Limit lower and upper bound, since 0 and 1 create problems in the DBN
        epsilon = 1e-5
        beat_prob = beat_prob * (1 - epsilon) + epsilon / 2
        downbeat_prob = downbeat_prob * (1 - epsilon) + epsilon / 2
        
        # Process each item in the batch
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._process_single_item, beat_prob, downbeat_prob, padding_mask
                )
            )
            
        # Remove the batch dimension if it was added
        if not batched:
            postp_beat = postp_beat[0]
            postp_downbeat = postp_downbeat[0]
            
        return postp_beat, postp_downbeat
    
    def _process_single_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        """Process a single item in the batch."""
        # Unpad the predictions
        beat_prob = padded_beat_prob[mask]
        downbeat_prob = padded_downbeat_prob[mask]
        
        # Build an artificial multiclass prediction, as suggested by Böck et al.
        # Limit the lower bound to avoid problems with the DBN
        epsilon = 1e-5
        combined_act = np.vstack(
            (
                np.maximum(
                    beat_prob.cpu().numpy() - downbeat_prob.cpu().numpy(), epsilon / 2
                ),
                downbeat_prob.cpu().numpy(),
            )
        ).T
        
        # Run the DBN (this will trigger lazy initialization)
        dbn_out = self.dbn(combined_act)
        
        # Extract beat and downbeat times in seconds
        postp_beat = dbn_out[:, 0]  # First column contains beat times
        postp_downbeat = dbn_out[dbn_out[:, 1] == 1][:, 0]  # Second column indicates if a beat is a downbeat
        
        return postp_beat, postp_downbeat 