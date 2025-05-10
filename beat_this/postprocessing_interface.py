# beat_this/postprocessing_interface.py
import numpy as np
import torch
from typing import Protocol, Tuple, List, runtime_checkable

@runtime_checkable
class Postprocessor(Protocol):
    """Interface for beat and downbeat postprocessors."""
    def __call__(
        self,
        beat_logits: torch.Tensor,
        downbeat_logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        ... 