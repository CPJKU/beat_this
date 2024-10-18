"""
Loss definitions for the Beat This! beat tracker.
"""

import torch
import torch.nn.functional as F


class MaskedBCELoss(torch.nn.Module):
    """
    Plain binary cross-entropy loss. Expects predictions to be given as logits,
    and accepts an optional mask with zeros indicating the entries to ignore.

    Args:
        pos_weight (float): Weight for positive examples compared to negative
            examples (default: 1)
    """

    def __init__(self, pos_weight: float = 1):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        return F.binary_cross_entropy_with_logits(
            preds, targets, weight=mask, pos_weight=self.pos_weight
        )


class ShiftTolerantBCELoss(torch.nn.Module):
    """
    BCE loss variant for sequence labeling that tolerates small shifts between
    predictions and targets. This is accomplished by max-pooling the
    predictions with a given tolerance and a stride of 1, so the gradient for a
    positive label affects the largest prediction in a window around it.
    Expects predictions to be given as logits, and accepts an optional mask
    with zeros indicating the entries to ignore. Note that the edges of the
    sequence will not receive a gradient, as it is assumed to be unknown
    whether there is a nearby positive annotation.

    Args:
        pos_weight (float): Weight for positive examples compared to negative
            examples (default: 1)
        tolerance (int): Tolerated shift in time steps in each direction
            (default: 3)
    """

    def __init__(self, pos_weight: float = 1, tolerance: int = 3):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.tolerance = tolerance

    def spread(self, x: torch.Tensor, factor: int = 1):
        if self.tolerance == 0:
            return x
        return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, x: torch.Tensor, factor: int = 1):
        return x[..., factor * self.tolerance : -factor * self.tolerance or None]

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # spread preds and crop targets to match
        spreaded_preds = self.crop(self.spread(preds))
        cropped_targets = self.crop(targets, factor=2)
        # ignore around the positive targets
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:  # consider padding and no-downbeat mask
            look_at = look_at * self.crop(mask, factor=2)
        # compute loss
        return F.binary_cross_entropy_with_logits(
            spreaded_preds,
            cropped_targets,
            weight=look_at,
            pos_weight=self.pos_weight,
        )


class SplittedShiftTolerantBCELoss(torch.nn.Module):
    """
    Alternative implementation of ShiftTolerantBCELoss that splits the loss for
    positive and negative targets. This is mainly provided as it may be a bit
    easier to understand and compare with the Beat This! paper. Note that for
    non-binary targets (e.g., with label smoothing), this implementation
    matches the equation in the paper (Section 3.3), while ShiftTolerantBCELoss
    deviates from it. For binary targets, the results are identical.

    Args:
        pos_weight (int): weight of positive targets
        spread_preds (int): amount of temporal max-pooling applied to predictions
    """

    def __init__(self, pos_weight: float = 1, tolerance: int = 3):
        super().__init__()
        self.tolerance = 3
        self.spread_preds = tolerance
        self.spread_targets = 2 * tolerance  # targets are always spreaded twice as much
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )

    def spread(self, x: torch.Tensor, amount: int):
        if amount:
            return F.max_pool1d(x, 1 + 2 * amount, 1)
        else:
            return x

    def crop(self, x: torch.Tensor, desired_length: int):
        amount = (x.shape[-1] - desired_length) // 2
        if amount > 0:
            return x[..., amount:-amount]
        elif amount == 0:
            return x
        else:
            raise ValueError("Desired length must be smaller than input length")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        output_length = targets.size(-1) - 2 * self.spread_targets
        # compute loss for positive targets, we spread preds
        preds = self.spread(preds, self.spread_preds)
        # we crop preds and targets (and mask) to ignore problems at the edges due to the maxpool operation
        cropped_preds = self.crop(preds, output_length)
        cropped_targets = self.crop(targets, output_length)
        cropped_mask = self.crop(mask, output_length)
        loss_positive = F.binary_cross_entropy_with_logits(
            cropped_preds,
            cropped_targets,
            weight=cropped_targets * cropped_mask,
            pos_weight=self.pos_weight,
        )

        # compute loss for negative targets, we spread targets and preds (already spreaded above)
        targets = self.spread(targets, self.spread_targets)
        cropped_targets = self.crop(targets, output_length)
        loss_negative = F.binary_cross_entropy_with_logits(
            cropped_preds,
            cropped_targets,
            weight=(1 - cropped_targets) * cropped_mask,
            pos_weight=self.pos_weight,  # ensures identical results to the other implementation
        )
        # sum the two losses
        return loss_positive + loss_negative
