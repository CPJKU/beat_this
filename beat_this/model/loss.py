import torch
import torch.nn.functional as F
from einops import rearrange


class FastShiftTolerantBCELoss(torch.nn.Module):
    def __init__(self, pos_weight: int=1, tolerance:int=3):
        super().__init__()
        if tolerance < 1:
            raise ValueError("tolerance must be greater than 0. Use normal BCELoss for tolerance 0")
        self.tolerance = tolerance
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.int32), persistent=False)

    def spread(self, x, factor=1):
        return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, x, factor=1):
        return x[..., factor * self.tolerance:-factor * self.tolerance or None]

    def forward(self, preds, targets, mask=None):
        # crop preds and targets
        cropped_targets= self.crop(targets, factor=2)
        wide_preds = self.crop(self.spread(preds))
        # ignore around the positive targets
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None: # consider padding and no-downbeat mask
            look_at = look_at * self.crop(mask, factor=2)
        # compute loss
        return F.binary_cross_entropy_with_logits(
            wide_preds, cropped_targets, weight=look_at,
            pos_weight=self.pos_weight, reduction='none').view(targets.size(0), -1).mean(1).mean()

class MaskedBCELoss(torch.nn.Module):
    def __init__(self, pos_weight: int=1):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.int32), persistent=False)

    def forward(self, preds, targets, mask):
        return F.binary_cross_entropy_with_logits(
            preds, targets, weight=mask, pos_weight=self.pos_weight)


class ShiftTolerantBCELoss(torch.nn.Module):
    """
    BCE loss applied separately to positive and negative targets with
    temporal max-pooling. It is meant to accept predictions that are
    in close temporal proximity to the target.
    The target and preds are cropped to ignore the edges of the sequence,
    where max-pooling would introduce artifacts.

    Args:
        pos_weight (int): weight of positive targets
        spread_preds (int): amount of temporal max-pooling applied to predictions
    """
    def __init__(self, pos_weight:int=1, spread_preds=3):
        super().__init__()
        self.spread_preds = spread_preds
        self.spread_targets = 2 * spread_preds # targets are always spreaded twice as much
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.int32), persistent=False)

    def spread(self, x, amount):
        if amount:
            return F.max_pool1d(x, 1 + 2 * amount, 1)
        else:
            return x

    def crop(self, x, desired_length):
        amount = (x.shape[-1] - desired_length) // 2
        if amount > 0:
            return x[..., amount:-amount]
        elif amount == 0:
            return x
        else:
            raise ValueError("Desired length must be smaller than input length")


    def forward(self, preds, targets, mask):
        output_length = targets.size(-1) - 2 * self.spread_targets
        # compute loss for positive targets, we spread preds
        preds = self.spread(preds, self.spread_preds)
        # we crop preds and targets (and mask) to ignore problems at the edges due to the maxpool operation
        cropped_preds = self.crop(preds, output_length)
        cropped_targets = self.crop(targets, output_length)
        cropped_mask = self.crop(mask, output_length)
        loss_positive = F.binary_cross_entropy_with_logits(cropped_preds, cropped_targets, weight=cropped_targets*cropped_mask, pos_weight=self.pos_weight)

        # compute loss for negative targets, we spread targets and preds (already spreaded above)
        targets = self.spread(targets, self.spread_targets)
        cropped_targets = self.crop(targets, output_length)
        loss_negative = F.binary_cross_entropy_with_logits(cropped_preds, cropped_targets, weight=(1 - cropped_targets)*cropped_mask)
        # sum the two losses
        return loss_positive + loss_negative