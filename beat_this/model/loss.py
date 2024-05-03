import torch
import torch.nn.functional as F


# class ShiftTolerantBCELoss(torch.nn.Module):
#     def __init__(self, pos_weight: int=1, tolerance:int=3):
#         super().__init__(pos_weight)
#         if tolerance < 1:
#             raise ValueError("tolerance must be greater than 0. Use normal BCELoss for tolerance 0")
#         self.tolerance = tolerance
#         self.pos_weight = pos_weight

#     def spread(self, x, factor=1):
#         return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

#     def crop(self, x, factor=1):
#         return x[..., factor * self.tolerance:-factor * self.tolerance or None]

#     def forward(self, preds, targets, mask=None):
#         # crop preds and targets
#         # TODO: do we want this cropping?
#         targets= self.crop(targets, factor=2)
#         wide_preds = self.widen(preds)
#         preds = self.crop(wide_preds, factor=2)
#         # maxpoll preds
        
#         # ignore around the positive targets
#         look_at = targets + (1 - self.spread(targets, factor=2))
#         if mask is not None:
#             look_at = look_at * self.crop(mask, factor=2)
#         # compute loss
#         return F.binary_cross_entropy_with_logits(
#             wide_preds, targets, weight=look_at,
#             pos_weight=self.pos_weight, reduction='none').view(targets.size(0), -1).mean(1)


class ShiftTolerantBCELoss(torch.nn.Module):
    """
    BCE loss applied to only the positive or negative targets, with optional
    temporal max-pooling of predictions or targets. Can be used to recover
    variants of WideTargetMaskedBCELoss and ShiftTolerantBCELoss. When `hinge`
    is given, predictions closer than `hinge` to the target are clamped. When
    `focal` is given, treats it as the gamma exponent for a focal loss.
    """
    def __init__(self, pos_weight:int=1, spread_preds=3):
        super().__init__(pos_weight)
        self.spread_preds = spread_preds
        self.spread_targets = 2 * spread_preds # targets are always spreaded twice as much

    def spread(self, x, amount):
        if amount:
            return F.max_pool1d(x, 1 + 2 * amount, 1)
        else:
            return x

    def crop(self, x):
        return x[..., self.spread_targets:-self.spread_targets]


    def forward(self, preds, targets):
        # compute loss for positive targets, we spread preds
        preds = self.spread(preds, self.spread_preds)
        # we crop preds and targets to ignore problems at the edges and compute loss (with pos weights)
        loss_positive = F.binary_cross_entropy_with_logits(self.crop(preds), self.crop(targets), weight=self.crop(targets), pos_weight=self.pos_weight, reduction='none')

        # compute loss for negative targets, we spread targets and preds (already spreaded above)
        targets = self.spread(targets, self.spread_targets)
        loss_negative = F.binary_cross_entropy_with_logits(self.crop(preds), self.crop(targets), weight=1 - self.crop(targets), reduction='none')

        return (loss_positive + loss_negative).mean(1)