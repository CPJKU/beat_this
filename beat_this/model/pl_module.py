"""
Pytorch Lightning module, wraps a BeatThis model along with losses, metrics and
optimizers for training.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mir_eval
import numpy as np
import torch
from pytorch_lightning import LightningModule

import beat_this.model.loss
from beat_this.inference import split_predict_aggregate
from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor
from beat_this.postprocessing_interface import Postprocessor as PostprocessorInterface
from beat_this.utils import replace_state_dict_key


class PLBeatThis(LightningModule):
    def __init__(
        self,
        spect_dim=128,
        fps=50,
        transformer_dim=512,
        ff_mult=4,
        n_layers=6,
        stem_dim=32,
        dropout={"frontend": 0.1, "transformer": 0.2},
        lr=0.0008,
        weight_decay=0.01,
        pos_weights={"beat": 1, "downbeat": 1},
        head_dim=32,
        loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000,
        max_epochs=100,
        use_dbn=False,  # Deprecated, use use_dbn_eval instead
        use_dbn_eval=False,
        eval_dbn_beats_per_bar=(3, 4),
        eval_dbn_min_bpm=55.0,
        eval_dbn_max_bpm=215.0,
        eval_dbn_transition_lambda=100,
        eval_trim_beats=5,
        sum_head=True,
        partial_transformers=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.fps = fps
        # create model
        self.model = BeatThis(
            spect_dim=spect_dim,
            transformer_dim=transformer_dim,
            ff_mult=ff_mult,
            stem_dim=stem_dim,
            n_layers=n_layers,
            head_dim=head_dim,
            dropout=dropout,
            sum_head=sum_head,
            partial_transformers=partial_transformers,
        )
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        # set up the losses
        self.pos_weights = pos_weights
        if loss_type == "shift_tolerant_weighted_bce":
            self.beat_loss = beat_this.model.loss.ShiftTolerantBCELoss(
                pos_weight=pos_weights["beat"]
            )
            self.downbeat_loss = beat_this.model.loss.ShiftTolerantBCELoss(
                pos_weight=pos_weights["downbeat"]
            )
        elif loss_type == "weighted_bce":
            self.beat_loss = beat_this.model.loss.MaskedBCELoss(
                pos_weight=pos_weights["beat"]
            )
            self.downbeat_loss = beat_this.model.loss.MaskedBCELoss(
                pos_weight=pos_weights["downbeat"]
            )
        elif loss_type == "bce":
            self.beat_loss = beat_this.model.loss.MaskedBCELoss()
            self.downbeat_loss = beat_this.model.loss.MaskedBCELoss()
        elif loss_type == "splitted_shift_tolerant_weighted_bce":
            self.beat_loss = beat_this.model.loss.SplittedShiftTolerantBCELoss(
                pos_weight=pos_weights["beat"]
            )
            self.downbeat_loss = beat_this.model.loss.SplittedShiftTolerantBCELoss(
                pos_weight=pos_weights["downbeat"]
            )
        else:
            raise ValueError(
                "loss_type must be one of 'shift_tolerant_weighted_bce', 'weighted_bce', 'bce'"
            )

        # Handle deprecated use_dbn parameter
        if use_dbn and not use_dbn_eval:
            import warnings
            warnings.warn(
                "The 'use_dbn' parameter is deprecated and will be removed in a future version. "
                "Use 'use_dbn_eval' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            use_dbn_eval = use_dbn

        # Configure evaluation postprocessor
        if use_dbn_eval:
            self.eval_postprocessor = DbnPostprocessor(
                fps=self.fps,
                beats_per_bar=eval_dbn_beats_per_bar,
                min_bpm=eval_dbn_min_bpm,
                max_bpm=eval_dbn_max_bpm,
                transition_lambda=eval_dbn_transition_lambda
            )
        else:
            self.eval_postprocessor = MinimalPostprocessor(fps=self.fps)

        self.eval_trim_beats = eval_trim_beats
        self.metrics = Metrics(eval_trim_beats=eval_trim_beats)

    def _compute_loss(self, batch, model_prediction):
        beat_mask = batch["padding_mask"]
        beat_loss = self.beat_loss(
            model_prediction["beat"], batch["truth_beat"].float(), beat_mask
        )
        # downbeat mask considers padding and also pieces which don't have downbeat annotations
        downbeat_mask = beat_mask * batch["downbeat_mask"][:, None]
        downbeat_loss = self.downbeat_loss(
            model_prediction["downbeat"], batch["truth_downbeat"].float(), downbeat_mask
        )
        # sum the losses and return them in a dictionary for logging
        return {
            "beat": beat_loss,
            "downbeat": downbeat_loss,
            "total": beat_loss + downbeat_loss,
        }

    def _compute_metrics(self, batch, postp_beat, postp_downbeat, step="val"):
        """ """
        # compute for beat
        metrics_beat = self._compute_metrics_target(
            batch, postp_beat, target="beat", step=step
        )
        # compute for downbeat
        metrics_downbeat = self._compute_metrics_target(
            batch, postp_downbeat, target="downbeat", step=step
        )

        # concatenate dictionaries
        metrics = {**metrics_beat, **metrics_downbeat}

        return metrics

    def _compute_metrics_target(self, batch, postp_target, target, step):

        def compute_item(pospt_pred, truth_orig_target):
            # take the ground truth from the original version, so there are no quantization errors
            piece_truth_time = np.frombuffer(truth_orig_target)
            # run evaluation
            metrics = self.metrics(piece_truth_time, pospt_pred, step=step)

            return metrics

        # if the input was not batched, postp_target is an array instead of a tuple of arrays
        # make it a tuple for consistency
        if not isinstance(postp_target, tuple):
            postp_target = (postp_target,)

        with ThreadPoolExecutor() as executor:
            piecewise_metrics = list(
                executor.map(
                    compute_item,
                    postp_target,
                    batch[f"truth_orig_{target}"],
                )
            )

        # average the beat metrics across the dictionary
        batch_metric = {
            key + f"_{target}": np.mean([x[key] for x in piecewise_metrics])
            for key in piecewise_metrics[0].keys()
        }

        return batch_metric

    def log_losses(self, losses, batch_size, step="train"):
        # log for separate targets
        for target in "beat", "downbeat":
            self.log(
                f"{step}_loss_{target}",
                losses[target].item(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        # log total loss
        self.log(
            f"{step}_loss",
            losses["total"].item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def log_metrics(self, metrics, batch_size, step="val"):
        for key, value in metrics.items():
            self.log(
                f"{step}_{key}",
                value,
                prog_bar=key.startswith("F-measure"),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        # run the model
        model_prediction = self.model(batch["spect"])
        # compute loss
        losses = self._compute_loss(batch, model_prediction)
        self.log_losses(losses, len(batch["spect"]), "train")
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        # run the model
        model_prediction = self.model(batch["spect"])
        # compute loss
        losses = self._compute_loss(batch, model_prediction)
        # postprocess the predictions
        postp_beat, postp_downbeat = self.eval_postprocessor(
            model_prediction["beat"],
            model_prediction["downbeat"],
            batch["padding_mask"],
        )
        # compute the metrics
        metrics = self._compute_metrics(batch, postp_beat, postp_downbeat, step="val")
        # log
        self.log_losses(losses, len(batch["spect"]), "val")
        self.log_metrics(metrics, batch["spect"].shape[0], "val")

    def test_step(self, batch, batch_idx):
        metrics, model_prediction, _, _ = self.predict_step(batch, batch_idx)
        losses = self._compute_loss(batch, model_prediction)
        # log
        self.log_losses(losses, len(batch["spect"]), "test")
        self.log_metrics(metrics, batch["spect"].shape[0], "test")

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        chunk_size: int = 1500,
        overlap_mode: str = "keep_first",
    ) -> Any:
        """
        Compute predictions and metrics for a batch (a dictionary with an "spect" key).
        It splits up the audio into multiple chunks of chunk size,
         which should correspond to the length of the sequence the model was trained with.
        Potential overlaps between chunks can be handled in two ways:
        by keeping the predictions of the excerpt coming first (overlap_mode='keep_first'), or
        by keeping the predictions of the excerpt coming last (overlap_mode='keep_last').
        Note that overlaps appear as the last excerpt is moved backwards
        when it would extend over the end of the piece.
        """
        if batch["spect"].shape[0] != 1:
            raise ValueError(
                "When predicting full pieces, only `batch_size=1` is supported"
            )
        if torch.any(~batch["padding_mask"]):
            raise ValueError(
                "When predicting full pieces, the Dataset must not pad inputs"
            )
        # compute border size according to the loss type
        if hasattr(
            self.beat_loss, "tolerance"
        ):  # discard the edges that are affected by the max-pooling in the loss
            border_size = 2 * self.beat_loss.tolerance
        else:
            border_size = 0
        model_prediction = split_predict_aggregate(
            batch["spect"][0], chunk_size, border_size, overlap_mode, self.model
        )
        # add the batch dimension back in the prediction for consistency
        model_prediction = {
            key: value.unsqueeze(0) for key, value in model_prediction.items()
        }
        # postprocess the predictions
        postp_beat, postp_downbeat = self.eval_postprocessor(
            model_prediction["beat"], model_prediction["downbeat"], None
        )
        # compute the metrics
        metrics = self._compute_metrics(batch, postp_beat, postp_downbeat, step="test")
        return metrics, model_prediction, batch["dataset"], batch["spect_path"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW
        # only decay 2+-dimensional tensors, to exclude biases and norms
        # (filtering on dimensionality idea taken from Kaparthy's nano-GPT)
        params = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if p.dim() >= 2 and not n.startswith("model.frontend.norm")
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if p.dim() < 2 or n.startswith("model.frontend.norm")
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = optimizer(params, lr=self.lr, betas=(0.9, 0.95))
        # compute the cosine decay from the number of training steps
        # in case of NaN or unreasonable warmup_steps
        warmup_steps = self.warmup_steps
        if not 0 <= warmup_steps < 1e5:
            warmup_steps = 1000

        scheduler = CosineWarmupScheduler(
            optimizer=opt,
            warmup=warmup_steps,
            max_iters=self.max_epochs,
            # give a little bump to the LR at the very end
            # (could be useful for evaluation when we use the last checkpoint)
            raise_last=1,
            raise_to=0.5,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [opt], [scheduler_config]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        return state_dict


class Metrics:
    def __init__(self, eval_trim_beats: int) -> None:
        self.eval_trim_beats = eval_trim_beats

    def __call__(self, truth, preds, step) -> Any:
        """
        Compute metrics for a single piece.
        The metrics are computed for the validation set, and also for the test set (default).
        """
        # cut a few beats at the beginning and end of the ground truth
        # (because they are less consistent in the annotations, e.g. drum fills)
        trimmed_truth = truth.copy()
        if self.eval_trim_beats > 0 and len(trimmed_truth) > 2 * self.eval_trim_beats + 1:
            trimmed_truth = trimmed_truth[
                self.eval_trim_beats : -self.eval_trim_beats
            ]
        # compute the metrics on the validation set only if already fitted, on train and validation and test
        if preds.size == 0:
            # return zeroes
            return {
                "F-measure": 0,
                "P-score": 0,
                "R-score": 0,
                "AMLt": 0,
                "AMLc": 0,
                "Informaation Gain": 0,
            }
        if step == "val":
            # compute only the f-measure as a cheap approximation for validation when finding hyperparameters
            # we need at least one value to compute the metric
            return mir_eval.beat.f_measure(trimmed_truth, preds)
        else:
            # val-short and test set: let's compute precise values for publishing (costlier)
            scores = {}
            scores["F-measure"], scores["P-score"], scores[
                "R-score"
            ] = mir_eval.beat.f_measure(trimmed_truth, preds, beta=1.0)
            scores["AMLt"] = mir_eval.beat.continuity(trimmed_truth, preds)["total"]
            scores["AMLc"] = mir_eval.beat.continuity(
                trimmed_truth, preds, continuity_phase=False
            )["total"]
            scores["Information Gain"] = mir_eval.beat.information_gain(
                trimmed_truth, preds
            )
            return scores


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup and cosine decay.
    """

    def __init__(self, optimizer, warmup, max_iters, raise_last=0, raise_to=0.5):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.raise_last = raise_last
        self.raise_to = raise_to
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step=None, epoch=None):
        """
        Get a learning rate factor based on the current epoch or step.
        """
        if step is None:
            if epoch is None:
                step = 0
            else:
                step = epoch  # epochs for us
        if self.raise_last > 0 and step >= self.max_num_iters - self.raise_last:
            return self.raise_to
        if step < self.warmup:
            return float(step) / float(max(1, self.warmup))
        elif step >= self.max_num_iters:
            return 0.0
        else:
            return 0.5 * (
                1.0
                + np.cos(
                    np.pi
                    * float(step - self.warmup)
                    / float(max(1, self.max_num_iters - self.warmup))
                )
            )
