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
from beat_this.model.postprocessor import Postprocessor
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
        use_dbn=False,
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

        self.postprocessor = Postprocessor(
            type="dbn" if use_dbn else "minimal", fps=fps
        )
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
        postp_beat, postp_downbeat = self.postprocessor(
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
        postp_beat, postp_downbeat = self.postprocessor(
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
                "params": (
                    p for p in self.parameters() if p.requires_grad and p.ndim >= 2
                ),
                "weight_decay": self.weight_decay,
            },
            {
                "params": (
                    p for p in self.parameters() if p.requires_grad and p.ndim <= 1
                ),
                "weight_decay": 0,
            },
        ]

        optimizer = optimizer(params, lr=self.lr)

        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches
        )

        result = dict(optimizer=optimizer)
        result["lr_scheduler"] = {"scheduler": self.lr_scheduler, "interval": "step"}
        return result

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
        self.min_beat_time = eval_trim_beats

    def __call__(self, truth, preds, step) -> Any:
        truth = mir_eval.beat.trim_beats(truth, min_beat_time=self.min_beat_time)
        preds = mir_eval.beat.trim_beats(preds, min_beat_time=self.min_beat_time)
        if (
            step == "val"
        ):  # limit the metrics that are computed during validation to speed up training
            fmeasure = mir_eval.beat.f_measure(truth, preds)
            cemgil = mir_eval.beat.cemgil(truth, preds)
            return {"F-measure": fmeasure, "Cemgil": cemgil}
        elif step == "test":  # compute all metrics during testing
            CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(truth, preds)
            fmeasure = mir_eval.beat.f_measure(truth, preds)
            cemgil = mir_eval.beat.cemgil(truth, preds)
            return {"F-measure": fmeasure, "Cemgil": cemgil, "CMLt": CMLt, "AMLt": AMLt}
        else:
            raise ValueError("step must be either val or test")


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing over `max_iters` steps with `warmup` linear warmup steps.
    Optionally re-raises the learning rate for the final `raise_last` fraction
    of total training time to `raise_to` of the full learning rate, again with
    a linear warmup (useful for stochastic weight averaging).
    """

    def __init__(self, optimizer, warmup, max_iters, raise_last=0, raise_to=0.5):
        self.warmup = warmup
        self.max_num_iters = int((1 - raise_last) * max_iters)
        self.raise_to = raise_to
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if step < self.max_num_iters:
            progress = step / self.max_num_iters
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
            if step <= self.warmup:
                lr_factor *= step / self.warmup
        else:
            progress = (step - self.max_num_iters) / self.warmup
            lr_factor = self.raise_to * min(progress, 1)
        return lr_factor
