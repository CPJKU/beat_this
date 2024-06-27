""" 
    Pytorch lightning modules
"""

from typing import Any
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule
from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import Postprocessor
import mir_eval
import torch.nn.functional as F
import beat_this.model.loss
try:
    import wandb
except ImportError:
    wandb = None
from concurrent.futures import ThreadPoolExecutor
from beat_this.dataset.dataset import split_piece


class PLBeatThis(LightningModule):
    def __init__(
        self,
        spect_dim = 128,
        fps = 50,
        transformer_dim = 512,
        ff_mult = 4,
        n_layers=6,
        stem_dim=32,
        dropout={"input" : 0.2, "frontend": 0.1, "transformer": 0.2}, 
        lr=0.0008,
        weight_decay=0.01,
        pos_weights = {"beat": 1, "downbeat": 1},
        head_dim = 32,
        loss_type = "shift_tolerant_weighted_bce",
        warmup_steps = 1000,
        max_epochs = 100,
        use_dbn = False,
        eval_trim_beats=5,
        predict_full_pieces = False,
    ):
        super().__init__()
        self.save_hyperparameters()     
        self.lr = lr
        self.weight_decay = weight_decay
        self.fps = fps
        # create model
        self.model = BeatThis(spect_dim=spect_dim, transformer_dim=transformer_dim, ff_mult=ff_mult, stem_dim=stem_dim, n_layers=n_layers, head_dim=head_dim, dropout=dropout)
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        # set up the losses
        self.pos_weights = pos_weights
        if loss_type == "shift_tolerant_weighted_bce":
            self.beat_loss = beat_this.model.loss.ShiftTolerantBCELoss(pos_weight=pos_weights["beat"])
            self.downbeat_loss = beat_this.model.loss.ShiftTolerantBCELoss(pos_weight=pos_weights["downbeat"])
        elif loss_type == "weighted_bce":
            self.beat_loss = beat_this.model.loss.MaskedBCELoss(pos_weight=pos_weights["beat"])
            self.downbeat_loss = beat_this.model.loss.MaskedBCELoss(pos_weight=pos_weights["downbeat"])
        elif loss_type == "bce":
            self.beat_loss = beat_this.model.loss.MaskedBCELoss()
            self.downbeat_loss = beat_this.model.loss.MaskedBCELoss()
        elif loss_type == "fast_shift_tolerant_weighted_bce":
            self.beat_loss = beat_this.model.loss.FastShiftTolerantBCELoss(pos_weight=pos_weights["beat"])
            self.downbeat_loss = beat_this.model.loss.FastShiftTolerantBCELoss(pos_weight=pos_weights["downbeat"])
        else:
            raise ValueError("loss_type must be one of 'shift_tolerant_weighted_bce', 'weighted_bce', 'bce'")

        self.postprocessor = Postprocessor(type="dbn" if use_dbn else "minimal", fps=fps)
        self.eval_trim_beats = eval_trim_beats
        self.predict_full_pieces = predict_full_pieces
        self.metrics = Metrics(eval_trim_beats=eval_trim_beats)
        

    def _compute_loss(self, batch, model_prediction):
        beat_mask = batch["padding_mask"]
        beat_loss = self.beat_loss(model_prediction["beat"], batch["truth_beat"].float(), beat_mask)
        # downbeat mask considers padding and also pieces which don't have downbeat annotations
        downbeat_mask = beat_mask  * batch["downbeat_mask"][:,None]
        downbeat_loss = self.downbeat_loss(model_prediction["downbeat"], batch["truth_downbeat"].float(), downbeat_mask)
        # sum the losses and return them in a dictionary for logging
        return {"beat": beat_loss, "downbeat" : downbeat_loss, "total" : beat_loss+downbeat_loss}

    def _compute_metrics(self, batch, model_prediction, step="val"):
        # compute for beat
        metrics_beat, piecewise_beat = self._compute_metrics_target(batch, model_prediction, target="beat", step=step)	
        # compute for downbeat
        metrics_downbeat, piecewise_downbeat = self._compute_metrics_target(batch,model_prediction, target="downbeat", step=step)
        
        # concatenate dictionaries
        metrics = {**metrics_beat, **metrics_downbeat}
        piecewise = {**piecewise_beat, **piecewise_downbeat}

        return metrics, piecewise
    
    def _compute_metrics_target(self, batch, model_prediction, target, step):  

        def compute_item(pospt_pred, truth_orig_target):
            # take the ground truth from the original version, so there are no quantization errors
            piece_truth_time = np.frombuffer(truth_orig_target)
            # run evaluation
            metrics = self.metrics(piece_truth_time, pospt_pred, step=step)
            
            return metrics, piece_truth_time

        with ThreadPoolExecutor() as executor:
            (piecewise_metrics, truth_time) = zip(*executor.map(compute_item,
                                                model_prediction[f"postp_{target}"], 
                                                batch[f"truth_orig_{target}"],
                                                ))

        
        # average the beat metrics across the dictionary
        batch_metric = {key + f"_{target}": np.mean([x[key] for x in piecewise_metrics]) for key in piecewise_metrics[0].keys()}
        # save non-averaged results for piecewise evaluation
        piecewise = {}
        if step == "test":
            piecewise[f"F-measure_{target}"] = [p["F-measure"] for p in piecewise_metrics]
            piecewise[f"CMLt_{target}"] = [p["CMLt"] for p in piecewise_metrics]
            piecewise[f"AMLt_{target}"] = [p["AMLt"] for p in piecewise_metrics]
 
        return batch_metric, piecewise

    def log_losses(self, losses, batch_size, step="train"):
        # log for separate targets
        for target in "beat", "downbeat":
            self.log(f"{step}_loss_{target}", losses[target].item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        # log total loss
        self.log(f"{step}_loss", losses["total"].item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

    def log_metrics(self, metrics, batch_size, step="val"):
        for key, value in metrics.items():
            self.log(f"{step}_{key}", value, prog_bar=key.startswith("F-measure"), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

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
        model_prediction = self.postprocessor(model_prediction, batch["padding_mask"])
        # compute the metrics
        metrics, piecewise = self._compute_metrics(batch, model_prediction, step="val")
        # log
        self.log_losses(losses, len(batch["spect"]), "val")
        self.log_metrics(metrics, batch["spect"].shape[0], "val")

    def test_step(self, batch, batch_idx):
        # run the model
        model_prediction = self.model(batch["spect"])
        # compute loss
        losses = self._compute_loss(batch, model_prediction)
        # postprocess the predictions
        model_prediction = self.postprocessor(model_prediction, batch["padding_mask"])
        # compute the metrics
        metrics, piecewise = self._compute_metrics(batch, model_prediction, step="test")
        # log
        self.log_losses(losses, len(batch["spect"]), "test")
        self.log_metrics(metrics, batch["spect"].shape[0], "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, chunk_size: int = 1500, overlap_mode: str = 'keep_first') -> Any:
        """
        Compute predictions and metrics for a batch (a dictionary with an "spect" key).
        If self.predict_full_pieces==True, will split up the audio into multiple chunks of chunk size,
         which should correspond to the length of the sequence the model was trained with.
        Potential overlaps between chunks can be handled in two ways:
        by keeping the predictions of the excerpt coming first (overlap_mode='keep_first'), or
        by keeping the predictions of the excerpt coming last (overlap_mode='keep_last').
        Note that overlaps appear as the last excerpt is moved backwards
        when it would extend over the end of the piece.
        """
        if self.predict_full_pieces:
            if batch["spect"].shape[0] != 1:
                raise ValueError("When `predict_full_pieces` is True, only `batch_size=1` is supported")
            if torch.any(~batch["padding_mask"]):
                raise ValueError("When `predict_full_pieces` is True, the Dataset must not pad inputs")
            # compute border size according to the loss type
            if hasattr(self.beat_loss,"spread_targets"): # discard the edges that are affected by the max-pooling in the loss
                border_size = self.beat_loss.spread_targets
            else:
                border_size = 0
            model_prediction = split_predict_aggregate(batch["spect"][0], chunk_size, border_size, overlap_mode, self.model)
            
        else:
            # run the model
            model_prediction = self.model(batch["spect"])

        # postprocess the predictions
        model_prediction = self.postprocessor(model_prediction, batch["padding_mask"])
        # compute the metrics
        metrics, piecewise = self._compute_metrics(batch, model_prediction, step="test")
        return metrics, piecewise, model_prediction, batch["dataset"], batch["spect_path"]


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW
        # only decay 2+-dimensional tensors, to exclude biases and norms
        # (filtering on dimensionality idea taken from Kaparthy's nano-GPT)
        params = [{'params': (p for p in self.parameters()
                                if p.requires_grad and p.ndim >= 2),
                    'weight_decay': self.weight_decay},
                    {'params': (p for p in self.parameters()
                                if p.requires_grad and p.ndim <= 1),
                    'weight_decay': 0}]

        optimizer = optimizer(params, lr=self.lr)

        self.lr_scheduler = CosineWarmupScheduler(optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches)

        result = dict(optimizer=optimizer)
        result['lr_scheduler'] = {"scheduler": self.lr_scheduler, "interval": "step"}
        return result
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # compiled models have _orig_mod in the state dict keys, remove it
        for key in list(state_dict.keys()):  # use list to take a snapshot of the keys
            if "._orig_mod" in key:
                state_dict[key.replace("._orig_mod", "")] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove _orig_mod prefixes for compiled models
        for key in list(state_dict.keys()):  # use list to take a snapshot of the keys
            if "._orig_mod" in key:
                state_dict[key.replace("._orig_mod", "")] = state_dict.pop(key)
        return state_dict


class Metrics:
    def __init__(self, eval_trim_beats : int) -> None:
        self.min_beat_time = eval_trim_beats
        
    def __call__(self, truth, preds, step) -> Any:
        truth = mir_eval.beat.trim_beats(truth, min_beat_time=self.min_beat_time)
        preds = mir_eval.beat.trim_beats(preds, min_beat_time=self.min_beat_time)
        if step == "val": # limit the metrics that are computed during validation to speed up training
            fmeasure =  mir_eval.beat.f_measure(truth, preds)
            cemgil = mir_eval.beat.cemgil(truth, preds)
            return {'F-measure':fmeasure, "Cemgil":cemgil}
        elif step == "test": # compute all metrics during testing
            CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(truth, preds)
            fmeasure =  mir_eval.beat.f_measure(truth, preds)
            cemgil = mir_eval.beat.cemgil(truth, preds)
            return {'F-measure':fmeasure, "Cemgil":cemgil, "CMLt":CMLt, "AMLt":AMLt}
        else:
            raise ValueError("step must be either val or test")
        

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing over `max_iters` steps with `warmup` linear warmup steps.
    Optionally re-raises the learning rate for the final `raise_last` fraction
    of total training time to `raise_to` of the full learning rate, again with
    a linear warmup (useful for stochastic weight averaging).
    """
    def __init__(self, optimizer, warmup, max_iters, raise_last=0, raise_to=.5):
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

def aggregate_prediction(pred_chunks, starts, full_size, chunk_size, border_size, overlap_mode, device):
    # cut the predictions to discard the border
    pred_chunks = [{'beat': pchunk['beat'][border_size:-border_size], 'downbeat': pchunk['downbeat'][border_size:-border_size]} for pchunk in pred_chunks]
    # assert all([chunk["beat"].shape[0] == chunk_size - 2*border_size for chunk in pred_chunks])
    # assert all([chunk["downbeat"].shape[0] == chunk_size - 2*border_size for chunk in pred_chunks])
    # aggregate the predictions for the whole piece
    piece_prediction_beat = torch.full((full_size,), -1000., device=device)
    piece_prediction_downbeat = torch.full((full_size,), -1000., device=device)
    if overlap_mode == 'keep_first':
        # process in reverse order, so predictions of earlier excerpts overwrite later ones
        pred_chunks = reversed(list(pred_chunks))
        starts = reversed(list(starts))
    for start, pchunk in zip(starts, pred_chunks):
        piece_prediction_beat[start + border_size:start + chunk_size - border_size] = pchunk["beat"]
        piece_prediction_downbeat[start + border_size:start + chunk_size - border_size] = pchunk["downbeat"]
    return piece_prediction_beat, piece_prediction_downbeat


def split_predict_aggregate(spect: torch.Tensor, chunk_size: int, border_size: int, overlap_mode: str, model: torch.nn.Module):
    """
    Function for pieces that are longer than the training length of the model.
    Split the input piece into chunks, run the model on them, and aggregate the predictions.

    Args:
        spect (torch.Tensor): the input piece
        chunk_size (int): the length of the chunks
        border_size (int): the size of the border that is discarded from the predictions
        overlap_mode (str): how to handle overlaps between chunks
        model (torch.nn.Module): the model to run

    Returns:
        dict: the model predictions
    """
    # split the piece into chunks
    chunks, starts = split_piece(spect, chunk_size, border_size= border_size, avoid_short_end=True)
    # run the model
    pred_chunks = [model(chunk.unsqueeze(0)) for chunk in chunks]
    # remove the extra dimension in beat and downbeat prediction due to batch size 1
    pred_chunks = [{"beat": p["beat"][0], "downbeat": p["downbeat"][0]} for p in pred_chunks]
    piece_prediction_beat, piece_prediction_downbeat = aggregate_prediction(pred_chunks, starts, spect.shape[0], chunk_size, border_size, overlap_mode, spect.device)
    # save it to model_prediction
    model_prediction = {}
    model_prediction["beat"] = piece_prediction_beat.unsqueeze(0)
    model_prediction["downbeat"] = piece_prediction_downbeat.unsqueeze(0)
    return model_prediction