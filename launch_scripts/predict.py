#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes predictions for a given model and dataset, prints metrics, and
optionally dumps predictions to a given file.
For usage information, call with --help.
"""

from pytorch_lightning import Trainer, seed_everything
import torch
import numpy as np
import argparse
from pathlib import Path
import wandb
import pandas as pd

from beat_this.dataset.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis

# for repeatability
seed_everything(0, workers=True)

def main():
    parser = argparse.ArgumentParser(
        description="Computes predictions for a given model and dataset, "
        "prints metrics, and optionally dumps predictions to a given file.")
    parser.add_argument("--models", type=str,
                        nargs='+',
                        help="Local checkpoint file to use")
    parser.add_argument("--output_type", type=str, default="dict", choices=("dict", "beat"),
                        help="output type: dict or .beat files (default: %(default)s)")
    parser.add_argument("--predict_datasplit", type=str,
                        choices=("train", "val", "test"),
                        default="val",
                        help="data split to use: train, val or test "
                        "(default: %(default)s)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of data loading workers ")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="batch size (default: as stored in model")
    parser.add_argument("--eval_trim_beats", metavar="SECONDS",
                        type=float, default=None,
                        help="Override whether to skip the first given seconds "
                        "per piece in evaluating (default: as stored in model)")
    parser.add_argument(
        "--dbn",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="override the option to use madmom postprocessing dbn",
    )
    parser.add_argument(
        "--full_piece_prediction", 
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Predict on full pieces instead of 1500 frames (~30s) segments",
    )

    args = parser.parse_args()

    # split the models in multiple single model
    if Path(args.models[0]).is_file():
        modelfiles = args.models
    else:
        # infer the path from the run id
        modelfiles = []
        for id in args.models:
            folder = Path("JBT",id,"checkpoints")
            potential_ckpts = list(folder.glob("*.ckpt"))
            if len(potential_ckpts) != 1:
                raise ValueError(f"Expected one .ckpt file in {folder}, found {len(potential_ckpts)}")
            modelfiles.append(potential_ckpts[0])
    
    print("Loading models", modelfiles)

    override_hparams = {}
    if args.eval_trim_beats is not None:
        override_hparams['eval_trim_beats'] = args.eval_trim_beats
    if args.dbn is not None:
        override_hparams['use_dbn'] = args.dbn
    if args.full_piece_prediction:
        override_hparams['predict_full_pieces'] = True
    
    # Load the first model to get the datamodule, we assume all datamodules have the same parameter
    print("Creating datamodule")
    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / 'data'
    hparams = torch.load(modelfiles[0], map_location='cpu')['datamodule_hyper_parameters']
    # update the hparams with the ones from the arguments
    hparams.update(
            data_dir=data_dir,
            length_based_oversampling_factor=0,
            augmentations={})
    for k in 'batch_size', 'num_workers':
        v = getattr(args, k)
        if v is not None:
            hparams[k] = v
    if args.full_piece_prediction:
        hparams['train_length'] = None
    datamodule = BeatDataModule(**hparams)
    datamodule.setup(stage='test' if args.predict_datasplit == 'test' else 'fit')

    if len(modelfiles) == 1:
        # single model prediction
        model = PLBeatThis.load_from_checkpoint(modelfiles[0], map_location='cpu',
                                                **override_hparams)
        print("Creating trainer")
        trainer = Trainer(
            accelerator="auto",
            devices=[args.gpu],
            logger=None,
            deterministic=True,
            precision='16-mixed',
        )
        print("Computing predictions")
        predict_dataloader = getattr(datamodule, f'{args.predict_datasplit}_dataloader')()
        metrics, piecewise = trainer.predict(model, predict_dataloader)
        print("predicted")

    # # Load all models and compute predictions
    # for modelfile in modelfiles:
    #     model = BeatTracker.load_from_checkpoint(modelfile, map_location='cpu',
    #                                             **override_hparams)
        
    #     print("Creating trainer")
    #     trainer = Trainer(
    #         accelerator="auto",
    #         devices=eval(args.gpus),
    #         logger=None,
    #         deterministic=True,
    #         precision='16-mixed',
    #     )

    #     print("Computing predictions")
    #     dataloader = getattr(datamodule, '%s_dataloader' % args.datasplit)()
    #     preds = trainer.predict(model, dataloader)
    #     # convert from list of per-batch dictionaries to a single dictionary,
    #     # excluding outputs that average over the batch since they are not useful
        
    #     preds = {k: concat([pred[k] for pred in preds])
    #             for k, v in preds[0].items()
    #             if getattr(v, 'ndim', 1) > 0}
    #     # add to the global ones
    #     for k, v in preds.items():
    #         if isinstance(v, list) and isinstance(v[0], float):
    #             if k not in all_metrics:
    #                 all_metrics[k] = []
    #             # We add the average
    #             all_metrics[k].append(np.mean(v))

        
    # # compute mean and standard deviations for all model averages
    # all_metrics_mean = {k: np.mean(v) for k, v in all_metrics.items()}
    # all_metrics_std = {k: np.std(v) for k, v in all_metrics.items()}
    # all_metrics = {k: (v, all_metrics_mean[k], all_metrics_std[k]) for k, v in all_metrics.items()}
    # # print all metrics
    # print("Metrics")
    # for k, v in all_metrics.items():
    #     print(f"{k}: {v[1]} +- {v[2]}")

    # # print something that is copiable in latex
    # print("Latex")
    # order= ['F-measure_beat', 'CMLt_beat', 'AMLt_beat', 'F-measure_downbeat', 'CMLt_downbeat', 'AMLt_downbeat']
    # # print for no_postp
    # print("No postprocessing")
    # outstring = ""
    # for k in order:
    #     # add rounded to 1 decimals and multiplied by 100
    #     mean = round(all_metrics[f"{k}_no_postp"][1]*100, 1)
    #     std = round(all_metrics[f"{k}_no_postp"][2]*100, 1)
    #     outstring += f" ${mean} \pm {std} $ & "
    # outstring = outstring[:-2]
    # outstring+= " \\"
    # print(outstring)

    # # print for postp
    # if args.postp is not None:
    #     print(f"Postprocessing {args.postp}")
    #     outstring = ""
    #     for k in order:
    #         # add rounded to 1 decimals and multiplied by 100
    #         mean = round(all_metrics[f"{k}_postp"][1]*100, 1)
    #         std = round(all_metrics[f"{k}_postp"][2]*100, 1)
    #         outstring += f" ${mean} \pm {std} $ & "
    #     outstring = outstring[:-2]
    #     outstring+= " \\\\"
    #     print(outstring)


if __name__ == "__main__":
    main()
