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
                        required=True,
                        help="Local checkpoint files to use")
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
    parser.add_argument(
        "--aggregation-type",
        type=str,
        choices=("mean-std", "K-fold"),
        default="mean-std",
        help="Type of aggregation to use for multiple models; ignored if only one model is given",
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
    
    if len(modelfiles) == 1:
        # single model prediction
        modelfile = modelfiles[0]
        # create datamodule
        datamodule = datamodule_setup(args, modelfile)
        predict_dataloader = getattr(datamodule, f'{args.predict_datasplit}_dataloader')()
        # create model and trainer
        model, trainer = model_setup(args, modelfile)
        # predict
        metrics, dataset, preds, piece = compute_predictions(model, trainer, predict_dataloader)
        
        # compute averaged metrics
        averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
        # compute metrics averaged by dataset
        dataset_metrics = {k: {d: np.mean(v[dataset == d]) for d in np.unique(dataset)} for k, v in metrics.items()}
        # create a dataframe with the dataset_metrics
        dataset_metrics_df = pd.DataFrame(dataset_metrics)
        # print for dataset
        print("Metrics")
        for k, v in averaged_metrics.items():
            print(f"{k}: {v}")
        print("Dataset metrics")
        for k, v in dataset_metrics.items():
            print(k)
            for d, value in v.items():
                print(f"{d}: {value}")
            print("------")
    else: # multiple models
        if args.aggregation_type == "mean-std":
            # computing result variability for the same dataset and different model seeds
            # create datamodule only once, as we assume it is the same for all models
            datamodule = datamodule_setup(args, modelfiles[0])
            predict_dataloader = getattr(datamodule, f'{args.predict_datasplit}_dataloader')()
            # create model and trainer
            models = []
            trainers = []
            for modelfile in modelfiles:
                model, trainer = model_setup(args, modelfile)
                models.append(model)
                trainers.append(trainer)
            # predict
            all_metrics = []
            for model, trainer in zip(models, trainers):
                metrics, dataset, preds, piece = compute_predictions(model, trainer, predict_dataloader)
                # compute averaged metrics for one model
                averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
                all_metrics.append(averaged_metrics)
            # compute mean and standard deviations for all model averages
            all_metrics_mean = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
            all_metrics_std = {k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]}
            all_metrics_stats = {k: (all_metrics_mean[k], all_metrics_std[k]) for k, v in all_metrics[0].items()}
            # print all metrics
            print("Metrics")
            for k, v in all_metrics_stats.items():
                print(f"{k}: {v[0]} +- {v[1]}")

                
                




def datamodule_setup(args, modelfile):
    # Load the datamodule
    print("Creating datamodule")
    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / 'data'
    datamodule_hparams = torch.load(modelfile, map_location='cpu')['datamodule_hyper_parameters']
    # update the hparams with the ones from the arguments, or the one required for full piece prediction
    datamodule_hparams.update(
            data_dir=data_dir,
            length_based_oversampling_factor=0,
            augmentations={},
            batch_size=1,)
    if args.num_workers is not None:
        datamodule_hparams["num_workers"] = args.num_workers
    if args.full_piece_prediction:
        datamodule_hparams['train_length'] = None
    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage='test' if args.predict_datasplit == 'test' else 'fit')
    return datamodule

def model_setup(args, modelfile):
    model_hparams = {}
    if args.eval_trim_beats is not None:
        model_hparams['eval_trim_beats'] = args.eval_trim_beats
    if args.dbn is not None:
        model_hparams['use_dbn'] = args.dbn
    if args.full_piece_prediction:
        model_hparams['predict_full_pieces'] = True

    model = PLBeatThis.load_from_checkpoint(modelfile, map_location='cpu',
                                                **model_hparams)
    trainer = Trainer(
        accelerator="auto",
        devices=[args.gpu],
        logger=None,
        deterministic=True,
        precision='16-mixed',
    )
    return model, trainer

def compute_predictions(model, trainer, dataloader):
    print("Computing predictions ...")
    out = trainer.predict(model, dataloader)
    metrics = [o[0] for o in out]
    preds = [o[2] for o in out]
    dataset = np.asarray([o[3][0] for o in out])
    piece = np.asarray([o[4][0] for o in out])
    # convert metrics from list of per-batch dictionaries to a single dictionary with np arrays as values
    metrics = {k: np.asarray([m[k] for m in metrics]) for k in metrics[0]}
    return metrics, dataset, preds, piece

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
