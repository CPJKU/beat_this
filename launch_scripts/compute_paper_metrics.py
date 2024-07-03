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


def main(args):
    if len(args.models) == 1:
        print("Single model prediction for", args.models[0])
        # single model prediction
        modelfile = args.models[0]
        # create datamodule
        datamodule = datamodule_setup(args, modelfile)
        # create model and trainer
        model, trainer = model_setup(args, modelfile)
        # predict
        metrics, dataset, preds, piece = compute_predictions(
            model, trainer, datamodule.predict_dataloader())

        # compute averaged metrics
        averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
        # compute metrics averaged by dataset
        dataset_metrics = {k: {d: np.mean(v[dataset == d]) for d in np.unique(
            dataset)} for k, v in metrics.items()}
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
    else:  # multiple models
        if args.aggregation_type == "mean-std":
            # computing result variability for the same dataset and different model seeds
            # create datamodule only once, as we assume it is the same for all models
            datamodule = datamodule_setup(args, args.models[0])
            # create model and trainer
            models = []
            trainers = []
            for modelfile in args.models:
                model, trainer = model_setup(args, checkpoint_path=modelfile)
                models.append(model)
                trainers.append(trainer)
            # predict
            all_metrics = []
            for model, trainer in zip(models, trainers):
                metrics, dataset, preds, piece = compute_predictions(
                    model, trainer, datamodule.predict_dataloader())
                # compute averaged metrics for one model
                averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
                all_metrics.append(averaged_metrics)
            # compute mean and standard deviations for all model averages
            all_metrics_mean = {k: np.mean(
                [m[k] for m in all_metrics]) for k in all_metrics[0]}
            all_metrics_std = {
                k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]}
            all_metrics_stats = {
                k: (all_metrics_mean[k], all_metrics_std[k]) for k, v in all_metrics[0].items()}
            # print all metrics
            print("Metrics")
            for k, v in all_metrics_stats.items():
                print(f"{k}: {v[0]} +- {v[1]}")
        elif args.aggregation_type == "k-fold":
            # computing results in the K-fold setting. Every fold has a different dataset
            all_piece_metrics = []
            all_piece_dataset = []
            all_piece = []
            # create datamodule for each model
            for i_model, modelfile in enumerate(args.models):
                print(f"Model {i_model+1}/{len(args.models)}")
                datamodule = datamodule_setup(args, modelfile)
                # create model and trainer
                model, trainer = model_setup(args, modelfile)
                # predict
                metrics, dataset, preds, piece = compute_predictions(
                    model, trainer, datamodule.predict_dataloader())
                all_piece_metrics.append(metrics)
                all_piece_dataset.append(dataset)
                all_piece.append(piece)
            # aggregate across folds
            all_piece_metrics = {k: np.concatenate(
                [m[k] for m in all_piece_metrics]) for k in all_piece_metrics[0]}
            all_piece_dataset = np.concatenate(all_piece_dataset)
            all_piece = np.concatenate(all_piece)
            # double check that there are no errors in the fold and there are not repeated pieces
            assert len(all_piece) == len(np.unique(all_piece)
                                         ), "There are repeated pieces in the folds"
            dataset_metrics = {k: {d: np.mean(v[all_piece_dataset == d]) for d in np.unique(
                all_piece_dataset)} for k, v in all_piece_metrics.items()}
            # create a dataframe with the dataset_metrics
            dataset_metrics_df = pd.DataFrame(dataset_metrics)
            # print for dataset
            print("Dataset metrics")
            for k, v in dataset_metrics.items():
                print(k)
                for d, value in v.items():
                    print(f"{d}: {value}")
                print("------")
        else:
            raise ValueError(
                f"Unknown aggregation type {args.aggregation_type}")


def datamodule_setup(args, modelfile):
    # Load the datamodule
    print("Creating datamodule")
    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / 'data'
    if str(modelfile).startswith("https://"):
        datamodule_hparams = torch.hub.load_state_dict_from_url(modelfile)[
        'datamodule_hyper_parameters']
    else:
        datamodule_hparams = torch.load(modelfile, map_location='cpu')[
        'datamodule_hyper_parameters']
    # update the hparams with the ones from the arguments
    if args.num_workers is not None:
        datamodule_hparams["num_workers"] = args.num_workers
    datamodule_hparams["predict_datasplit"] = args.datasplit
    datamodule_hparams["data_dir"] = data_dir
    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage='test' if args.datasplit ==
                     'test' else 'fit')
    return datamodule


def model_setup(args, checkpoint_path):
    model_hparams = {}
    if args.eval_trim_beats is not None:
        model_hparams['eval_trim_beats'] = args.eval_trim_beats
    if args.dbn is not None:
        model_hparams['use_dbn'] = args.dbn

    model = PLBeatThis.load_from_checkpoint(checkpoint_path, map_location='cpu',
                                            **model_hparams)
    trainer = Trainer(
        accelerator="auto",
        devices=[args.gpu],
        logger=None,
        deterministic=True,
        precision='16-mixed',
    )
    return model, trainer


def compute_predictions(model, trainer, predict_dataloader):
    print("Computing predictions ...")
    out = trainer.predict(model, predict_dataloader)
    metrics = [o[0] for o in out]
    preds = [o[1] for o in out]
    dataset = np.asarray([o[2][0] for o in out])
    piece = np.asarray([o[3][0] for o in out])
    # convert metrics from list of per-batch dictionaries to a single dictionary with np arrays as values
    metrics = {k: np.asarray([m[k] for m in metrics]) for k in metrics[0]}
    return metrics, dataset, preds, piece


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes predictions for a given model and dataset, "
        "prints metrics, and optionally dumps predictions to a given file.")
    parser.add_argument("--models", type=str,
                        nargs='+',
                        required=True,
                        help="Local checkpoint files to use")
    parser.add_argument("--datasplit", type=str,
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
        "--aggregation-type",
        type=str,
        choices=("mean-std", "k-fold"),
        default="mean-std",
        help="Type of aggregation to use for multiple models; ignored if only one model is given",
    )

    args = parser.parse_args()

    main(args)
