import argparse
from pathlib import Path

import numpy as np
from pytorch_lightning import Trainer, seed_everything

from beat_this.dataset import BeatDataModule
from beat_this.inference import load_checkpoint
from beat_this.model.pl_module import PLBeatThis

# for repeatability
seed_everything(0, workers=True)


def main(args):
    if len(args.models) == 1:
        print("Single model prediction for", args.models[0])
        # single model prediction
        checkpoint_path = args.models[0]
        checkpoint = load_checkpoint(checkpoint_path)

        # create datamodule
        datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)
        # create model and trainer
        model, trainer = plmodel_setup(
            checkpoint, args.eval_trim_beats, args.dbn, args.use_dbn_eval,
            args.eval_dbn_beats_per_bar, args.eval_dbn_min_bpm, 
            args.eval_dbn_max_bpm, args.eval_dbn_transition_lambda, args.gpu
        )
        # predict
        metrics, dataset, preds, piece = compute_predictions(
            model, trainer, datamodule.predict_dataloader()
        )

        # compute averaged metrics
        averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
        # compute metrics averaged by dataset
        dataset_metrics = {
            k: {d: np.mean(v[dataset == d]) for d in np.unique(dataset)}
            for k, v in metrics.items()
        }
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
            checkpoint = load_checkpoint(args.models[0])
            datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)
            # create model and trainer
            all_metrics = []
            for checkpoint_path in args.models:
                checkpoint = load_checkpoint(checkpoint_path)
                model, trainer = plmodel_setup(
                    checkpoint, args.eval_trim_beats, args.dbn, args.use_dbn_eval,
                    args.eval_dbn_beats_per_bar, args.eval_dbn_min_bpm, 
                    args.eval_dbn_max_bpm, args.eval_dbn_transition_lambda, args.gpu
                )

                metrics, dataset, preds, piece = compute_predictions(
                    model, trainer, datamodule.predict_dataloader()
                )
                # compute averaged metrics for one model
                averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
                all_metrics.append(averaged_metrics)
            # compute mean and standard deviations for all model averages
            all_metrics_mean = {
                k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]
            }
            all_metrics_std = {
                k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]
            }
            all_metrics_stats = {
                k: (all_metrics_mean[k], all_metrics_std[k])
                for k, v in all_metrics[0].items()
            }
            # print all metrics
            print("Metrics")
            for k, v in all_metrics_stats.items():
                # round to 3 decimal places
                print(f"{k}: {round(v[0],3)} +- {round(v[1],3)}")
        elif args.aggregation_type == "k-fold":
            # computing results in the K-fold setting. Every fold has a different dataset
            all_piece_metrics = []
            all_piece_dataset = []
            all_piece = []
            # create datamodule for each model
            for i_model, checkpoint_path in enumerate(args.models):
                print(f"Model {i_model+1}/{len(args.models)}")
                checkpoint = load_checkpoint(checkpoint_path)
                datamodule = datamodule_setup(
                    checkpoint, args.num_workers, args.datasplit
                )
                # create model and trainer
                model, trainer = plmodel_setup(
                    checkpoint, args.eval_trim_beats, args.dbn, args.use_dbn_eval,
                    args.eval_dbn_beats_per_bar, args.eval_dbn_min_bpm, 
                    args.eval_dbn_max_bpm, args.eval_dbn_transition_lambda, args.gpu
                )
                # predict
                metrics, dataset, preds, piece = compute_predictions(
                    model, trainer, datamodule.predict_dataloader()
                )
                all_piece_metrics.append(metrics)
                all_piece_dataset.append(dataset)
                all_piece.append(piece)
            # aggregate across folds
            all_piece_metrics = {
                k: np.concatenate([m[k] for m in all_piece_metrics])
                for k in all_piece_metrics[0]
            }
            all_piece_dataset = np.concatenate(all_piece_dataset)
            all_piece = np.concatenate(all_piece)
            # double check that there are no errors in the fold and there are not repeated pieces
            assert len(all_piece) == len(
                np.unique(all_piece)
            ), "There are repeated pieces in the folds"
            dataset_metrics = {
                k: {
                    d: np.mean(v[all_piece_dataset == d])
                    for d in np.unique(all_piece_dataset)
                }
                for k, v in all_piece_metrics.items()
            }
            # print for dataset
            print("Dataset metrics")
            for k, v in dataset_metrics.items():
                print(k)
                for d, value in v.items():
                    print(f"{d}: {round(value,3)}")
                print("------")
        else:
            raise ValueError(f"Unknown aggregation type {args.aggregation_type}")


def datamodule_setup(checkpoint, num_workers, datasplit):
    # Load the datamodule
    print("Creating datamodule")
    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    datamodule_hparams = checkpoint["datamodule_hyper_parameters"]
    # update the hparams with the ones from the arguments
    if num_workers is not None:
        datamodule_hparams["num_workers"] = num_workers
    datamodule_hparams["predict_datasplit"] = datasplit
    datamodule_hparams["data_dir"] = data_dir
    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage="predict")
    return datamodule


def plmodel_setup(checkpoint, eval_trim_beats, dbn, use_dbn_eval, 
                 eval_dbn_beats_per_bar, eval_dbn_min_bpm,
                 eval_dbn_max_bpm, eval_dbn_transition_lambda, gpu):
    """
    Set up the pytorch lightning model and trainer for evaluation.

    Args:
        checkpoint (dict): The dict containing the checkpoint to load.
        eval_trim_beats (int or None): The number of beats to trim during evaluation. If None, the setting is taken from the pretrained model.
        dbn (bool or None): Whether to use the Dynamic Bayesian Network (DBN) module during evaluation (deprecated). If None, the default behavior from the pretrained model is used.
        use_dbn_eval (bool or None): Whether to use the DBN postprocessor for evaluation. If None, the default behavior from the pretrained model is used.
        eval_dbn_beats_per_bar (tuple or None): Possible beats per bar for DBN postprocessor. If None, the default is used.
        eval_dbn_min_bpm (float or None): Minimum BPM for DBN postprocessor. If None, the default is used.
        eval_dbn_max_bpm (float or None): Maximum BPM for DBN postprocessor. If None, the default is used.
        eval_dbn_transition_lambda (float or None): Transition lambda for DBN postprocessor. If None, the default is used.
        gpu (int): The index of the GPU device to use for training.

    Returns:
        tuple: A tuple containing the initialized pytorch lightning model and trainer.
    """
    hparams = checkpoint["hyper_parameters"]
    
    # Override evaluation parameters if provided
    if eval_trim_beats is not None:
        hparams["eval_trim_beats"] = eval_trim_beats
    
    # Handle deprecated dbn parameter for backward compatibility
    if dbn is not None:
        import warnings
        warnings.warn(
            "The 'dbn' parameter is deprecated and will be removed in a future version. "
            "Use 'use_dbn_eval' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if use_dbn_eval is None:  # Only use deprecated param if new one is not provided
            hparams["use_dbn_eval"] = dbn
    
    # Override postprocessor configuration if provided
    if use_dbn_eval is not None:
        hparams["use_dbn_eval"] = use_dbn_eval
    
    if eval_dbn_beats_per_bar is not None:
        hparams["eval_dbn_beats_per_bar"] = eval_dbn_beats_per_bar
    
    if eval_dbn_min_bpm is not None:
        hparams["eval_dbn_min_bpm"] = eval_dbn_min_bpm
    
    if eval_dbn_max_bpm is not None:
        hparams["eval_dbn_max_bpm"] = eval_dbn_max_bpm
    
    if eval_dbn_transition_lambda is not None:
        hparams["eval_dbn_transition_lambda"] = eval_dbn_transition_lambda

    model = PLBeatThis(**hparams)
    model.load_state_dict(checkpoint["state_dict"])
    # set correct device and accelerator
    if gpu >= 0:
        devices = [gpu]
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"
    # create trainer
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=None,
        deterministic=True,
        precision="16-mixed",
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
        "prints metrics, and optionally dumps predictions to a given file."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Local checkpoint files to use",
    )
    parser.add_argument(
        "--datasplit",
        type=str,
        choices=("train", "val", "test"),
        default="val",
        help="data split to use: train, val or test " "(default: %(default)s)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--num_workers", type=int, default=8, help="number of data loading workers "
    )
    parser.add_argument(
        "--eval_trim_beats",
        metavar="SECONDS",
        type=float,
        default=None,
        help="Override whether to skip the first given seconds "
        "per piece in evaluating (default: as stored in model)",
    )
    parser.add_argument(
        "--dbn",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Override the option to use madmom postprocessing dbn (deprecated, use --use-dbn-eval instead)",
    )
    parser.add_argument(
        "--use-dbn-eval",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Override whether to use DBN postprocessor for evaluation",
    )
    parser.add_argument(
        "--eval-dbn-beats-per-bar",
        type=int,
        nargs='+',
        default=None,
        help="Override possible beats per bar for DBN postprocessor"
    )
    parser.add_argument(
        "--eval-dbn-min-bpm",
        type=float,
        default=None,
        help="Override minimum BPM for DBN postprocessor"
    )
    parser.add_argument(
        "--eval-dbn-max-bpm",
        type=float,
        default=None,
        help="Override maximum BPM for DBN postprocessor"
    )
    parser.add_argument(
        "--eval-dbn-transition-lambda",
        type=float,
        default=None,
        help="Override transition lambda for DBN postprocessor"
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
