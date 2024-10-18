import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from beat_this.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis


def main(args):
    # for repeatability
    seed_everything(args.seed, workers=True)

    print("Starting a new run with the following parameters:")
    print(args)

    params_str = f"{'noval ' if not args.val else ''}{'hung ' if args.hung_data else ''}{'fold' + str(args.fold) + ' ' if args.fold is not None else ''}{args.loss}-h{args.transformer_dim}-aug{args.tempo_augmentation}{args.pitch_augmentation}{args.mask_augmentation}{' nosumH ' if not args.sum_head else ''}{' nopartialT ' if not args.partial_transformers else ''}"
    if args.logger == "wandb":
        logger = WandbLogger(
            project="beat_this", name=f"{args.name} {params_str}".strip()
        )
    else:
        logger = None

    if args.force_flash_attention:
        print("Forcing the use of the flash attention.")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    checkpoint_dir = (
        Path(__file__).parent.parent.relative_to(Path.cwd()) / "checkpoints"
    )
    augmentations = {}
    if args.tempo_augmentation:
        augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}
    if args.pitch_augmentation:
        augmentations["pitch"] = {"min": -5, "max": 6}
    if args.mask_augmentation:
        # kind, min_count, max_count, min_len, max_len, min_parts, max_parts
        augmentations["mask"] = {
            "kind": "permute",
            "min_count": 1,
            "max_count": 6,
            "min_len": 0.1,
            "max_len": 2,
            "min_parts": 5,
            "max_parts": 9,
        }

    datamodule = BeatDataModule(
        data_dir,
        batch_size=args.batch_size,
        train_length=args.train_length,
        spect_fps=args.fps,
        num_workers=args.num_workers,
        test_dataset="gtzan",
        length_based_oversampling_factor=args.length_based_oversampling_factor,
        augmentations=augmentations,
        hung_data=args.hung_data,
        no_val=not args.val,
        fold=args.fold,
    )
    datamodule.setup(stage="fit")

    # compute positive weights
    pos_weights = datamodule.get_train_positive_weights(widen_target_mask=3)
    print("Using positive weights: ", pos_weights)
    dropout = {
        "frontend": args.frontend_dropout,
        "transformer": args.transformer_dropout,
    }
    pl_model = PLBeatThis(
        spect_dim=128,
        fps=50,
        transformer_dim=args.transformer_dim,
        ff_mult=4,
        n_layers=args.n_layers,
        stem_dim=32,
        dropout=dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weights=pos_weights,
        head_dim=32,
        loss_type=args.loss,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        use_dbn=args.dbn,
        eval_trim_beats=args.eval_trim_beats,
        sum_head=args.sum_head,
        partial_transformers=args.partial_transformers,
    )
    for part in args.compile:
        if hasattr(pl_model.model, part):
            setattr(pl_model.model, part, torch.compile(getattr(pl_model.model, part)))
            print("Will compile model", part)
        else:
            raise ValueError("The model is missing the part", part, "to compile")

    callbacks = [LearningRateMonitor(logging_interval="step")]
    # save only the last model
    callbacks.append(
        ModelCheckpoint(
            every_n_epochs=1,
            dirpath=str(checkpoint_dir),
            filename=f"{args.name} S{args.seed} {params_str}".strip(),
        )
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=[args.gpu],
        num_sanity_val_steps=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_frequency,
    )

    trainer.fit(pl_model, datamodule)
    trainer.test(pl_model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--force-flash-attention", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--compile",
        action="store",
        nargs="*",
        type=str,
        default=["frontend", "transformer_blocks", "task_heads"],
        help="Which model parts to compile, among frontend, transformer_encoder, task_heads",
    )
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument(
        "--frontend-dropout",
        type=float,
        default=0.1,
        help="dropout rate to apply in the frontend",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.2,
        help="dropout rate to apply in the main transformer blocks",
    )
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--fps", type=int, default=50, help="The spectrograms fps.")
    parser.add_argument(
        "--loss",
        type=str,
        default="shift_tolerant_weighted_bce",
        choices=[
            "shift_tolerant_weighted_bce",
            "fast_shift_tolerant_weighted_bce",
            "weighted_bce",
            "bce",
        ],
        help="The loss to use",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="warmup steps for optimizer"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="max epochs for training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size for training"
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument(
        "--train-length",
        type=int,
        default=1500,
        help="maximum seq length for training in frames",
    )
    parser.add_argument(
        "--dbn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use madmom postprocessing DBN",
    )
    parser.add_argument(
        "--eval-trim-beats",
        metavar="SECONDS",
        type=float,
        default=5,
        help="Skip the first given seconds per piece in evaluating (default: %(default)s)",
    )
    parser.add_argument(
        "--val-frequency",
        metavar="N",
        type=int,
        default=5,
        help="validate every N epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--tempo-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed tempo aumentation",
    )
    parser.add_argument(
        "--pitch-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed pitch aumentation",
    )
    parser.add_argument(
        "--mask-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use online mask aumentation",
    )
    parser.add_argument(
        "--sum-head",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use SumHead instead of two separate Linear heads",
    )
    parser.add_argument(
        "--partial-transformers",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use Partial transformers in the frontend",
    )
    parser.add_argument(
        "--length-based-oversampling-factor",
        type=float,
        default=0.65,
        help="The factor to oversample the long pieces in the dataset. Set to 0 to only take one excerpt for each piece.",
    )
    parser.add_argument(
        "--val",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Train on all data, including validation data, escluding test data. The validation metrics will still be computed, but they won't carry any meaning.",
    )
    parser.add_argument(
        "--hung-data",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Limit the training to Hung et al. data. The validation will still be computed on all datasets.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If given, the CV fold number to *not* train on (0-based).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random number generators.",
    )

    args = parser.parse_args()

    main(args)
