from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning import Trainer, seed_everything
import torch
import argparse
import os
from pathlib import Path
# import wandb


from beat_this.dataset.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis


torch.multiprocessing.set_sharing_strategy("file_system")
# for repeatability
JBT_SEED = int(os.environ.get("JBT_SEED", 0))
seed_everything(JBT_SEED, workers=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-flash-attention", default=False,
        action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--compile",
        action='store', 
        nargs="*",
        type=str,
        default=[],
        # default=["frontend","transformer_blocks"],
        help="Which model parts to compile, among frontend, transformer_encoder"
    )
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--total-dim", type=int, default=512)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout rate to apply throughout the model",
    )
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--logger", type=str, choices=["wandb", "none"], default="none"
    )
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--fps", type=int, default=50, help="The spectrograms fps.")
    parser.add_argument(
        "--loss",
        type=str,
        default="shift_tolerant_weighted_bce",
        choices=["shift_tolerant_weighted_bce", "weighted_bce", "bce"],
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
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument(
        "--val-frequency",
        metavar="N",
        type=int,
        default=5,
        help="validate every N epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--time-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed time aumentation",
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
        "--test-mode", action="store_true", help="test mode to fast check the system"
    )
    parser.add_argument(
        "--lenght-based-oversampling-factor",
        type=float,
        default=0.65,
        help="The factor to oversample the long pieces in the dataset. Set to 0 to only take one excerpt for each piece.",
    )
    parser.add_argument(
        "--train-datasets",
        type=str,
        default="None",
        help="A comma separated list of datasets to train on. None for all datasets.",
    )
    parser.add_argument(
        "--val-datasets",
        type=str,
        default="None",
        help="A comma separated list of datasets to validate on. None for all datasets, and empty string for no validation.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If given, the CV fold number to *not* train on (0-based).",
    )

    args = parser.parse_args()

    print("Starting a new run with the following parameters:")
    print(args)

    if args.logger == "wandb":
        name = f"BTr-{args.loss}-lr{args.lr}-n{args.n_layers}-h{args.total_dim}-d{args.dropout}-bs{args.batch_size}-aug{args.time_augmentation}{args.pitch_augmentation}{args.mask_augmentation}"
        logger = WandbLogger(project="JBT", entity="vocsep", name=name)
    else:
        logger= None
            

    if args.force_flash_attention:
        print("Forcing the use of the flash attention")
        # set for flash attention    
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    augmentations = {}
    if args.time_augmentation:
        augmentations["time"] = {"min": -20, "max": 20, "stride": 4}
    if args.pitch_augmentation:
        augmentations["pitch"] = {"min": -5, "max": 6}
    if args.mask_augmentation:
        # kind, min_count, max_count, min_len, max_len, min_parts, max_parts
        augmentations["mask"] = {"kind":"permute",
                                 "min_count":1,
                                 "max_count":6,
                                 "min_len":0.1,
                                 "max_len": 2,
                                 "min_parts": 5,
                                 "max_parts": 9}
        
    datamodule = BeatDataModule(
        data_dir,
        batch_size=args.batch_size,
        train_length=args.train_length,
        spect_fps = args.fps,
        num_workers=args.num_workers,
        test_dataset="gtzan",
        test_mode=args.test_mode,
        lenght_based_oversampling_factor=args.lenght_based_oversampling_factor,
        augmentations=augmentations,
        train_datasets=(
            args.train_datasets.split(",") if args.train_datasets != "None" else None
        ),
        val_datasets=(
            args.val_datasets.split(",") if args.val_datasets != "None" else None
        ),
        fold=args.fold,
    )
    datamodule.setup()

    # compute positive weights
    pos_weights = datamodule.get_train_positive_weights(
        widen_target_mask=3
    )
    print("Using positive weights: ", pos_weights)
    pl_model = PLBeatThis(spect_dim=128, fps=50, total_dim=args.total_dim, ff_mult=4, n_layers=args.n_layers, stem_dim=32, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay, pos_weights=pos_weights, head_dim=32, loss_type=args.loss, warmup_steps=args.warmup_steps, max_epochs=args.max_epochs, use_dbn=args.dbn, eval_trim_beats=args.eval_trim_beats, predict_full_pieces=False)
    for part in args.compile:
        if hasattr(pl_model.model, part):
            setattr(pl_model.model, part, torch.compile(getattr(pl_model.model, part)))
            print("Will compile model", part)
        else:
            print("The model is missing the part", part, "to compile")

    callbacks = [LearningRateMonitor(logging_interval="step")]
    # save only the last model
    callbacks.append(ModelCheckpoint(every_n_epochs=1))

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
    trainer.test(
        pl_model, datamodule
    )  


if __name__ == "__main__":
    main()
