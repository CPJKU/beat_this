import argparse
from pathlib import Path

import torch


def main(args):
    # check if output path exists
    if Path(args.output_path).exists():
        print(f"Output path {args.output_path} already exists. Exiting.")
        return

    # load the lightning checkpoit
    checkpoint = torch.load(args.input_path, map_location="cpu")

    # clean and keep only the keys "state_dict" and "datamodule" to save space
    checkpoint = {
        k: v
        for k, v in checkpoint.items()
        if k
        in [
            "state_dict",
            "datamodule_hyper_parameters",
            "hyper_parameters",
            "pytorch-lightning_version",
        ]
    }

    # remove the "data_dir" key from "datamodule_hyper_parameters" because it is a
    # Posix path and creates problems when loading in Windows.
    if "data_dir" in checkpoint["datamodule_hyper_parameters"]:
        del checkpoint["datamodule_hyper_parameters"]["data_dir"]

    # save the cleaned checkpoint
    torch.save(checkpoint, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()

    main(args)
