#!/usr/bin/env python3
"""
Beat This! command line inference tool.
"""
import argparse
import sys
from pathlib import Path

import torch
try:
    import tqdm
except ImportError:
    tqdm = None

from beat_this.inference import File2File


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detects beats in given audio files with a Beat This! model."
    )
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="An audio file to process, or a directory of such files. Can be given multiple times.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name, path or URL of checkpoint to use, will be downloaded if needed (default:%(default)s).",
        default="final0",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file name for a single input file, or output directory for multiple input files. If omitted, outputs are saved next to each input file by replacing or appending a suffix (see --suffix and --append).",
    )
    parser.add_argument(
        "--suffix",
        "-s",
        type=str,
        default=".beats",
        help="Suffix for output file names (default: %(default)s). Also see --append. Ignored if an explicit output file name is given.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="If given, append suffix to output file names instead of replacing the existing suffix. Ignored if an explicit output file name is given.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If given, do not overwrite existing output files, but skip them.",
    )
    parser.add_argument(
        "--touch-first",
        action="store_true",
        help="If given, create empty output file before processing. Combined with --skip-existing, allows to run multiple processes in parallel on the same set of files.",
    )
    parser.add_argument(
        "--dbn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Override the option to use madmom's postprocessing DBN.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Which GPU to use (not the number of GPUs), or -1 for CPU. Ignored if CUDA is not available. (default: %(default)s)",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="If given, uses half precision floating point arithmetics. Required for flash attention on GPU. (default: %(default)s)",
    )
    return parser


def derive_output_path(input_path, suffix, append, output=None, parent=None):
    """
    Determine the output file name for `input_path` using the given
    suffix. If given, `output` is the base directory for outputs, and
    `parent` is the directory that was given on the command line.
    """
    # output directory
    if output is None:
        output_path = input_path
    else:
        if parent is not None:
            input_path = input_path.relative_to(parent)
        else:
            input_path = input_path.name
        output_path = output / input_path
    # suffix
    if append:
        return output_path.parent / (output_path.name + suffix)
    else:
        return output_path.with_suffix(suffix)


def run(
    inputs, model, output, suffix, append, skip_existing, touch_first, dbn, gpu, float16
):
    # determine device
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    # prepare model
    file2file = File2File(model, device, float16, dbn)

    # process inputs
    inputs = [Path(item) for item in inputs]
    if output is not None:
        output = Path(output)
    if len(inputs) == 1 and not inputs[0].is_dir():
        # special case: single input file
        if output is None or output.is_dir():
            output = derive_output_path(inputs[0], suffix, append, output)
        file2file(inputs[0], output)
    else:
        # multiple inputs: first collect tasks so we can have a progress bar
        tasks = []
        for item in inputs:
            if item.is_dir():
                for fn in item.rglob("*"):
                    if not fn.name.endswith(suffix) and not fn.is_dir():
                        output_path = derive_output_path(
                            fn, suffix, append, output, parent=item
                        )
                        if not skip_existing or not output_path.exists():
                            tasks.append((fn, output_path))
            else:
                tasks.append((item, derive_output_path(item, suffix, append, output)))
        # then process all of them
        if tqdm is not None:
            tasks = tqdm.tqdm(tasks)
        for item, output in tasks:
            if touch_first:
                try:
                    output.touch(exist_ok=not skip_existing)
                except FileExistsError:
                    continue
            elif skip_existing and output.exists():
                continue
            try:
                file2file(item, output)
            except Exception:
                print(
                    f'Could not process "{item}". Rerun with this file alone for details.',
                    file=sys.stderr,
                )


def main():
    run(**vars(get_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
