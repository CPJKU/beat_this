# Beat This!
Official implementation of the beat tracker from the ISMIR 2024 paper "Beat This! Accurate Beat Tracking Without DBN Postprocessing".

## Available models
We release 3 main models, which were trained on all data, except the GTZAN dataset, with three different seeds. You can use them with the shortcut `final0`, `final1`, and `final2`. These correspond to "Our System" in Table 2 on the paper.
Please be aware that, as the models ```final*``` were trained on all data except the GTZAN dataset, the results may be unfairly good, if you run the inference on some data that was used for training.

The K-Fold models and single split from the paper are also available.

All the models are provided as pytorch lightning checkpoints, stripped of the optimizer state to reduce their size. This is useful for reproducing the paper results.
During inference, PyTorch lighting is not used, and the checkpoints are converted and loaded into vanilla PyTorch modules.


## Inference

To predict beats for audio files, you can either use our command line tool or call the beat tracker from Python. Both have the same requirements.

### Requirements

The beat tracker requires Python with a set of packages installed:
1. [Install PyTorch](https://pytorch.org/get-started/locally/) 2.0 or later following the instructions for your platform.
2. Install further modules with `pip install tqdm einops soxr rotary-embedding-torch`. (If using conda, we still recommend pip. You may try installing `soxr-python` and `einops` from conda-forge, but `rotary-embedding-torch` is only on PyPI.)
3. To read other audio formats than `.wav`, install `ffmpeg` or another supported backend for `torchaudio`. (`ffmpeg` can be installed via conda or via your operating system.)

Finally, install our beat tracker with:
```bash
pip install https://github.com/CPJKU/beat_this/archive/master.zip
```

### Command line

Along with the python package, a command line application called `beat_this` is installed. For a full documentation of the command line options, run:
```bash
beat_this --help
```
The basic usage is:
```bash
beat_this path/to/audio.file -o path/to/output.beats
```
To process multiple files, specify multiple input files or directories, and give an output directory instead:
```bash
beat_this path/to/*.mp3 path/to/whole_directory/ -o path/to/output_directory
```
The beat tracker will use the first GPU in your system by default, and fall back to CPU if PyTorch does not have CUDA access. With `--gpu=2`, it will use the third GPU, and with `--gpu=-1` it will force the CPU.
If you have a lot of files to process, you can distribute the load over multiple processes by running the same command multiple times with `--touch-first`, `--skip-existing` and potentially different options for `--gpu`:
```bash
for gpu in {0..3}; do beat_this input_dir -o output_dir --touch-first --skip-existing --gpu=$gpu &; done
```
If you want to use the DBN for postprocessing, add `--dbn`. The DBN parameters are the default ones from madmom. This requires installing the `madmom` package.

### Python class

If you are a Python user, you can directly use the `beat_this.inference` module.

First, instantiate an instance of the `Audio2Beat` class that encapsulates the model along with pre- and postprocessing:
```python
from beat_this.inference import Audio2Beat
audio2beat = Audio2Beat(model_checkpoint="final0", device="cuda", dbn=False)
```
To obtain a list of beats and downbeats for an audio file, run:
```python
audio_path = "path/to/audio.file"
beats, downbeats = audio2beat(audio_path)
```
Optionally, you can produce a `.beats` file (e.g., for importing into [Sonic Visualizer](https://www.sonicvisualiser.org/)):
```python
from beat_this.utils import save_beat_tsv
output_path = "path/to/output.beats"
save_beat_tsv(beats, downbeats, outpath)
```

## Training

This part will be available soon.


## Reproducing metrics from the paper

This part will be completed soon.

Compute results on the test set (GTZAN):
```
python launch_scripts/compute_paper_metrics.py --models final0 final1 final2 --datasplit test
```

Compute k-fold results:
```
python launch_scripts/compute_paper_metrics.py --aggregation-type k-fold --models checkpoints/fold0.ckpt checkpoints/fold1.ckpt checkpoints/fold2.ckpt checkpoints/fold3.ckpt checkpoints/fold4.ckpt checkpoints/fold5.ckpt checkpoints/fold6.ckpt checkpoints/fold7.ckpt 
```


## Citation

```bibtex
@inproceedings{foscarin2024beatthis,
    author = {Francesco Foscarin and Jan Schl{\"u}ter and Gerhard Widmer},
    title = {Beat this! Accurate beat tracking without DBN postprocessing}
    year = 2024,
    month = nov,
    booktitle = {Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR)},
	address = {San Francisco, CA, United States},
}
```
