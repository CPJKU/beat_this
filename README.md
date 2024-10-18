# Beat This!
Official implementation of the beat tracker from the ISMIR 2024 paper "[Beat This! Accurate Beat Tracking Without DBN Postprocessing](https://arxiv.org/abs/2407.21658)" by Francesco Foscarin, Jan Schlüter and Gerhard Widmer.

* [Inference](#inference)
* [Available models](#available-models)
* [Data](#data)
* [Reproducing metrics from the paper](#reproducing-metrics-from-the-paper)
* [Training](#training)
* [Reusing the loss](#reusing-the-loss)
* [Reusing the model](#reusing-the-model)
* [Citation](#citation)


## Inference

To predict beats for audio files, you can either use our command line tool or call the beat tracker from Python. Both have the same requirements unless you go for the online demo.

### Online demo

To process a small set of audio files without installing anything, [open our example notebook in Google Colab](https://colab.research.google.com/github/CPJKU/beat_this/blob/main/beat_this_example.ipynb) and follow the instructions.

### Requirements

The beat tracker requires Python with a set of packages installed:
1. [Install PyTorch](https://pytorch.org/get-started/locally/) 2.0 or later following the instructions for your platform.
2. Install further modules with `pip install tqdm einops soxr rotary-embedding-torch`. (If using conda, we still recommend pip. You may try installing `soxr-python` and `einops` from conda-forge, but `rotary-embedding-torch` is only on PyPI.)
3. To read other audio formats than `.wav`, install `ffmpeg` or another supported backend for `torchaudio`. (`ffmpeg` can be installed via conda or via your operating system.)

Finally, install our beat tracker with:
```bash
pip install https://github.com/CPJKU/beat_this/archive/main.zip
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
The beat tracker will use the first GPU in your system by default, and fall back to CPU if PyTorch does not have CUDA access. With `--gpu=2`, it will use the third GPU, and with `--gpu=-1` it will force the CPU. For recent GPUs, passing `--float16` may improve speed.
If you have a lot of files to process, you can distribute the load over multiple processes by running the same command multiple times with `--touch-first`, `--skip-existing` and potentially different options for `--gpu`:
```bash
for gpu in {0..3}; do beat_this input_dir -o output_dir --touch-first --skip-existing --gpu=$gpu & done
```
If you want to use the DBN for postprocessing, add `--dbn`. The DBN parameters are the default ones from madmom. This requires installing the `madmom` package.

### Python class

If you are a Python user, you can directly use the `beat_this.inference` module.

First, instantiate an instance of the `File2Beats` class that encapsulates the model along with pre- and postprocessing:
```python
from beat_this.inference import File2Beats
file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
```
To obtain a list of beats and downbeats for an audio file, run:
```python
audio_path = "path/to/audio.file"
beats, downbeats = file2beats(audio_path)
```
Optionally, you can produce a `.beats` file (e.g., for importing into [Sonic Visualizer](https://www.sonicvisualiser.org/)):
```python
from beat_this.utils import save_beat_tsv
outpath = "path/to/output.beats"
save_beat_tsv(beats, downbeats, outpath)
```
If you already have an audio tensor loaded, instead of `File2Beats`, use `Audio2Beats` and pass the tensor and its sample rate. We also provide `Audio2Frames` for framewise logits and `Spect2Frames` for spectrogram inputs.


## Available models

Models are available for manual download at [our cloud space](https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp), but will also be downloaded automatically by the above inference code. By default, the inference will use `final0`, but it is possible to select another model via a command line option (`--model`) or Python parameter (`checkpoint_path`).

Main models:
* `final0`, `final1`, `final2`: Our main model, trained on all data except the GTZAN dataset, with three different seeds. This corresponds to "Our system" in Table 2 of the paper. About 78 MB per model.
* `small0`, `small1`, `small2`: A smaller model, again trained on all data except GTZAN, with three different seeds. This corresponds to "smaller model" in Table 2 of the paper. About 8.1 MB per model.
* `single_final0`, `single_final1`, `single_final2`: Our main model, trained on the single split described in Section 4.1 of the paper, with three different seeds. This corresponds to "Our system" in Table 3 of the paper. About 78 MB per model.
* `fold0`, `fold1`, `fold2`, `fold3`, `fold4`, `fold5`, `fold6`, `fold7`: Our main model, trained in the 8-fold cross-validation setting with a single seed per fold. This corresponds to "Our" in Table 1 of the paper. About 78 MB per model.

Other models, available mainly for result reproducibility:
* `hung0`, `hung1`, `hung2`: A model trained on all the data used by the "Modeling Beats and Downbeats with a Time-Frequency Transformer" system by Hung et al. (except GTZAN dataset), with three different seeds. This corresponds to "limited to data of [10]" in Table 2 of the paper.
* the other models used for the ablation studies in Table 3, all trained with 3 seeds on the single split described in Section 4.1 of the paper:
    * `single_notempoaug0`, `single_notempoaug1`, `single_notempoaug2`
    * `single_nosumhead0`, `single_nosumhead1`, `single_nosumhead2`
    * `single_nomaskaug0`, `single_nomaskaug1`, `single_nomaskaug2`
    * `single_nopartialt0`, `single_nopartialt1`, `single_nopartialt2`
    * `single_noshifttol0`, `single_noshifttol1`, `single_noshifttol2`
    * `single_nopitchaug0`, `single_nopitchaug1`, `single_nopitchaug2`
    * `single_noshifttolnoweights0`, `single_noshifttolnoweights1`, `single_noshifttolnoweights0`


Please be aware that the results may be unfairly good if you run inference on any file from the training datasets. For example, an evaluation with `final*` or `small*` can only be performed fairly on GTZAN or other datasets we didn't consider in our paper.

If you need to run an evaluation on some datasets we used other than GTZAN, consider targeting the validation part of the single split (with `single_final*`), or of the 8-fold cross-validation (with `fold*`).

All the models are provided as PyTorch Lightning checkpoints, stripped of the optimizer state to reduce their size. This is useful for reproducing the paper results or verifying the hyperparameters (stored in the checkpoint under `hyper_parameters` and `datamodule_hyper_parameters`).
During inference, PyTorch Lighting is not used, and the checkpoints are converted and loaded into vanilla PyTorch modules.

## Data

### Annotations
All annotations we used to train our models are available [in a separate GitHub repo](https://github.com/CPJKU/beat_this_annotations). Note that if you want to obtain the exact paper results, you should use [version 1.0](https://github.com/CPJKU/beat_this_annotations/releases/tag/v1.0). Other releases with corrected annotations may be published in the future.

To use the annotations for training or evaluation, you first need to download and extract or clone the annotations repo to `data/annotations`:
```bash
mkdir -p data
git clone https://github.com/CPJKU/beat_this_annotations data/annotations
# cd data/annotations; git checkout v1.0  # optional
```
### Spectrograms
The spectrograms used for training are released [as a Zenodo dataset](https://zenodo.org/records/13922116). They are distributed as a separate .zip file per dataset, each holding a .npz file with the spectrograms. For evaluation of the test set, download `gtzan.zip`; for training and evaluation of the validation set, download all (except `beat_this_annotations.zip`). Extract all .zip files into `data/audio/spectrograms`, so that you have, for example, `data/audio/spectrograms/gtzan.npz`. As an alternative, the code also supports directories of .npy files such as `data/audio/spectrograms/gtzan/gtzan_blues_00000/track.npy`, which you can obtain by unzipping `gtzan.npz`.

### Recreating spectrograms
If you have access to the original audio files, or want to add another dataset, create a text file `data/audio_paths.tsv` that has, on each line, the name of a dataset, a tab character, and the path to the audio directory. The corresponding annotations must also be present under `data/annotations`. Install pandas and pedalboard:
```bash
pip install pandas pedalboard
```
Then run:
```bash
python launch_scripts/preprocess_audio.py
```
It will create monophonic 22 kHz wave files in `data/audio/mono_tracks`, convert those to spectrograms in `data/audio/spectrograms`, and create spectrogram bundles. Intermediary files are kept and will not be recreated when rerunning the script.


## Reproducing metrics from the paper

### Requirements

In addition to the [inference requirements](#requirements), computing evaluation metrics requires installing PyTorch Lightning, Pandas, and `mir_eval`.
```bash
pip install pytorch_lightning pandas mir_eval
```
You must also obtain and set up the annotations and spectrogram datasets [as indicated above](#data). Specifically, the GTZAN dataset suffices for commands that include `--data split test`, while all other datasets are required for commands that include `--data split val`.


### Command line

#### Compute results on the test set (GTZAN) corresponding to Table 2 in the paper.

Main results for our system:
```bash
python launch_scripts/compute_paper_metrics.py --models final0 final1 final2 --datasplit test
```

Smaller model:
```bash
python launch_scripts/compute_paper_metrics.py --models small0 small1 small2 --datasplit test
```

Hung data:
```bash
python launch_scripts/compute_paper_metrics.py --models hung0 hung1 hung2 --datasplit test
```

With DBN (this requires installing the madmom package):
```bash
python launch_scripts/compute_paper_metrics.py --models final0 final1 final2 --datasplit test --dbn
```

#### Compute 8-fold cross-validation results, corresponding to Table 1 in the paper.

```bash
python launch_scripts/compute_paper_metrics.py --models fold0  fold1 fold2 fold3 fold4 fold5 fold6 fold7 --datasplit val --aggregation-type k-fold
```

#### Compute ablation studies on the validation set of the single split, correponding to Table 3 in the paper.

Our system:
```bash
python launch_scripts/compute_paper_metrics.py --models single_final0 single_final1 single_final2 --datasplit val
```

No sum head:
```bash
python launch_scripts/compute_paper_metrics.py --models single_nosumhead0 single_nosumhead1 single_nosumhead2 --datasplit val
```

No tempo augmentation:
```bash
python launch_scripts/compute_paper_metrics.py --models single_notempoaug0 single_notempoaug1 single_notempoaug2 --datasplit val
```

No mask augmentation:
```bash
python launch_scripts/compute_paper_metrics.py --models single_nomaskaug0 single_nomaskaug1 single_nomaskaug2 --datasplit val
```

No partial transformers:
```bash
python launch_scripts/compute_paper_metrics.py --models single_nopartialt0 single_nopartialt1 single_nopartialt2 --datasplit val
```

No shift tolerance:
```bash
python launch_scripts/compute_paper_metrics.py --models single_noshifttol0 single_noshifttol1 single_noshifttol2 --datasplit val
```

No pitch augmentation:
```bash
python launch_scripts/compute_paper_metrics.py --models single_nopitchaug0 single_nopitchaug1 single_nopitchaug2 --datasplit val
```

No shift tolerance and no weights:
```bash
python launch_scripts/compute_paper_metrics.py --models single_noshifttolnoweights0 single_noshifttolnoweights1 single_noshifttolnoweights2  --datasplit val
```

## Training

### Requirements

The training requirements match the [evaluation requirements](#requirements-1) for the validation set. All 16 datasets and annotations must be [correctly set up](#data).

### Command line

#### Train models listed in Table 2 in the paper.

Main results for our system (final0, final1, final2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --no-val
done
```

Smaller model (small0, small1, small2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --no-val --transformer-dim=128
done
```

Hung data (hung0, hung1, hung2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --no-val --hung-data
done
```

#### Train models with 8-fold cross-validation, corresponding to Table 1 in the paper.

```bash
for fold in {0..7}; do
    python launch_scripts/train.py --fold=$fold
done
```

#### Train models for the ablation studies, corresponding to Table 3 in the paper.

Our system (single_final0, single_final1, single_final2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed
done
```

No sum head (single_nosumhead0, single_nosumhead1, single_nosumhead2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --no-sum-head
done
```

No tempo augmentation (single_notempoaug0, single_notempoaug1, single_notempoaug2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --no-tempo-augmentation
done
```

No mask augmentation (single_nomaskaug0, single_nomaskaug1, single_nomaskaug2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --no-mask-augmentation
done
```

No partial transformers (single_nopartialt0, single_nopartialt1, single_nopartialt2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --no-partial-transformers
done
```

No shift tolerance (single_noshifttol0, single_noshifttol1, single_noshifttol2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --loss weighted_bce
done
```

No pitch augmentation (single_nopitchaug0, single_nopitchaug1, single_nopitchaug2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --no-pitch-augmentation
done
```

No shift tolerance and no weights (single_noshifttolnoweights0, single_noshifttolnoweights1, single_noshifttolnoweights2):
```bash
for seed in 0 1 2; do
    python launch_scripts/train.py --seed=$seed --loss bce
done
```


## Reusing the loss

To reuse our shift-invariant binary cross-entropy loss, just copy out the `ShiftTolerantBCELoss` class from [`loss.py`](beat_this/model/loss.py), it does not have any dependencies.


## Reusing the model

To reuse the BeatThis model, you have multiple options:

### From the package

When installing the `beat_this` package, you can directly import the model class:
```
from beat_this.model.beat_tracker import BeatThis
```
Instantiating this class will give you an untrained model from spectrograms to frame-wise beat and downbeat logits. For a pretrained model, use `load_model`:
```
from beat_this.inference import load_model
beat_this = load_model('final0', device='cuda')
```
### From torch.hub

To quickly try the model without installing the package, just install the [requirements for inference](#requirements) and do:
```
import torch
beat_this = torch.hub.load('CPJKU/beat_this', 'beat_this', 'final0', device='cuda')
```
### Copy and paste

To copy the BeatThis model into your own project, you will need the [`beat_tracker.py`](beat_this/model/beat_tracker.py) and [`roformer.py`](beat/this/model/roformer.py) files. If you remove the `BeatThis.state_dict()` and `BeatThis._load_from_state_dict()` methods that serve as a workaround for compiled models, then there are no other internal dependencies, only external dependencies (`einops`, `rotary-embedding-torch`).


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
