# Beat This
Accurate and general beat tracker from the paper "Beat This! Accurate Beat Tracking Without DBN Postprocessing".

## Available models
We release 3 main models, which were trained on all data, except the GTZAN dataset, with three different seeds. You can use them with the shortcut `final0`, `final1`, and `final2`. These correspond to "Our System" in Table 2 on the paper.
Please be aware that, as the models ```final*``` were trained on all data except the GTZAN dataset, the results may be unfairly good, if you run the inference on some data that was used for training.

The K-Fold models and single split from the paper are also available.

All the models are provided as pytorch lightning checkpoints, stripped of the optimizer state to reduce their size. This is useful for reproducing the paper results.
During inference, PyTorch lighting is not used, and the checkpoints are converted and loaded into vanilla PyTorch modules.

## Inference

### Python function and classes
If you are a Python user, you can call the available function from the `beat_this.inference` module.

For a single file, you can use the `audio2beat` function as follows:

```python
from beat_this.inference import audio2beat

# Path to your audio file
audio_path = "path_to_your_audio_file.wav"

# Model checkpoint to use (default is "final0")
model_checkpoint = "final0"

# Set to True if you want to use the DBN postprocessor
# The DBN parameters are the default ones
use_dbn = False

# Device to run the inference on
device = "cpu"

from beat_this.inference import Audio2Beat

audio2beat = Audio2Beat(model_checkpoint=model_checkpoint, device=device, dbn=use_dbn)
beat, downbeat = audio2beat(audio_path)

print("Beat positions (in seconds):", beat)
print("Downbeat positions (in seconds):", downbeat)
```

If you plan on converting multiple files, the `Audio2Beat` class is more efficient, because it only loads the model once:

```python
from beat_this.inference import Audio2Beat

a2b = Audio2Beat(model_checkpoint=model_checkpoint, device=device, dbn=use_dbn)
beat, downbeat = audio2beat(audio_path)
```

You can produce a `.beat` file, that can be imported into [Sonic Visualizer](https://www.sonicvisualiser.org/), with the command:
```python
from beat_this.utils import save_beat_csv

outpath = "path/to/output/file.beat"
save_beat_csv(beat, downbeat, outpath)
```

### Command line
A command line option is also available if you prefer launching from the terminal.
 
```sh
python launch_scripts/predict.py --model final0 --audio-path path/to/audio/file --output_path path/to/output/file
```

Set ```--gpu -1``` if you want to run on cpu.
You can add ```--dbn``` if you want to use the DBN. The DBN parameters are the default ones from madmom.



## Training
This part will be available soon.

## Reproducing metrics from the paper
Compute results on the test set GTZAN

```
python launch_scripts/compute_paper_metrics.py --models final0 final1 final2 --datasplit test
```

Compute k-fold
```
python launch_scripts/compute_paper_metrics.py --aggregation-type k-fold --models checkpoints/fold0.ckpt checkpoints/fold1.ckpt checkpoints/fold2.ckpt checkpoints/fold3.ckpt checkpoints/fold4.ckpt checkpoints/fold5.ckpt checkpoints/fold6.ckpt checkpoints/fold7.ckpt 
```
