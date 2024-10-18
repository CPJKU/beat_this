dependencies = [
    "torch",
    "torchaudio",
    "numpy",
    "rotary_embedding_torch",
    "einops",
    "soxr",
]

from beat_this.inference import (
    load_model as beat_this,
    BeatThis,
    Spect2Frames,
    Audio2Frames,
    Audio2Beats,
    File2Beats,
    File2File,
)
