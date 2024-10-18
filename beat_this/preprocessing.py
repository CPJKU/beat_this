import numpy as np
import torch
import torchaudio


def load_audio(path, dtype="float64"):
    try:
        waveform, samplerate = torchaudio.load(path, channels_first=False)
        waveform = np.asanyarray(waveform.squeeze().numpy(), dtype=dtype)
        return waveform, samplerate
    except Exception:
        # in case torchaudio fails, try soundfile
        try:
            import soundfile as sf

            return sf.read(path, dtype=dtype)
        except Exception:
            # some files are not readable by soundfile, try madmom
            try:
                import madmom

                return madmom.io.load_audio_file(str(path), dtype=dtype)
            except Exception:
                raise RuntimeError(f'Could not load audio from "{path}".')


class LogMelSpect(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=441,
        f_min=30,
        f_max=11000,
        n_mels=128,
        mel_scale="slaney",
        normalized="frame_length",
        power=1,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        self.spect_class = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=normalized,
            power=power,
        ).to(device)
        self.log_multiplier = log_multiplier

    def forward(self, x):
        """Input is a waveform as a monodimensional array of shape T,
        output is a 2D log mel spectrogram of shape (F,128)."""
        return torch.log1p(self.log_multiplier * self.spect_class(x).T)
