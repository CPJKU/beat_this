from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import librosa
import madmom
import numpy as np
import os
import pandas as pd
import argparse
from pedalboard import time_stretch, Pedalboard, PitchShift
import concurrent.futures
from git import Repo
import wave
import torchaudio
import torch
from beat_this.utils import get_spect_len, filename_to_augmentation
from beat_this.dataset.augment import precomputed_augmentation_filenames


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_audio(path):
    try:
        return sf.read(path, dtype='float64')
    except Exception:
        # some files are not readable by soundfile, try madmom
        return madmom.io.load_audio_file(str(path), dtype="float64")


def save_audio(path, waveform, samplerate, resample_from=None):
    if resample_from and resample_from != samplerate:
        waveform = librosa.resample(waveform,
                                    orig_sr=resample_from,
                                    target_sr=samplerate)
    try:
        waveform = np.asarray(waveform, dtype=np.float64)
        sf.write(path, waveform, samplerate=samplerate)
    except KeyboardInterrupt:
        path.unlink()  # avoid half-written files
        raise


def save_spectrogram(path, spectrogram):
    try:
        np.save(path, spectrogram)
    except KeyboardInterrupt:
        path.unlink()  # avoid half-written files
        raise


def get_wav_length(filename):
    """Returns length of a wav file in samples."""
    with open(filename, 'rb') as f:
        return wave.open(f, 'r').getnframes()


class SpectCreation():
    def __init__(self, pitch_shift, time_stretch, audio_sr, mel_args):
        super(SpectCreation, self).__init__()
        # define the directories
        self.preprocessed_dir = BASEPATH / 'data' / 'preprocessed'
        self.mono_tracks_dir = self.preprocessed_dir / 'mono_tracks'
        self.spectrograms_dir = self.preprocessed_dir / 'spectrograms'
        self.annotations_dir = BASEPATH / 'data' / 'beat_annotations'
        # remember the audio metadata
        self.audio_sr = audio_sr
        # create the mel spectrogram class
        self.spect_class = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_sr, **mel_args)
        self.augmentations = {"pitch": {"min": pitch_shift[0], "max": pitch_shift[1]}, "tempo": { "min": -time_stretch[0], "max": time_stretch[0]+1, "stride": time_stretch[1]}}
        # compute the names to consider according to the augmentations
        self.filenames = precomputed_augmentation_filenames(self.augmentations, "wav")

    def create_spects(self):
        print("Creating spectrograms ...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for dataset_dir in self.mono_tracks_dir.iterdir():
                for piece_dir in dataset_dir.iterdir():
                    futures.append(executor.submit(self.create_spect_piece,
                                                    piece_dir,
                                                    Path(dataset_dir.name) / 'annotations' / 'beats' / f'{piece_dir.name}.beats',
                                                    dataset_dir.name))
            metadata = [future.result()
                        for future in tqdm(concurrent.futures.as_completed(futures),
                                           total=len(futures))]

        # save metadata
        df = pd.DataFrame.from_dict([m[0] for m in metadata if m is not None]).sort_values(by="spect_folder")
        df.to_csv(self.spectrograms_dir /
                  'spectrograms_metadata.csv', index=False)
        print(f"Created {len(df)} spectrograms in {self.spectrograms_dir}")

    def create_spect_piece(self, preprocessed_audio_folder, beat_path, dataset_name):
        metadata = []
        spect_lens = {} # store the length of the spectrograms for each tempo augmentation. The key is the stretch.
        for filename in self.filenames:
            if not (self.annotations_dir / beat_path).exists():
                print(f"beat annotation {beat_path} not found for {preprocessed_audio_folder}")
                return
            audio_path = preprocessed_audio_folder / filename
            spect_path = self.spectrograms_dir / dataset_name/ preprocessed_audio_folder.name / f'{Path(filename).stem}.npy'
            if spect_path.exists():
                # load the spectrogram to get the length
                try:
                    spect = np.load(spect_path, mmap_mode='r')
                    compute_spect = False
                except:
                    compute_spect = True
            else:
                compute_spect = True
            if compute_spect:
                waveform, sr = load_audio(audio_path)
                assert sr == self.audio_sr, f"Sample rate mismatch: {sr} != {self.audio_sr}"
                # compute the mel spectrogram
                spect = self.spect_class(torch.tensor(
                    waveform, dtype=torch.float32).unsqueeze(0)).squeeze(0).T
                # scale the values with log(1 + 1000 * x)
                spect = torch.log1p(1000*spect)
                # save the spectrogram as numpy array
                spect_path.parent.mkdir(parents=True, exist_ok=True)
                save_spectrogram(spect_path, spect.numpy())
                
            # save the length of the spectrogram
            spect_lens[filename_to_augmentation(audio_path.stem)["stretch"]] = spect.shape[0]
        # save the metadata. Each tempo augmentation get a dedicated column
        metadata.append({"spect_folder": spect_path.parent.relative_to(self.spectrograms_dir),
                         "beat_path": beat_path,
                         "dataset": dataset_name,
                            **{f"spect_len_ts{ts}": spect_lens.get(ts, 0) for ts in sorted(spect_lens.keys())}})
        return metadata


class AudioPreprocessing(object):
    def __init__(self, orig_audio_paths, out_sr=22050, aug_sr=44100, ext='wav', pitch_shift=(-5, 6), time_stretch=(20, 4), verbose=False):
        """
        Class for converting audio files to mono, resampling, and applying augmentations.
        Only use this if you want to start from new audio files, otherwise use the spectrograms provided in the repo.

        Args:
            data_dir (Path): path to the data directory
            out_sr (int): output sample rate
            aug_sr (int): sample rate for computing the augmentations, this needs to be high enough (e.g., 44100) to not create problems during pitch shifting
            ext (str): extension of the audio files
            pitch_shift (tuple): lowest and highest (included) pitch shift in semitones (e.g., -5, 6).
            time_stretch (tuple): maximum percentage and stride for time stretching
            verbose (bool): verbose output
        """
        super(AudioPreprocessing, self).__init__()
        self.preprocessed_dir = BASEPATH / 'data' / 'preprocessed'
        self.annotation_dir = BASEPATH / 'data' / 'beat_annotations'
        # load data_dir from audio_path.csv which has the format: dataset_name, audio_path
        self.audio_dirs = {row[0]: row[1] for row in pd.read_csv(
            orig_audio_paths, header=None).values}
        # check if annotations exists, otherwise clone them from the repo
        if not self.annotation_dir.exists():
            Repo.clone_from(
                "https://github.com/fosfrancesco/beat_annotations.git", self.annotation_dir)
            assert self.annotation_dir.exists(
            ), "Annotations not found, something wrong during cloning"

        print("Annotations ready in data/beat_annotations")

        self.out_sr = out_sr
        self.aug_sr = aug_sr
        self.ext = ext
        self.pitch_shift = pitch_shift
        if time_stretch:
            # interpret tuple as (maximum percentage, stride)
            time_stretch = range(-time_stretch[0], time_stretch[0] + 1,
                                 time_stretch[1] if len(time_stretch) > 1 else 1)
        self.time_stretch = time_stretch
        self.verbose = verbose

    def preprocess_audio(self):
        print("Preprocessing audio files ...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for dataset_name, audio_dir in self.audio_dirs.items():
                for audio_path in Path(audio_dir).iterdir():
                    if audio_path.stem[:12] in ('gtzan_speech', 'gtzan_music_'):
                        continue
                    futures.append(executor.submit(self.process_audio_file,
                                                   dataset_name,
                                                   audio_path))
            metadata = [future.result()
                        for future in tqdm(concurrent.futures.as_completed(futures),
                                           total=len(futures))]
        print("Done!")

        # merge all metadata in a pandas dataframe
        df = pd.DataFrame.from_dict(sorted([m for m in metadata if m is not None],
                                           key=lambda x: x['processed_path']))
        print("Processed", len(df), "audio files")
        return df

    def process_audio_file(self, dataset_name, audio_path):
        annotation_dir = Path(self.annotation_dir, dataset_name, 'annotations')
        # load annotations
        beat_path = Path(annotation_dir, "beats", audio_path.stem+'.beats')
        if not beat_path.exists():
            print(f"beat annotation {beat_path} not found for {audio_path}", )
            return
        # create a folder with the name of the track
        folder_path = Path(self.preprocessed_dir, "mono_tracks",
                           dataset_name, audio_path.stem)
        # derive the name of the unaugmented file
        mono_path = folder_path / f'track_ps0.{self.ext}'
        # derive the name of all augmented files
        augmentations = {"pitch": {"min": self.pitch_shift[0], "max": self.pitch_shift[1]}, "tempo": { "min": -self.time_stretch[0], "max": self.time_stretch[0]+1, "stride": self.time_stretch[1]}}
        augmentations_path = precomputed_augmentation_filenames(augmentations, self.ext)
        # stop here if all files exists
        if mono_path.exists() and all((folder_path / aug).exists() for aug in augmentations_path):
            if self.verbose:
                print(f"All files in {folder_path} exists, skipping")
            return

        # load audio
        try:
            waveform, sr = load_audio(audio_path)
        except Exception as e:
            print("Problem with loading waveform", audio_path, e)
            return
        folder_path.mkdir(parents=True, exist_ok=True)
        if waveform.ndim == 1 and sr == self.out_sr and audio_path.suffix == f'.{self.ext}':
            # shortcut: copy original file to mono path location
            os.system("cp '{}' '{}'".format(audio_path, mono_path))
        else:
            # we need to do some conversions for the unaugmented file
            if waveform.ndim != 1:
                waveform = np.mean(waveform, axis=1)
            if not mono_path.exists():
                if sr != self.out_sr:
                    waveform_out = librosa.resample(waveform, orig_sr=sr,
                                                    target_sr=self.out_sr)
                else:
                    waveform_out = waveform
                # save mono file
                save_audio(mono_path, waveform_out, self.out_sr)
        if (self.pitch_shift or self.time_stretch) and (sr != self.aug_sr):
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=self.aug_sr)
            
        # handle the requested augmentations          
        # pedalboard requires float32, convert
        waveform = np.asarray(waveform, dtype=np.float32)
        shifts = range(
            self.pitch_shift[0], self.pitch_shift[1]+1) if self.pitch_shift else [0]
        stretches = self.time_stretch if self.time_stretch else [0]
        for shift in shifts:  # pitch augmentation
            augment_audio_file(
                folder_path, waveform, aug_type = "shift", amount = shift, aug_sr = self.aug_sr, out_sr = self.out_sr, ext = self.ext, verbose = self.verbose)
        for stretch in stretches:  # tempo augmentation
            augment_audio_file(
                folder_path, waveform, aug_type = "stretch", amount = stretch, aug_sr = self.aug_sr, out_sr = self.out_sr, ext = self.ext, verbose=self.verbose)

        return

    
def augment_audio_file(folder_path, waveform, aug_type, amount, aug_sr, out_sr, ext, verbose):
    # figure out the file name
    if aug_type == "stretch":
        stretch = amount
        shift = 0
    elif aug_type == "shift":
        shift = amount
        stretch = 0
    else:
        raise ValueError(f"Unknown augmentation mode {aug_type}")
    suffix = f"_ps{shift}"
    if stretch != 0:
        suffix = suffix + f"_ts{stretch}"
    out_path = Path(folder_path, f'track{suffix}.{ext}')
    # skip if it exists
    if out_path.exists():
        if verbose:
            print(f"{out_path} exists, skipping")
        return
    # otherwise compute it and write it out
    # time stretch or pitch shift alone
    if aug_type == "shift":
        if verbose: print(f"computing {out_path} with {shift=}")
        # pitch shift alone
        board = Pedalboard([
            PitchShift(semitones=shift),
        ])
        # apply pedalboard
        augmented = board(waveform, aug_sr)
    else: # type == stretch
        if verbose:
            print(
                f"computing {out_path} with {stretch=}")
        augmented = time_stretch(waveform, aug_sr,
                                stretch_factor=1 + stretch / 100,
                                pitch_shift_in_semitones=0.0,
                                ).squeeze()
    # save to file
    if verbose:
        print(f"writing {out_path}")
    save_audio(out_path, augmented, out_sr,
            resample_from=aug_sr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_audio_paths", type=str,
                        help="path to the file with the original audio paths for each dataset", default='data/audio_paths.csv')
    parser.add_argument(
        "--pitch_shift", metavar="LOW:HIGH",
        type=str, default="-5:6", help="pitch shift in semitones (default: %(default)s)")
    parser.add_argument("--time_stretch", metavar="MAX:STRIDE",
                        type=str, default="20:4", help="time stretch in percentage and stride (default: %(default)s)")
    parser.add_argument("--verbose", action='store_true',
                        help="verbose output")
    args = parser.parse_args()

    # set the base path
    global BASEPATH
    BASEPATH = Path(__file__).parent.parent.parent.relative_to(Path.cwd())
    # check if the orig_audio_paths exists, supporting both relative and absolute paths
    if Path(args.orig_audio_paths).exists():
        orig_audio_paths = args.orig_audio_paths
    elif Path(BASEPATH, args.orig_audio_paths).exists():
        orig_audio_paths = Path(BASEPATH, args.orig_audio_paths)
    else:
        raise FileNotFoundError(f"File {args.orig_audio_paths} not found")

    # preprocess audio
    dp = AudioPreprocessing(orig_audio_paths=orig_audio_paths, out_sr=22050, aug_sr=44100,
                            pitch_shift=args.pitch_shift and tuple(
                                map(int, args.pitch_shift.split(':'))),
                            time_stretch=args.time_stretch and tuple(map(int, args.time_stretch.split(':'))), verbose=args.verbose)
    dp.preprocess_audio()

    # compute spectrograms
    mel_args = dict(n_fft=1024, hop_length=441, f_min=30, f_max=11000,
                    n_mels=128, mel_scale='slaney', normalized='frame_length', power=1)
    audio_folder = Path(BASEPATH, 'data', 'preprocessed', 'mono_tracks')
    sc = SpectCreation(pitch_shift=args.pitch_shift and tuple(map(int, args.pitch_shift.split(':'))),
                        time_stretch=args.time_stretch and tuple(map(int, args.time_stretch.split(':'))), 
                            audio_sr=22050, mel_args=mel_args)
    sc.create_spects()
