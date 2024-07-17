#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import soxr
import numpy as np
import pandas as pd
import madmom
from pedalboard import time_stretch, Pedalboard, PitchShift
import concurrent.futures
from git import Repo
import torch
from beat_this.utils import filename_to_augmentation
from beat_this.dataset.augment import precomputed_augmentation_filenames

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

BASEPATH = Path(__file__).parent.parent.relative_to(Path.cwd())


def save_audio(path, waveform, samplerate, resample_from=None):
    if resample_from and resample_from != samplerate:
        waveform = soxr.resample(waveform,
                                 in_rate=resample_from,
                                 out_rate=samplerate)
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


class SpectCreation():
    def __init__(self, pitch_shift, time_stretch, audio_sr, mel_args, verbose=False):
        """
        Initialize the SpectCreation class. This assume that the audio files have been preprocessed with all the requested augmentations and are stored in the `mono_tracks` directory with the proper naming defined in AudioPreprocessing.

        Args:
            pitch_shift (tuple or None): A tuple specifying the minimum and maximum (inclusive) pitch shift values considered from the available audio files.
                                        If None, pitch shifting augmentation files will not be considered.
            time_stretch (tuple or None): A tuple specifying the min/max and stride percentage to consider from the available audio files.
                                        If None, time stretching augmentation files will not be considered.
            audio_sr (int): The sample rate of the audio.
            mel_args (dict): A dictionary of arguments to be passed to the MelSpectrogram class.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.
        """
        super(SpectCreation, self).__init__()
        # define the directories
        self.preprocessed_dir = BASEPATH / 'data' / 'preprocessed'
        self.mono_tracks_dir = self.preprocessed_dir / 'mono_tracks'
        self.spectrograms_dir = self.preprocessed_dir / 'spectrograms'
        self.annotations_dir = BASEPATH / 'data' / 'beat_annotations'

        if verbose:
            print("Preprocessed dir: ", self.preprocessed_dir.absolute())
            print("Mono tracks dir: ", self.mono_tracks_dir.absolute())
            print("Spectrograms dir: ", self.spectrograms_dir.absolute())
            print("Annotations dir: ", self.annotations_dir.absolute())
        self.verbose = verbose
        # remember the audio metadata
        self.audio_sr = audio_sr
        # create the mel spectrogram class
        self.logspect_class = LogMelSpect(audio_sr, **mel_args)
        # define the augmentations
        self.augmentations = {}
        if pitch_shift is not None:
            self.augmentations["pitch"] = {"min": pitch_shift[0], "max": pitch_shift[1]}
        if time_stretch is not None:
            self.augmentations["tempo"] = {"min": -time_stretch[0], "max": time_stretch[0], "stride": time_stretch[1]}
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
        """
        Create spectrogram for a single audio piece.

        This method creates a spectrogram for a single audio piece located in the `preprocessed_audio_folder`.
        The beat annotations for the audio piece are loaded from the `beat_path` file.
        The created spectrogram is saved in the `spectrograms_dir` directory.

        Args:
            preprocessed_audio_folder (Path): The path to the preprocessed audio folder.
            beat_path (Path): The path to the beat annotations file.
            dataset_name (str): The name of the dataset.

        Returns:
            metadata (list): A list containing the metadata of the created spectrogram.
        """
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
            if self.verbose:
                if compute_spect:
                    print(f"Computing {spect_path}")
                else:
                    print(f"Skipping {spect_path} because it exists")
            if compute_spect:
                waveform, sr = load_audio(audio_path)
                assert sr == self.audio_sr, f"Sample rate mismatch: {sr} != {self.audio_sr}"
                # compute the mel spectrogram and scale the values with log(1 + 1000 * x)
                spect = self.logspect_class(torch.tensor(
                    waveform, dtype=torch.float32))
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
            orig_audio_paths (Path): The path to the file with the original audio paths for each dataset.
            out_sr (int, optional): The output sample rate. Defaults to 22050.
            aug_sr (int, optional): The sample rate for the augmentations. Defaults to 44100.
            ext (str, optional): The extension of the audio files. Defaults to 'wav'.
            pitch_shift (tuple, optional): A tuple specifying the minimum and maximum (inclusive) pitch shift values considered. Defaults to (-5, 6).
            time_stretch (tuple, optional): A tuple specifying the min/max (inclusive) time stretch and stride in percentage considered. Defaults to (20, 4).
            verbose (bool, optional): Whether to print verbose information. Defaults to False.
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
                    waveform_out = soxr.resample(waveform, in_rate=sr,
                                                 out_rate=self.out_sr)
                else:
                    waveform_out = waveform
                # save mono file
                save_audio(mono_path, waveform_out, self.out_sr)
        if (self.pitch_shift or self.time_stretch) and (sr != self.aug_sr):
            waveform = soxr.resample(
                waveform, in_rate=sr, out_rate=self.aug_sr)
            
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


def main(orig_audio_paths, pitch_shift, time_stretch, verbose):
    # preprocess audio
    dp = AudioPreprocessing(orig_audio_paths=orig_audio_paths, out_sr=22050, aug_sr=44100,
                            pitch_shift=pitch_shift and tuple(
                                map(int, pitch_shift.split(':'))),
                            time_stretch=time_stretch and tuple(map(int, time_stretch.split(':'))), verbose=verbose)
    dp.preprocess_audio()

    # compute spectrograms
    mel_args = dict(n_fft=1024, hop_length=441, f_min=30, f_max=11000,
                    n_mels=128, mel_scale='slaney', normalized='frame_length', power=1)
    sc = SpectCreation(pitch_shift=pitch_shift and tuple(map(int, pitch_shift.split(':'))),
                        time_stretch=time_stretch and tuple(map(int, time_stretch.split(':'))), 
                            audio_sr=22050, mel_args=mel_args, verbose=verbose)
    sc.create_spects()


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

    main(args.orig_audio_paths, args.pitch_shift, args.time_stretch, args.verbose)
