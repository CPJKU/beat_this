
import argparse
import os
from pathlib import Path
from beat_this.preprocessing import AudioPreprocessing, SpectCreation

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
