import tempfile
import unittest
from unittest.mock import patch
import os
from pathlib import Path

from beat_this.desktop_core import Options, build_tasks, estimate_bpm, is_accelerator_error, commit_outputs


class DesktopCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.song = self.root / 'שיר עם רווח.wav'
        self.song.touch()

    def test_defaults_and_append(self):
        task = build_tasks([self.song], Options())[0]
        self.assertEqual(task.output, self.song.with_suffix('.beats'))
        task = build_tasks([self.song], Options(append=True))[0]
        self.assertEqual(task.output.name, self.song.name + '.beats')

    def test_folder_audio_only_preserves_tree_and_deduplicates(self):
        nested = self.root / 'album'
        nested.mkdir()
        (nested / 'song.mp3').touch()
        (nested / 'notes.txt').touch()
        output = self.root / 'out'
        output.mkdir()
        (output / 'old.wav').touch()
        tasks = build_tasks([self.root, self.song], Options(output=str(output)))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].output.parent, output / 'album')

    def test_explicit_single_file_and_skip(self):
        output = self.root / 'result.beats'
        options = Options(output=str(output), output_is_file=True, skip_existing=True)
        self.assertEqual(build_tasks([self.song], options)[0].output, output)
        output.touch()
        self.assertTrue(build_tasks([self.song], options)[0].skip)

    def test_rejects_overwrite_of_input_or_activation(self):
        with self.assertRaises(ValueError):
            build_tasks([self.song], Options(suffix='.wav'))
        with self.assertRaises(ValueError):
            build_tasks([self.song], Options(suffix='.npy', activations=True))

    def test_rejects_duplicate_output_names(self):
        other = self.song.with_suffix('.mp3')
        other.touch()
        with self.assertRaises(ValueError):
            build_tasks([self.song, other], Options())

    def test_rejects_bad_suffix_and_directory_as_file(self):
        for suffix in ('', 'beats', '../.beats'):
            with self.assertRaises(ValueError):
                build_tasks([self.song], Options(suffix=suffix))
        with self.assertRaises(ValueError):
            build_tasks([self.root], Options(output=str(self.root / 'out'), output_is_file=True))

    def test_bpm_robust_to_one_outlier_and_empty_input(self):
        self.assertEqual(estimate_bpm([0, .5, 1, 1.5, 3]), 120.0)
        self.assertIsNone(estimate_bpm([]))
        self.assertIsNone(estimate_bpm([1, 1]))

    def test_gpu_errors_and_wrapped_errors_trigger_fallback_only_on_gpu(self):
        for message in ('CUDA error: device kernel image is invalid',
                        'HIP error: invalid device function', 'HIP out of memory',
                        'cuDNN error: execution failed'):
            error = RuntimeError(message)
            self.assertTrue(is_accelerator_error(error, 'cuda:0'))
            self.assertFalse(is_accelerator_error(error, 'cpu'))
        wrapped = ValueError('Could not load the checkpoint')
        wrapped.__cause__ = RuntimeError('HIP error: no kernel image is available')
        self.assertTrue(is_accelerator_error(wrapped, 'cuda:0'))
        self.assertFalse(is_accelerator_error(ValueError('Could not load audio'), 'cuda:0'))

    def test_failed_beats_commit_restores_previous_activation(self):
        beat_temp = self.root / 'new-beats.tmp'
        raw_temp = self.root / 'new-raw.tmp'
        beat_output = self.root / 'result.beats'
        raw_output = self.root / 'result.npy'
        for path, content in [(beat_temp, b'new beats'), (raw_temp, b'new raw'),
                              (beat_output, b'old beats'), (raw_output, b'old raw')]:
            path.write_bytes(content)
        real_replace = os.replace
        def replace(source, target):
            if Path(target) == beat_output:
                raise PermissionError('Simulated locked beats output')
            return real_replace(source, target)
        with patch('beat_this.desktop_core.os.replace', side_effect=replace):
            with self.assertRaises(PermissionError):
                commit_outputs(beat_temp, beat_output, raw_temp, raw_output)
        self.assertEqual(beat_output.read_bytes(), b'old beats')
        self.assertEqual(raw_output.read_bytes(), b'old raw')

    def test_successful_commit_updates_both_outputs(self):
        beat_temp, raw_temp = self.root / 'beats.tmp', self.root / 'raw.tmp'
        beat_temp.write_bytes(b'beats')
        raw_temp.write_bytes(b'raw')
        beat_output, raw_output = self.root / 'out.beats', self.root / 'out.npy'
        commit_outputs(beat_temp, beat_output, raw_temp, raw_output)
        self.assertEqual(beat_output.read_bytes(), b'beats')
        self.assertEqual(raw_output.read_bytes(), b'raw')


if __name__ == '__main__':
    unittest.main()
