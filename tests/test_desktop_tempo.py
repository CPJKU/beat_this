import math
from pathlib import Path
import tempfile
import unittest

from beat_this.tempo_analysis import TempoSettings, analyze_tempo, load_beats, export_curve_csv, export_changes_csv


def synthetic_beats(segments):
    """Exact inverse of integrated linear BPM segments (duration, first BPM, last BPM)."""
    beats, offset, phase = [0.0], 0.0, 0.0
    next_beat = 1
    for duration, first, last in segments:
        slope = (last - first) / duration
        total_phase = duration * (first + last) / 120
        while next_beat <= phase + total_phase + 1e-9:
            target = (next_beat - phase) * 60
            t = target / first if abs(slope) < 1e-10 else 2 * target / (first + math.sqrt(first**2 + 2 * slope * target))
            beats.append(offset + t)
            next_beat += 1
        offset += duration
        phase += total_phase
    return beats


class TempoTests(unittest.TestCase):
    def test_constant_with_jitter_missing_and_extra_beats_has_no_changes(self):
        beats = [i * .5 + (.004 * math.sin(i * 1.37) if i else 0) for i in range(181)]
        beats.pop(40)
        beats.insert(80, (beats[79] + beats[80]) / 2)
        result = analyze_tempo(beats)
        self.assertEqual(result.changes, [])
        valid = [p.bpm for p in result.points if p.bpm is not None]
        self.assertTrue(all(abs(bpm - 120) < 1.5 for bpm in valid))

    def test_sustained_step_is_one_abrupt_change_near_ground_truth(self):
        result = analyze_tempo(synthetic_beats([(30, 120, 120), (35, 150, 150)]))
        self.assertEqual(len(result.changes), 1)
        change = result.changes[0]
        self.assertEqual(change.kind, 'abrupt')
        self.assertAlmostEqual(change.start, 30, delta=2.5)
        self.assertAlmostEqual(change.before, 120, delta=1)
        self.assertAlmostEqual(change.after, 150, delta=1)

    def test_ramp_up_is_one_acceleration_with_start_and_end(self):
        result = analyze_tempo(synthetic_beats([(20, 120, 120), (30, 120, 150), (20, 150, 150)]))
        self.assertEqual(len(result.changes), 1)
        change = result.changes[0]
        self.assertEqual(change.kind, 'acceleration')
        self.assertAlmostEqual(change.start, 20, delta=4)
        self.assertAlmostEqual(change.end, 50, delta=4)
        self.assertAlmostEqual(change.after - change.before, 30, delta=2)

    def test_step_then_ramp_down_remain_distinct(self):
        result = analyze_tempo(synthetic_beats([(20, 100, 100), (20, 130, 130), (25, 130, 90), (15, 90, 90)]))
        self.assertEqual([c.kind for c in result.changes], ['abrupt', 'deceleration'])

    def test_brief_tempo_excursion_does_not_pass_confirmation(self):
        for duration in (1, 1.5, 2, 2.5, 3.5):
            with self.subTest(duration=duration):
                result = analyze_tempo(synthetic_beats([(25, 120, 120), (duration, 150, 150), (25, 120, 120)]))
                self.assertEqual(result.changes, [])

    def test_unconfirmed_short_change_at_end_is_not_reported(self):
        beats = synthetic_beats([(30, 120, 120), (2, 150, 150)])
        result = analyze_tempo(beats, TempoSettings(window=2, confirmation=4))
        self.assertEqual(result.changes, [])

    def test_whole_track_ramp_can_be_reported_without_external_plateaus(self):
        result = analyze_tempo(synthetic_beats([(60, 100, 160)]))
        self.assertEqual(len(result.changes), 1)
        self.assertEqual(result.changes[0].kind, 'acceleration')

    def test_sensitivity_reveals_a_small_sustained_change(self):
        beats = synthetic_beats([(25, 120, 120), (25, 122.5, 122.5)])
        self.assertEqual(analyze_tempo(beats).changes, [])
        sensitive = TempoSettings(window=4, min_bpm=1, min_percent=.5, confirmation=4)
        self.assertEqual(len(analyze_tempo(beats, sensitive).changes), 1)

    def test_silence_creates_a_gap_and_does_not_invent_a_transition(self):
        beats = [i * .5 for i in range(41)] + [40 + i * .4 for i in range(51)]
        result = analyze_tempo(beats, duration=65)
        self.assertTrue(any(p.bpm is None and 25 < p.time < 35 for p in result.points))
        self.assertEqual(result.changes, [])
        self.assertAlmostEqual(result.duration, 65)

    def test_short_and_invalid_input_are_handled(self):
        self.assertEqual(analyze_tempo([]).changes, [])
        self.assertEqual(analyze_tempo([0, .5]).changes, [])
        for beats in ([0, float('nan'), 1], [0, 2, 1], [-1, 0, 1]):
            with self.assertRaises(ValueError):
                analyze_tempo(beats)
        with self.assertRaises(ValueError):
            analyze_tempo([0, 1, 2], TempoSettings(window=0))

    def test_existing_beats_import_and_exports_preserve_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            beats = synthetic_beats([(20, 120, 120), (20, 150, 150)])
            path = root / 'שיר.beats'
            path.write_text(''.join(f'{t}\t1\n' for t in beats), encoding='utf-8')
            result = analyze_tempo(load_beats(path))
            export_curve_csv(result, root / 'curve.csv')
            export_changes_csv(result, root / 'changes.csv')
            self.assertIn('time_seconds', (root / 'curve.csv').read_text(encoding='utf-8-sig'))
            self.assertIn('abrupt', (root / 'changes.csv').read_text(encoding='utf-8-sig'))
            self.assertAlmostEqual(result.changes[0].start, 20, delta=2.5)


if __name__ == '__main__':
    unittest.main()
