"""Tempo curves and estimated transitions derived from beat timestamps.

Independent of Qt, numpy and the inference runtime. Local tempo is a trimmed
mean of plausible beat intervals; a piecewise-linear curve identifies sustained
transitions. This describes the tracker output, not a ground-truth tempo map.
"""
from beat_this.desktop_i18n import tr
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class TempoSettings:
    window: float = 4.0
    min_bpm: float = 3.0
    min_percent: float = 3.0
    confirmation: float = 4.0

    def validate(self):
        for value in (self.window, self.min_bpm, self.min_percent, self.confirmation):
            if not math.isfinite(value):
                raise ValueError(tr('הגדרות הרגישות חייבות להיות מספרים סופיים.'))
        if self.window <= 0 or self.min_bpm <= 0 or self.min_percent < 0 or self.confirmation <= 0:
            raise ValueError(tr('חלון ההחלקה, סף ה־BPM וזמן האישור חייבים להיות חיוביים.'))


@dataclass(frozen=True)
class TempoPoint:
    time: float
    bpm: float | None


@dataclass(frozen=True)
class TempoChange:
    start: float
    end: float
    before: float
    after: float
    kind: str

    @property
    def delta(self):
        return self.after - self.before


@dataclass
class TempoAnalysis:
    points: list[TempoPoint] = field(default_factory=list)
    changes: list[TempoChange] = field(default_factory=list)
    duration: float = 0.0
    beat_count: int = 0
    settings: TempoSettings = field(default_factory=TempoSettings)


def clean_beats(beats):
    result = []
    for item in beats:
        value = float(item)
        if not math.isfinite(value) or value < 0:
            raise ValueError(tr('זמני הפעימות חייבים להיות מספרים תקינים ולא שליליים.'))
        if result and value < result[-1]:
            raise ValueError(tr('זמני הפעימות אינם מסודרים בסדר עולה.'))
        if not result or value > result[-1]:
            result.append(value)
    return result


def load_beats(path):
    beats = []
    with Path(path).open(encoding='utf-8-sig') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            try:
                beats.append(float(line.split()[0]))
            except (ValueError, IndexError) as exc:
                raise ValueError(tr('קובץ הפעימות אינו תקין בשורה {0}.', number)) from exc
    return clean_beats(beats)


def _curve(beats, settings, duration):
    if len(beats) < 3:
        return []
    intervals = [b - a for a, b in zip(beats, beats[1:])]
    centers = [(a + b) / 2 for a, b in zip(beats, beats[1:])]
    gap_limit = max(4.0, median(intervals) * 4)
    step = max(.5, duration / 12000)
    points = []
    # Sampling on an absolute time grid makes event times independent of beat count.
    count = max(1, int(math.ceil(duration / step)))
    for index in range(count + 1):
        time = min(duration, index * step)
        left = bisect_left(centers, time - settings.window / 2)
        right = bisect_right(centers, time + settings.window / 2)
        values = [x for x in intervals[left:right] if x <= gap_limit]
        value = None
        # No interpolation across silence, and no tempo estimate from one interval.
        beat_index = bisect_right(beats, time) - 1
        in_gap = (0 <= beat_index < len(intervals) and intervals[beat_index] > gap_limit)
        outside = time < beats[0] or time > beats[-1]
        if len(values) >= 2 and not in_gap and not outside:
            center = median(values)
            values = sorted(x for x in values if .55 * center <= x <= 1.65 * center)
            if len(values) >= 2:
                trim = int(len(values) * .15)
                core = values[trim:len(values) - trim] if trim else values
                value = 60 * len(core) / sum(core)
        points.append(TempoPoint(time, value))
    return points


def _simplify(points, epsilon):
    """Iterative piecewise-linear approximation using vertical BPM error."""
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        first, last = pending.pop()
        if last - first < 2:
            continue
        a, b = points[first], points[last]
        if b.time == a.time:
            continue
        slope = (b.bpm - a.bpm) / (b.time - a.time)
        error, location = 0.0, None
        for index in range(first + 1, last):
            predicted = a.bpm + slope * (points[index].time - a.time)
            distance = abs(points[index].bpm - predicted)
            if distance > error:
                error, location = distance, index
        if location is not None and error > epsilon:
            keep.add(location)
            pending.extend(((first, location), (location, last)))
    return [points[index] for index in sorted(keep)]


def _run_changes(points, settings):
    if len(points) < 5 or points[-1].time - points[0].time < settings.confirmation * 2:
        return []
    typical = median(p.bpm for p in points)
    threshold = max(settings.min_bpm, typical * settings.min_percent / 100)
    epsilon = max(.35, threshold * .23)
    vertices = _simplify(points, epsilon)
    groups = []
    for a, b in zip(vertices, vertices[1:]):
        delta = b.bpm - a.bpm
        if abs(delta) < epsilon * .7:
            continue
        direction = 1 if delta > 0 else -1
        if groups and groups[-1][2] == direction and abs(groups[-1][1].time - a.time) < 1e-6:
            groups[-1] = (groups[-1][0], b, direction)
        else:
            groups.append((a, b, direction))
    times = [p.time for p in points]
    changes = []
    for a, b, direction in groups:
        span = b.time - a.time
        abrupt = span <= max(6.0, settings.window * 1.75)
        # A late/early step is not confirmed merely because the track ended.
        # Long ramps can legitimately extend to the boundaries of the piece.
        if abrupt and (a.time - points[0].time < settings.confirmation
                       or points[-1].time - b.time < settings.confirmation):
            continue
        before = [p.bpm for p in points[bisect_left(times, a.time - settings.confirmation):bisect_right(times, a.time)]]
        after = [p.bpm for p in points[bisect_left(times, b.time):bisect_right(times, b.time + settings.confirmation)]]
        # Ramps reaching the edges have no external plateau; use their edge samples.
        if len(before) < 3:
            before = [p.bpm for p in points[:min(5, len(points))]]
        if len(after) < 3:
            after = [p.bpm for p in points[-min(5, len(points)):]]
        old, new = median(before), median(after)
        required = max(settings.min_bpm, old * settings.min_percent / 100)
        if (new - old) * direction < required:
            continue
        tolerance = max(required * .5, abs(new - old) * .25)
        if (sum(abs(v - old) <= tolerance for v in before) / len(before) < .9
                or sum(abs(v - new) <= tolerance for v in after) / len(after) < .9):
            continue
        if abrupt:
            start = end = (a.time + b.time) / 2
            kind = 'abrupt'
        else:
            start, end = a.time, b.time
            kind = 'acceleration' if direction > 0 else 'deceleration'
        changes.append(TempoChange(round(start, 3), round(end, 3), round(old, 2), round(new, 2), kind))
    return changes


def analyze_tempo(beats, settings=None, duration=None):
    settings = settings or TempoSettings()
    settings.validate()
    beats = clean_beats(beats)
    if duration is None:
        duration = beats[-1] if beats else 0.0
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(tr('אורך השיר אינו תקין.'))
    duration = max(float(duration), beats[-1] if beats else 0.0)
    points = _curve(beats, settings, duration)
    changes, run = [], []
    for point in points + [TempoPoint(duration, None)]:
        if point.bpm is None:
            if run:
                changes.extend(_run_changes(run, settings))
                run = []
        else:
            run.append(point)
    return TempoAnalysis(points, changes, duration, len(beats), settings)


def export_curve_csv(analysis, path):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_seconds', 'bpm_smoothed'])
        for point in analysis.points:
            writer.writerow([f'{point.time:.3f}', '' if point.bpm is None else f'{point.bpm:.3f}'])


def export_changes_csv(analysis, path):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['start_seconds', 'end_seconds', 'type', 'bpm_before', 'bpm_after', 'delta_bpm'])
        for change in analysis.changes:
            writer.writerow([change.start, change.end, change.kind, change.before, change.after, round(change.delta, 2)])
