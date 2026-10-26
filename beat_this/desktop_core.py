"""Desktop task planning. Deliberately independent of Qt, torch and numpy."""
from beat_this.desktop_i18n import tr
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import math
import os
import shutil
import tempfile

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.oga', '.opus', '.m4a',
                    '.aac', '.aif', '.aiff', '.aifc', '.wma', '.mp4', '.webm', '.au'}
MODEL_NAMES = ([f'{prefix}{i}' for prefix in ('final', 'small', 'single_final', 'hung')
                for i in range(3)] + [f'fold{i}' for i in range(8)] +
               [f'single_{name}{i}' for name in ('notempoaug', 'nosumhead', 'nomaskaug',
                'nopartialt', 'noshifttol', 'nopitchaug', 'noshifttolnoweights')
                for i in range(3)])


@dataclass
class Options:
    model: str = 'final0'
    output: str = ''
    output_is_file: bool = False
    suffix: str = '.beats'
    append: bool = False
    skip_existing: bool = False
    touch_first: bool = False
    dbn: bool = False
    gpu: int = 0
    float16: bool = False
    activations: bool = False
    use_cache: bool = True
    force_reanalyze: bool = False


@dataclass
class Task:
    source: Path
    output: Path
    skip: bool = False


def estimate_bpm(beats):
    values = [float(x) for x in beats]
    intervals = [b - a for a, b in zip(values, values[1:])
                 if math.isfinite(a) and math.isfinite(b) and b > a]
    return round(60 / median(intervals), 2) if intervals else None


def is_accelerator_error(exc, device):
    if not str(device).startswith('cuda'):
        return False
    visited = set()
    while exc is not None and id(exc) not in visited:
        visited.add(id(exc))
        if any(word in str(exc).lower() for word in (
                'cuda error', 'cuda out of memory', 'hiperror', 'hip error',
                'hip out of memory', 'cudnn', 'cublas', 'rocfft', 'miopen')):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def commit_outputs(beat_temp, beat_output, raw_temp=None, raw_output=None):
    """Commit both outputs, restoring old raw data if the beats file is locked."""
    if raw_temp is None:
        os.replace(beat_temp, beat_output)
        return
    backup = None
    keep_backup = False
    if raw_output.exists():
        handle, name = tempfile.mkstemp(dir=raw_output.parent, prefix='.beat-this-backup-', suffix='.tmp')
        os.close(handle)
        backup = Path(name)
        try:
            shutil.copyfile(raw_output, backup)
        except Exception:
            backup.unlink(missing_ok=True)
            raise
    raw_committed = False
    try:
        os.replace(raw_temp, raw_output)
        raw_committed = True
        os.replace(beat_temp, beat_output)
    except Exception:
        if raw_committed:
            try:
                if backup is not None:
                    os.replace(backup, raw_output)
                else:
                    raw_output.unlink(missing_ok=True)
            except Exception as rollback_error:
                keep_backup = True
                raise RuntimeError(tr('לא ניתן לשחזר את פלט NPY. הגיבוי נשמר כאן: {0}', backup)) from rollback_error
        raise
    finally:
        if backup is not None and not keep_backup:
            backup.unlink(missing_ok=True)


def build_tasks(inputs, options):
    if not options.model.strip():
        raise ValueError(tr('יש לבחור מודל.'))
    if (not options.suffix.startswith('.') or len(options.suffix) < 2
            or any(c in options.suffix for c in '/\\:*?"<>|')
            or options.suffix.endswith(('.', ' '))):
        raise ValueError(tr('סיומת הפלט חייבת להתחיל בנקודה, למשל \u200e.beats'))
    sources = [Path(p).expanduser().resolve() for p in inputs]
    if not sources:
        raise ValueError(tr('יש להוסיף קובץ או תיקייה.'))
    for source in sources:
        if not source.exists():
            raise ValueError(tr('הקובץ או התיקייה לא קיימים: {0}', source))
    output = Path(options.output).expanduser().resolve() if options.output else None
    if options.output_is_file:
        if output is None or len(sources) != 1 or not sources[0].is_file():
            raise ValueError(tr('שמירה לקובץ מסוים זמינה רק עבור קובץ קלט יחיד.'))
        if output.is_dir():
            raise ValueError(tr('יש לבחור קובץ פלט, לא תיקייה.'))
    elif output and output.exists() and not output.is_dir():
        raise ValueError(tr('מיקום הפלט הוא קובץ. יש לבחור תיקייה או מצב שמירה לקובץ.'))
    tasks, seen = [], set()
    for source in sources:
        files = sorted(source.rglob('*')) if source.is_dir() else [source]
        for file in files:
            if not file.is_file() or file in seen:
                continue
            if source.is_dir():
                if file.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                if output and output != source and output in file.parents:
                    continue
            seen.add(file)
            if options.output_is_file:
                target = output
            else:
                relative = file.relative_to(source) if source.is_dir() else Path(file.name)
                base = output / relative if output else file
                target = (base.with_name(base.name + options.suffix) if options.append
                          else base.with_suffix(options.suffix))
            tasks.append(Task(file, target, options.skip_existing and target.exists()))
    if not tasks:
        raise ValueError(tr('לא נמצאו קובצי שמע בתיקיות שנבחרו.'))
    all_sources = {task.source for task in tasks}
    targets = set()
    for task in tasks:
        outputs = [task.output]
        if options.activations:
            outputs.append(task.output.with_suffix('.npy'))
        for target in outputs:
            if target in all_sources:
                raise ValueError(tr('הפלט ידרוס קובץ קלט. יש לשנות סיומת או מיקום: {0}', target.name))
            if target in targets:
                raise ValueError(tr('שני פלטים יקבלו אותו שם: {0}. יש להפעיל הוספת סיומת או לבחור קבצים בנפרד.', target.name))
            if target.is_dir():
                raise ValueError(tr('כבר קיימת תיקייה בשם קובץ הפלט: {0}', target))
            targets.add(target)
    return tasks
