"""Isolated inference process; stdin is one JSON request, stdout is JSON lines."""
from beat_this.desktop_i18n import tr
import contextlib
from dataclasses import fields
import json
from pathlib import Path
import sys
import tempfile
import traceback

from beat_this.desktop_core import Options, build_tasks, estimate_bpm, is_accelerator_error, commit_outputs
from beat_this.desktop_models import MODEL_CATALOG, find_model, download_file, model_cache, custom_download_path
from beat_this.analysis_cache import AnalysisCache, file_hash, inference_settings

PROTOCOL_STDOUT = sys.stdout


def emit(event, **data):
    print(json.dumps(dict(event=event, **data), ensure_ascii=False), file=PROTOCOL_STDOUT, flush=True)


def run(request):
    options = Options(**{f.name: request.get('options', {}).get(f.name, f.default)
                         for f in fields(Options)})
    if options.force_reanalyze:
        options.skip_existing = False
    tasks = build_tasks(request['inputs'], options)
    emit('plan', tasks=[dict(source=str(t.source), output=str(t.output), skip=t.skip)
                        for t in tasks])
    if all(t.skip for t in tasks):
        for index, task in enumerate(tasks):
            emit('result', index=index, status='skipped', output=str(task.output))
        emit('done', success=0, failed=0, skipped=len(tasks), cached=0)
        return 0
    model_ref = options.model
    path = find_model(options.model)
    if options.model in MODEL_CATALOG:
        if path is None:
            info = MODEL_CATALOG[options.model]
            emit('download', message=tr('מוריד מודל בפעם הראשונה…'), done=0, total=0)
            path = download_file(info.url, model_cache() / f'{info.key}.ckpt',
                progress=lambda done, total: emit('download', done=done, total=total))
    elif str(options.model).startswith(('http://', 'https://')):
        path = custom_download_path(options.model)
        if not path.is_file():
            emit('download', done=0, total=0)
            download_file(options.model, path,
                progress=lambda done, total: emit('download', done=done, total=total))
    if path is not None:
        options.model = str(path)
    cache = settings = model_digest = model_stamp = None
    if options.use_cache:
        try:
            if path is None:
                raise ValueError(tr('לשימוש במטמון יש לבחור מודל מקומי או מודל מהקטלוג.'))
            emit('cache_check', message=tr('מכין בדיקת מטמון…'))
            cache = AnalysisCache()
            before = path.stat()
            settings, model_digest = inference_settings(options), file_hash(path)
            model_stamp = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (model_stamp.st_size, model_stamp.st_mtime_ns, model_stamp.st_ino):
                raise RuntimeError(tr('המודל השתנה בזמן בדיקת המטמון.'))
        except Exception as exc:
            cache = None
            emit('cache_warning', message=tr('המטמון אינו זמין; הניתוח ימשיך כרגיל. {0}', exc))

    # Heavy inference imports are deliberately deferred until the first cache miss.
    tracker = torch = np = file2beats_type = load_audio_fn = save_tsv_fn = None
    device = 'cpu'

    def ensure_tracker():
        nonlocal tracker, torch, np, file2beats_type, load_audio_fn, save_tsv_fn, device
        if tracker is not None:
            return tracker
        with contextlib.redirect_stdout(sys.stderr):
            import importlib.util
            if options.dbn and importlib.util.find_spec('madmom') is None:
                raise RuntimeError(tr('DBN דורש את הספרייה madmom, שאינה מותקנת. אפשר לבטל את DBN ולהריץ שוב.'))
            import numpy as numpy_module
            import torch as torch_module
            from beat_this.inference import File2Beats, load_audio
            from beat_this.utils import save_beat_tsv
            torch, np = torch_module, numpy_module
            file2beats_type, load_audio_fn, save_tsv_fn = File2Beats, load_audio, save_beat_tsv
            if torch.cuda.is_available() and options.gpu >= 0:
                if options.gpu >= torch.cuda.device_count():
                    raise ValueError(tr('מספר הכרטיס הגרפי שנבחר אינו קיים. אפשר לבחור CPU או כרטיס 0.'))
                device = f'cuda:{options.gpu}'
            emit('loading', device=device)
            try:
                tracker = File2Beats(options.model, device, options.float16, options.dbn)
            except Exception as exc:
                if not accelerator_error(exc):
                    raise
                tracker = fallback()
        return tracker

    def accelerator_error(exc):
        return is_accelerator_error(exc, device)

    def fallback():
        nonlocal device
        device = 'cpu'
        emit('warning', message=tr('מנוע הכרטיס הגרפי אינו תואם. הניתוח ממשיך אוטומטית באמצעות CPU.'))
        with contextlib.redirect_stdout(sys.stderr):
            return file2beats_type(options.model, 'cpu', False, options.dbn)

    def analyze(tracker, source):
        if options.activations:
            signal, sr = load_audio_fn(source)
            spect = tracker.signal2spect(signal, sr)
            beat_logits, downbeat_logits = tracker.spect2frames(spect)
            beats, downbeats = tracker.frames2beats(beat_logits, downbeat_logits)
            raw = np.vstack([beat_logits.cpu().numpy(), downbeat_logits.cpu().numpy()])
        else:
            beats, downbeats = tracker(source)
            raw = None
        return beats, downbeats, raw

    success = failed = skipped = cached_count = 0
    for index, task in enumerate(tasks):
        emit('processing', index=index, source=str(task.source))
        if options.skip_existing and task.output.exists():
            skipped += 1
            emit('result', index=index, status='skipped', output=str(task.output))
            continue
        claimed = False
        temporary = []
        try:
            entry = audio_digest = stamp = None
            task_cache = cache
            if task_cache is not None:
                emit('cache_check', index=index, message=tr('בודק מטמון: {0}', task.source.name))
                try:
                    before = task.source.stat()
                    audio_digest = file_hash(task.source)
                    stamp = task.source.stat()
                    current_model = path.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (stamp.st_size, stamp.st_mtime_ns, stamp.st_ino):
                        raise RuntimeError(tr('קובץ השמע השתנה בזמן בדיקת המטמון.'))
                    if (current_model.st_size, current_model.st_mtime_ns, current_model.st_ino) != (model_stamp.st_size, model_stamp.st_mtime_ns, model_stamp.st_ino):
                        raise RuntimeError(tr('המודל השתנה בזמן הריצה.'))
                    if not options.force_reanalyze:
                        entry = task_cache.lookup(audio_digest, model_digest, settings, options.activations)
                except Exception as exc:
                    task_cache = None
                    emit('cache_warning', message=tr('לא ניתן להשתמש במטמון; מבצע ניתוח חדש. {0}', exc))
            task.output.parent.mkdir(parents=True, exist_ok=True)
            if options.touch_first and not task.output.exists():
                try:
                    with task.output.open('x'):
                        pass
                    claimed = True
                    emit('cleanup', path=str(task.output), empty=True)
                except FileExistsError:
                    if options.skip_existing:
                        skipped += 1
                        emit('result', index=index, status='skipped', output=str(task.output))
                        continue
            if entry is None:
                emit('cache_miss', index=index)
                ensure_tracker()
            with contextlib.redirect_stdout(sys.stderr):
                if entry is not None:
                    current = task.source.stat()
                    if (current.st_size, current.st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
                        raise RuntimeError(tr('קובץ השמע השתנה בזמן קריאת המטמון. נסו שוב.'))
                    beats_text, raw_bytes = entry.beats_text, entry.raw_data
                    bpm, beat_count, downbeat_count = entry.bpm, entry.beat_count, entry.downbeat_count
                else:
                    try:
                        beats, downbeats, raw = analyze(tracker, task.source)
                    except Exception as exc:
                        if not accelerator_error(exc):
                            raise
                        tracker = fallback()
                        beats, downbeats, raw = analyze(tracker, task.source)
                    bpm, beat_count, downbeat_count = estimate_bpm(beats), len(beats), len(downbeats)
                    beats_text = raw_bytes = None
                with tempfile.NamedTemporaryFile(dir=task.output.parent, suffix='.tmp', delete=False) as handle:
                    temp_path = Path(handle.name)
                temporary.append(temp_path)
                emit('cleanup', path=str(temp_path))
                if entry is not None:
                    temp_path.write_text(beats_text, encoding='utf-8')
                else:
                    save_tsv_fn(beats, downbeats, temp_path)
                    if task_cache is not None:
                        try:
                            beats_text = temp_path.read_text(encoding='utf-8')
                        except Exception as exc:
                            task_cache = None
                            emit('cache_warning', message=tr('לא ניתן לקרוא את הפעימות למטמון. הפלט יישמר כרגיל. {0}', exc))
                if options.activations:
                    with tempfile.NamedTemporaryFile(dir=task.output.parent, suffix='.tmp', delete=False) as handle:
                        temp_npy = Path(handle.name)
                        temporary.append(temp_npy)
                        emit('cleanup', path=str(temp_npy))
                        if entry is not None:
                            handle.write(raw_bytes)
                        else:
                            np.save(handle, raw)
                    if task_cache is not None and entry is None:
                        try:
                            raw_bytes = temp_npy.read_bytes()
                        except Exception as exc:
                            emit('cache_warning', message=tr('לא ניתן לשמור נתוני NPY במטמון. הפלט יישמר כרגיל. {0}', exc))
                    commit_outputs(temp_path, task.output, temp_npy, task.output.with_suffix('.npy'))
                else:
                    commit_outputs(temp_path, task.output)
            claimed = False
            emit('cleaned', path=str(task.output))
            key = entry.key if entry is not None else None
            if task_cache is not None:
                try:
                    if entry is not None:
                        task_cache.update_location(entry.key, task.source, task.output, model_ref)
                    else:
                        current = task.source.stat()
                        current_model = path.stat()
                        if (current_model.st_size, current_model.st_mtime_ns, current_model.st_ino) != (model_stamp.st_size, model_stamp.st_mtime_ns, model_stamp.st_ino):
                            raise RuntimeError(tr('המודל השתנה בזמן הניתוח; התוצאה לא נשמרה במטמון.'))
                        if (current.st_size, current.st_mtime_ns, current.st_ino) != (stamp.st_size, stamp.st_mtime_ns, stamp.st_ino):
                            raise RuntimeError(tr('קובץ השמע השתנה בזמן הניתוח; התוצאה לא נשמרה במטמון.'))
                        key = task_cache.store(audio_digest, model_digest, settings, task.source, model_ref,
                            beats_text, bpm, beat_count, downbeat_count, raw_data=raw_bytes,
                            output_path=str(task.output))
                except Exception as exc:
                    emit('cache_warning', message=tr('הפלט נשמר, אך המטמון לא עודכן. {0}', exc))
            success += 1
            cached_count += int(entry is not None)
            emit('result', index=index, status='cached' if entry is not None else 'success',
                 output=str(task.output), bpm=bpm, beats=beat_count, downbeats=downbeat_count,
                 cache_key=key)
        except Exception as exc:
            failed += 1
            traceback.print_exc(file=sys.stderr)
            emit('result', index=index, status='error', error=str(exc), output=str(task.output))
        finally:
            for temp_path in temporary:
                temp_path.unlink(missing_ok=True)
                emit('cleaned', path=str(temp_path))
            if claimed and task.output.exists() and task.output.stat().st_size == 0:
                task.output.unlink()
                emit('cleaned', path=str(task.output))
    emit('done', success=success, failed=failed, skipped=skipped, cached=cached_count)
    return 1 if failed else 0


def utilities(request):
    with contextlib.redirect_stdout(sys.stderr):
        print('Loading analysis engine…', file=sys.stderr, flush=True)
        import torch
        print('Loading beat tracker…', file=sys.stderr, flush=True)
        from beat_this.inference import load_model
        if request['action'] == 'verify_model':
            path = str(find_model(request['model']) or request['model'])
            load_model(path, 'cpu')
            emit('verified', model=path)
        elif request['action'] == 'doctor':
            import soundfile as sf
            from beat_this.preprocessing import LogMelSpect
            checks = []
            def check(title, operation):
                try:
                    detail = operation()
                    checks.append(dict(title=title, ok=True, detail=str(detail or tr('תקין'))))
                except Exception as exc:
                    checks.append(dict(title=title, ok=False, detail=str(exc)))
            check(tr('מנוע ניתוח CPU'), lambda: (LogMelSpect()(torch.zeros(22050)).shape, torch.__version__))
            check(tr('קריאת קובצי שמע'), lambda: 'MP3, WAV, FLAC' if 'MPEG-1/2 Audio' in str(sf.available_formats()) or 'MP3' in sf.available_formats() else str(sf.available_formats()))
            def dbn_check():
                from madmom.features.downbeats import DBNDownBeatTrackingProcessor
                DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=50)
                return tr('madmom מותקן ופעיל')
            check(tr('עיבוד DBN'), dbn_check)
            def cache_check():
                path = model_cache()
                with tempfile.TemporaryFile(dir=path) as handle:
                    handle.write(b'ok')
                return path
            check(tr('תיקיית הורדת מודלים'), cache_check)
            if request.get('model'):
                def model_check():
                    path = find_model(request['model'])
                    if not path:
                        raise ValueError(tr('המודל לא הורד. פתחו ניהול מודלים ובחרו הורדה.'))
                    load_model(str(path), 'cpu')
                    return path
                check(tr('המודל שנבחר'), model_check)
            emit('diagnostics', checks=checks, frozen=bool(getattr(sys, 'frozen', False)))
    return 0


def main():
    try:
        request = json.loads(sys.stdin.readline())
        return utilities(request) if request.get('action') else run(request)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit('fatal', error=str(exc))
        return 1


if __name__ == '__main__':
    sys.exit(main())
