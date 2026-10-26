"""Windowless launcher with a readable startup error if dependencies are missing."""
import ctypes
import os
from pathlib import Path
import sys
import traceback
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
os.chdir(root)
try:
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        from beat_this.desktop_process import frozen_worker_main
        sys.exit(frozen_worker_main(sys.argv[2]))
    from beat_this.desktop import main
    sys.exit(main())
except Exception:
    details = traceback.format_exc()
    log_root = Path(os.environ.get('LOCALAPPDATA', str(root))) / 'BPM Finder'
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / 'ui-startup-error.log').write_text(details, encoding='utf-8')
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        import json
        (Path(sys.argv[2]) / 'events.jsonl').write_text(
            json.dumps(dict(event='fatal', error=details), ensure_ascii=False) + '\n', encoding='utf-8')
        sys.exit(1)
    message = 'Could not open BPM Finder.\nExtract the complete application package again.\n\n'
    title = 'BPM Finder — Startup error'
    try:
        from beat_this.desktop_i18n import tr
        message = tr('לא ניתן לפתוח את BPM Finder.\nחלצו שוב את חבילת התוכנה לתיקייה מלאה.\n\n')
        title = tr('BPM Finder — שגיאת הפעלה')
    except Exception:
        pass
    ctypes.windll.user32.MessageBoxW(0, message + details[-1600:], title, 0x10)
