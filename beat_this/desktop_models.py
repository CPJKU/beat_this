"""Human-readable model catalog, local lookup, and atomic downloads (no Qt/torch)."""
from beat_this.desktop_i18n import tr
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
from urllib.parse import urlparse

from beat_this.desktop_core import MODEL_NAMES

MODEL_SOURCE = 'https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp'
MODEL_DOCS = 'https://github.com/CPJKU/beat_this#available-models'
DOWNLOAD_BASE = 'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp'


@dataclass(frozen=True)
class ModelInfo:
    key: str
    title: str
    description: str
    size_mb: float
    research: bool = False

    @property
    def url(self):
        return f'{DOWNLOAD_BASE}/{self.key}.ckpt'


def _catalog():
    catalog = {}
    experiments = {
        'notempoaug': tr('ללא שינויי קצב באימון'), 'nosumhead': tr('ראשי פלט נפרדים'),
        'nomaskaug': tr('ללא מיסוך באימון'), 'nopartialt': tr('ללא טרנספורמרים חלקיים'),
        'noshifttol': tr('ללא סבילות להיסטי זמן'), 'nopitchaug': tr('ללא שינויי גובה צליל באימון'),
        'noshifttolnoweights': tr('ללא סבילות זמן ומשקולות הפסד'),
    }
    for key in MODEL_NAMES:
        variant = int(key[-1]) + 1
        if key.startswith('final'):
            title = tr('רגיל — דיוק גבוה') + ('' if variant == 1 else tr(' · חלופה {0}', variant))
            description = tr('המודל המלא לשימוש יומיומי. החלופות הן אימונים שונים של אותה ארכיטקטורה.')
        elif key.startswith('small'):
            title = tr('מהיר — מודל קטן') + ('' if variant == 1 else tr(' · חלופה {0}', variant))
            description = tr('מודל קל יותר, המתאים לניתוח מהיר ולמחשבים עם פחות משאבים.')
        elif key.startswith('single_final'):
            title = tr('מחקר — חלוקת אימון יחידה · חלופה {0}', variant)
            description = tr('המודל המלא בחלוקת האימון והאימות היחידה של המאמר.')
        elif key.startswith('fold'):
            title = tr('מחקר — אימות צולב · קבוצה {0}', variant)
            description = tr('אחד משמונת המודלים שאומנו על חלוקות האימות הצולב של המאמר.')
        elif key.startswith('hung'):
            title = tr('מחקר — נתוני אימון מצומצמים · חלופה {0}', variant)
            description = tr('מודל עם קבוצת נתוני האימון של השוואת Hung במאמר.')
        else:
            name = key[len('single_'):-1]
            title = tr('מחקר — {0} · חלופה {1}', experiments[name], variant)
            description = tr('מודל ניסוי של המאמר, להשוואות מחקריות; אינו ברירת מחדל לשימוש רגיל.')
        catalog[key] = ModelInfo(key, title, description, 8.1 if key.startswith('small') else 78,
                                 not key.startswith(('final', 'small')))
    return catalog


MODEL_CATALOG = _catalog()


def app_root():
    return Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent


def data_dir():
    return Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.local' / 'share'))) / 'BPM Finder'


def model_cache():
    folder = data_dir() / 'models'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def model_folders():
    torch_home = Path(os.environ.get('TORCH_HOME', str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'torch')))
    return [data_dir() / 'models', app_root() / 'models', torch_home / 'hub' / 'checkpoints']


def find_model(key):
    if key not in MODEL_CATALOG:
        file = Path(key).expanduser()
        return file.resolve() if file.is_file() else None
    for folder in model_folders():
        for name in (f'{key}.ckpt', f'beat_this-{key}.ckpt'):
            file = folder / name
            if file.is_file() and file.stat().st_size > 0:
                return file
    return None


def model_label(value):
    if value in MODEL_CATALOG:
        return MODEL_CATALOG[value].title
    if str(value).startswith(('http://', 'https://')):
        return tr('מודל מקישור אישי')
    return tr('מודל מקומי — {0}', Path(value).name)


def custom_download_path(url):
    return model_cache() / ('custom-' + hashlib.sha256(url.encode()).hexdigest()[:16] + '.ckpt')


class DownloadCancelled(Exception):
    pass


def download_file(url, target, progress=None, cancelled=None):
    if urlparse(url).scheme not in ('https', 'http'):
        raise ValueError(tr('יש להזין קישור ישיר מסוג HTTPS או HTTP לקובץ מודל.'))
    cancelled = cancelled or (lambda: False)
    if cancelled():
        raise DownloadCancelled()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=target.parent, suffix='.part')
    temporary = Path(name)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'BPM-Finder/2.0'})
        with os.fdopen(handle, 'wb') as output:
            with urllib.request.urlopen(request, timeout=20) as response:
                length = int(response.headers.get('Content-Length') or 0)
                done = 0
                while True:
                    if cancelled():
                        raise DownloadCancelled()
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, length)
                if done == 0 or (length and done != length):
                    raise ValueError(tr('ההורדה לא הושלמה. המודל הקיים נשמר; אפשר לנסות שוב.'))
                if cancelled():
                    raise DownloadCancelled()
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
