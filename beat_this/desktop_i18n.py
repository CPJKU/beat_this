"""Small, dependency-free desktop localization and persistent language preference."""
import json
import os
from pathlib import Path
import tempfile


def preference_file():
    return Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.local/share'))) / 'BPM Finder' / 'language.json'


def read_language():
    override = os.environ.get('BPM_FINDER_LANGUAGE')
    if override in ('en', 'he'):
        return override
    try:
        value = json.loads(preference_file().read_text(encoding='utf-8')).get('language')
        return value if value in ('en', 'he') else 'en'
    except (OSError, ValueError, TypeError, AttributeError):
        return 'en'


def save_language(language):
    if language not in ('en', 'he'):
        raise ValueError('Unsupported language')
    target = preference_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, mode='w', encoding='utf-8', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump({'language':language}, handle)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


LANGUAGE = read_language()
ENGLISH = json.loads(Path(__file__).with_name('translations_en.json').read_text(encoding='utf-8'))


def tr(message, *values):
    result = message if LANGUAGE == 'he' else ENGLISH.get(message, message)
    return result.format(*values) if values else result
