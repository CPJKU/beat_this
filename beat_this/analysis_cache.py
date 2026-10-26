"""Persistent, content-addressed analysis results. Never stores source audio."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sqlite3
import time

from beat_this.desktop_models import data_dir, model_label, MODEL_CATALOG

CACHE_VERSION = 1
RUNTIME_PACKAGES = ('torch', 'torchaudio', 'torchcodec', 'soundfile', 'numpy', 'scipy',
                    'soxr', 'einops', 'rotary-embedding-torch', 'madmom')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def engine_signature():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ('inference.py', 'preprocessing.py', 'utils.py', 'model/beat_tracker.py',
                 'model/roformer.py', 'model/postprocessor.py'):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    for name in RUNTIME_PACKAGES:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = 'unavailable'
        digest.update(f'{name}:{version}'.encode())
    return digest.hexdigest()


def inference_settings(options, engine=None):
    return dict(dbn=bool(options.dbn), float16=bool(options.float16), gpu=int(options.gpu),
                engine=engine if engine is not None else engine_signature())


def cache_key(audio_hash, model_hash, settings):
    payload = json.dumps(dict(version=CACHE_VERSION, audio=audio_hash, model=model_hash,
                              settings=settings), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class CacheEntry:
    key: str
    source_path: str
    model_ref: str
    settings: dict
    beats_text: str
    raw_data: bytes | None
    bpm: float | None
    beat_count: int
    downbeat_count: int
    analyzed_at: float
    last_used_at: float
    hits: int
    has_raw: bool
    output_path: str


class AnalysisCache:
    def __init__(self, folder=None):
        self.folder = Path(folder) if folder is not None else data_dir() / 'analysis-cache'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / 'results.sqlite3'
        with self.connection() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('''CREATE TABLE IF NOT EXISTS results (
                key TEXT PRIMARY KEY, audio_hash TEXT NOT NULL, model_hash TEXT NOT NULL,
                settings_json TEXT NOT NULL, source_path TEXT NOT NULL, model_ref TEXT NOT NULL,
                search_text TEXT NOT NULL, beats_text TEXT NOT NULL, beats_hash TEXT NOT NULL,
                raw_data BLOB, raw_hash TEXT, bpm REAL, beat_count INTEGER NOT NULL,
                downbeat_count INTEGER NOT NULL, analyzed_at REAL NOT NULL,
                last_used_at REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0,
                output_path TEXT NOT NULL DEFAULT '')''')

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(str(self.path), timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def store(self, audio_hash, model_hash, settings, source_path, model_ref, beats_text,
              bpm, beat_count, downbeat_count, raw_data=None, output_path=''):
        key = cache_key(audio_hash, model_hash, settings)
        now = time.time()
        source_path = str(Path(source_path).resolve())
        if bpm is not None and not math.isfinite(bpm):
            raise ValueError('Invalid cached BPM')
        raw_hash = hashlib.sha256(raw_data).hexdigest() if raw_data is not None else None
        values = (key, audio_hash, model_hash, json.dumps(settings, sort_keys=True),
                  source_path, str(model_ref), (source_path + ' ' + model_label(model_ref)).casefold(),
                  beats_text, hashlib.sha256(beats_text.encode('utf-8')).hexdigest(),
                  raw_data, raw_hash, bpm, int(beat_count), int(downbeat_count), now, now, str(output_path))
        with self.connection() as connection:
            connection.execute('''INSERT INTO results
                (key,audio_hash,model_hash,settings_json,source_path,model_ref,search_text,
                 beats_text,beats_hash,raw_data,raw_hash,bpm,beat_count,downbeat_count,
                 analyzed_at,last_used_at,output_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET source_path=excluded.source_path,
                model_ref=excluded.model_ref,search_text=excluded.search_text,
                beats_text=excluded.beats_text,beats_hash=excluded.beats_hash,
                raw_data=excluded.raw_data,raw_hash=excluded.raw_hash,bpm=excluded.bpm,
                beat_count=excluded.beat_count,downbeat_count=excluded.downbeat_count,
                analyzed_at=excluded.analyzed_at,last_used_at=excluded.last_used_at,
                output_path=excluded.output_path''', values)
        return key

    def get(self, key, require_raw=False, touch=False):
        # Do not load potentially large activation blobs for ordinary beat reuse/history.
        raw = 'raw_data' if require_raw else 'NULL AS raw_data'
        with self.connection() as connection:
            row = connection.execute(f'''SELECT key,source_path,model_ref,settings_json,
                beats_text,beats_hash,{raw},raw_hash,bpm,beat_count,downbeat_count,
                analyzed_at,last_used_at,hits,output_path FROM results WHERE key=?''', (key,)).fetchone()
            if row is None:
                return None
            if hashlib.sha256(row['beats_text'].encode('utf-8')).hexdigest() != row['beats_hash']:
                return None
            if require_raw and (row['raw_data'] is None or not row['raw_hash']
                    or hashlib.sha256(row['raw_data']).hexdigest() != row['raw_hash']):
                return None
            try:
                settings = json.loads(row['settings_json'])
            except (ValueError, TypeError):
                return None
            if not isinstance(settings, dict):
                return None
            now = time.time() if touch else row['last_used_at']
            if touch:
                connection.execute('UPDATE results SET hits=hits+1,last_used_at=? WHERE key=?', (now, key))
            return CacheEntry(key=row['key'], source_path=row['source_path'], model_ref=row['model_ref'],
                settings=settings, beats_text=row['beats_text'], raw_data=row['raw_data'], bpm=row['bpm'],
                beat_count=row['beat_count'], downbeat_count=row['downbeat_count'],
                analyzed_at=row['analyzed_at'], last_used_at=now, hits=row['hits'] + int(touch),
                has_raw=bool(row['raw_hash']), output_path=row['output_path'])

    def lookup(self, audio_hash, model_hash, settings, require_raw=False):
        return self.get(cache_key(audio_hash, model_hash, settings), require_raw, touch=True)

    def update_location(self, key, source_path, output_path, model_ref=None):
        source_path = str(Path(source_path).resolve())
        with self.connection() as connection:
            row = connection.execute('SELECT model_ref FROM results WHERE key=?', (key,)).fetchone()
            if row:
                ref = str(model_ref) if model_ref is not None else row['model_ref']
                search = (source_path + ' ' + model_label(ref)).casefold()
                connection.execute('UPDATE results SET source_path=?,output_path=?,search_text=?,model_ref=? WHERE key=?',
                                   (source_path, str(output_path), search, ref, key))

    def list_entries(self, search='', limit=200, offset=0):
        # Older history rows may have been indexed in the other interface language.
        matches = [key for key in MODEL_CATALOG if search.casefold() in model_label(key).casefold()]
        model_query = ','.join('?' for _ in matches) or 'NULL'
        with self.connection() as connection:
            rows = connection.execute(f'''SELECT key,source_path,model_ref,settings_json,bpm,
                beat_count,downbeat_count,analyzed_at,last_used_at,hits,
                (raw_hash IS NOT NULL) AS has_raw,output_path FROM results
                WHERE instr(search_text,?) > 0 OR model_ref IN ({model_query})
                ORDER BY last_used_at DESC LIMIT ? OFFSET ?''',
                (search.casefold(), *matches, max(1, int(limit)), max(0, int(offset)))).fetchall()
            return [dict(row) for row in rows]

    def stats(self):
        with self.connection() as connection:
            row = connection.execute('''SELECT COUNT(*) AS count,COALESCE(SUM(hits),0) AS hits,
                COALESCE(SUM(length(CAST(beats_text AS BLOB))+COALESCE(length(raw_data),0)),0) AS bytes
                FROM results''').fetchone()
            return dict(row)

    def remove(self, keys):
        with self.connection() as connection:
            connection.executemany('DELETE FROM results WHERE key=?', [(str(key),) for key in keys])
        self.compact()

    def clear(self):
        with self.connection() as connection:
            connection.execute('DELETE FROM results')
        self.compact()

    def compact(self):
        # Space reclamation is best-effort if another app instance is currently using SQLite.
        try:
            with self.connection() as connection:
                connection.execute('VACUUM')
                connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        except sqlite3.OperationalError:
            pass
