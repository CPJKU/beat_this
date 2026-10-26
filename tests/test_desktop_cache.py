import hashlib
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from beat_this.analysis_cache import AnalysisCache, file_hash, cache_key, inference_settings
from beat_this.desktop_core import Options


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.song = self.root / 'שיר עם רווח.wav'
        self.song.write_bytes(b'audio data 1')
        self.folder = self.root / 'cache'
        self.cache = AnalysisCache(self.folder)
        self.settings = dict(dbn=False, float16=False, gpu=0, engine='test-runtime')
        self.audio = file_hash(self.song)
        self.model = hashlib.sha256(b'model').hexdigest()
        self.text = '0.0\t1\n0.5\t2\n1.0\t3\n'

    def store(self, raw=None):
        return self.cache.store(self.audio, self.model, self.settings, self.song, 'final0',
                                self.text, 120, 3, 1, raw_data=raw)

    def test_persists_across_instances_and_restores_exact_beats(self):
        key = self.store()
        reopened = AnalysisCache(self.folder)
        entry = reopened.lookup(self.audio, self.model, self.settings)
        self.assertEqual(entry.key, key)
        self.assertEqual(entry.beats_text, self.text)
        self.assertEqual(entry.bpm, 120)
        self.assertEqual(reopened.stats()['count'], 1)
        self.assertEqual(reopened.stats()['hits'], 1)

    def test_copy_or_rename_reuses_content_but_modified_bytes_do_not(self):
        self.store()
        copy = self.root / 'renamed.wav'
        copy.write_bytes(self.song.read_bytes())
        self.assertIsNotNone(self.cache.lookup(file_hash(copy), self.model, self.settings))
        before = self.song.stat()
        self.song.write_bytes(b'audio data 2')
        os.utime(self.song, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertIsNone(self.cache.lookup(file_hash(self.song), self.model, self.settings))

    def test_model_and_inference_changes_invalidate_but_export_settings_do_not(self):
        self.store()
        self.assertIsNone(self.cache.lookup(self.audio, 'another-model', self.settings))
        for key, value in [('dbn', True), ('float16', True), ('gpu', -1), ('engine', 'changed')]:
            self.assertIsNone(self.cache.lookup(self.audio, self.model, dict(self.settings, **{key: value})))
        a = inference_settings(Options(), engine='same')
        b = inference_settings(Options(output='somewhere', suffix='.txt', append=True,
                                      skip_existing=True, touch_first=True, activations=True), engine='same')
        self.assertEqual(a, b)

    def test_raw_data_is_required_only_when_requested(self):
        self.store()
        self.assertIsNone(self.cache.lookup(self.audio, self.model, self.settings, require_raw=True))
        raw = b'\x93NUMPY' + b'raw activations'
        self.store(raw)
        entry = self.cache.lookup(self.audio, self.model, self.settings, require_raw=True)
        self.assertEqual(entry.raw_data, raw)
        self.assertTrue(entry.has_raw)
        self.assertEqual(self.cache.stats()['count'], 1)

    def test_corrupted_payload_is_a_miss(self):
        key = self.store()
        with closing(sqlite3.connect(self.cache.path)) as connection:
            connection.execute('UPDATE results SET beats_text=? WHERE key=?', ('broken', key))
            connection.commit()
        self.assertIsNone(self.cache.lookup(self.audio, self.model, self.settings))

    def test_history_search_and_cleanup_do_not_touch_audio_or_exports(self):
        key = self.store()
        exported = self.root / 'song.beats'
        exported.write_text(self.text)
        self.assertEqual(len(self.cache.list_entries('שיר')), 1)
        self.assertEqual(self.cache.list_entries('not-present'), [])
        self.cache.remove([key])
        self.assertEqual(self.cache.stats()['count'], 0)
        self.store()
        self.cache.clear()
        self.assertEqual(self.cache.stats()['count'], 0)
        self.assertTrue(self.song.exists() and exported.exists())


if __name__ == '__main__':
    unittest.main()
