from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from beat_this import desktop_worker as worker
from beat_this.analysis_cache import AnalysisCache, file_hash, inference_settings, engine_signature
from beat_this.desktop_core import Options


class WorkerCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.model = self.root / 'song.wav', self.root / 'model.ckpt'
        self.source.write_bytes(b'audio fixture')
        self.model.write_bytes(b'model fixture')
        self.cache = AnalysisCache(self.root / 'cache')
        self.text = '0\t1\n0.5\t2\n1\t3\n'
        self.events = []
        self.options = Options(model=str(self.model), gpu=-1, output=str(self.root / 'output'))
        self.request = dict(inputs=[str(self.source)], options=vars(self.options))

    def run_fake(self, fault=None):
        class Logits:
            def cpu(self):
                return self
            def numpy(self):
                return [1]
        tracker = SimpleNamespace(signal2spect=lambda *_: [],
            spect2frames=lambda *_: (Logits(), Logits()), frames2beats=lambda *_: ([0, .5, 1], [0]))
        numpy = SimpleNamespace(asarray=lambda a:a, vstack=lambda a:a,
                                save=lambda handle, _:handle.write(b'raw-result'))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        inference = SimpleNamespace(File2Beats=lambda *_:tracker, load_audio=lambda _: ([], 22050))
        utils = SimpleNamespace(save_beat_tsv=lambda b,d,p:p.write_text(self.text,encoding='utf-8'))
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules', {'torch':torch, 'numpy':numpy,
                'beat_this.inference':inference, 'beat_this.utils':utils}))
            stack.enter_context(patch.object(worker, 'AnalysisCache', return_value=self.cache))
            stack.enter_context(patch.object(worker, 'emit', side_effect=lambda event,**kw:self.events.append(dict(event=event,**kw))))
            if fault:
                stack.enter_context(fault)
            result = worker.run(self.request)
        self.assertEqual(result, 0, self.events)
        self.assertEqual(next(e for e in self.events if e['event']=='result')['status'], 'success')
        self.assertTrue(any(e['event']=='cache_warning' for e in self.events))
        self.assertEqual((self.root/'output/song.beats').read_text(encoding='utf-8'), self.text)
        self.assertEqual((self.root/'output/song.npy').read_bytes(), b'raw-result')

    def test_hit_does_not_import_numpy_torch_or_inference(self):
        self.cache.store(file_hash(self.source), file_hash(self.model), inference_settings(self.options),
                         self.source, str(self.model), self.text, 120, 3, 1)
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in ('torch', 'numpy', 'beat_this.inference'):
                raise AssertionError('Cache hit imported heavy inference: ' + name)
            return original(name, *args, **kwargs)
        with patch.object(worker, 'AnalysisCache', return_value=self.cache), \
             patch.object(worker, 'emit', side_effect=lambda event,**kw:self.events.append(dict(event=event,**kw))), \
             patch.object(builtins, '__import__', side_effect=guarded):
            self.assertEqual(worker.run(self.request), 0)
        result = next(e for e in self.events if e['event']=='result')
        self.assertEqual(result['status'], 'cached')
        self.assertEqual(Path(result['output']).read_text(encoding='utf-8'), self.text)

    def test_optional_raw_read_failure_preserves_both_exports(self):
        self.options.activations = True
        original = Path.read_bytes
        def failing(path):
            if path.suffix == '.tmp':
                raise OSError('optional raw cache read blocked')
            return original(path)
        self.run_fake(patch.object(Path, 'read_bytes', failing))
        self.assertEqual(self.cache.stats()['count'], 1)
        self.assertFalse(self.cache.list_entries()[0]['has_raw'])

    def test_optional_beat_read_failure_preserves_both_exports(self):
        self.options.activations = True
        original = Path.read_text
        def failing(path, *args, **kwargs):
            if path.suffix == '.tmp':
                raise OSError('optional beats cache read blocked')
            return original(path, *args, **kwargs)
        self.run_fake(patch.object(Path, 'read_text', failing))
        self.assertEqual(self.cache.stats()['count'], 0)

    def test_cache_store_failure_preserves_both_exports(self):
        self.options.activations = True
        self.run_fake(patch.object(AnalysisCache, 'store', side_effect=OSError('cache disk unavailable')))

    def test_source_hash_failure_falls_back_to_inference(self):
        self.options.activations = True
        def failing(path):
            if Path(path) == self.source:
                raise OSError('cache fingerprint unavailable')
            return file_hash(path)
        self.run_fake(patch.object(worker, 'file_hash', side_effect=failing))

    def test_decoder_versions_invalidate_runtime_signature(self):
        for name in ('soundfile', 'torchcodec', 'scipy'):
            with self.subTest(name=name):
                with patch('beat_this.analysis_cache.metadata.version', return_value='1'):
                    before = engine_signature()
                with patch('beat_this.analysis_cache.metadata.version', side_effect=lambda package:'2' if package==name else '1'):
                    self.assertNotEqual(before, engine_signature())


if __name__ == '__main__':
    unittest.main()
