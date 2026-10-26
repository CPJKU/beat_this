import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from beat_this.desktop_models import (MODEL_CATALOG, model_label, find_model,
                                      download_file, DownloadCancelled)
from beat_this.desktop_i18n import tr


class ModelTests(unittest.TestCase):
    def test_labels_are_friendly_and_catalog_covers_all_original_ids(self):
        from beat_this.desktop_core import MODEL_NAMES
        self.assertEqual(set(MODEL_CATALOG), set(MODEL_NAMES))
        self.assertEqual(tr('רגיל — דיוק גבוה'), model_label('final0'))
        self.assertEqual(tr('מהיר — מודל קטן'), model_label('small0'))
        self.assertNotEqual(model_label('final0'), 'final0')
        self.assertIn('sample', model_label(r'C:\models\sample.ckpt'))

    def test_model_lookup_prefers_download_cache_then_bundled_then_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            user, bundle, legacy = (root / name for name in ('user', 'bundle', 'legacy'))
            for folder in (user, bundle, legacy):
                folder.mkdir()
            (bundle / 'final0.ckpt').write_bytes(b'bundled')
            with patch('beat_this.desktop_models.model_folders', return_value=[user, bundle, legacy]):
                self.assertEqual(find_model('final0'), bundle / 'final0.ckpt')
                (user / 'final0.ckpt').write_bytes(b'user')
                self.assertEqual(find_model('final0'), user / 'final0.ckpt')

    def test_download_is_atomic_and_cancel_does_not_replace_installed_model(self):
        payload = b'model weights' * 1000
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/model.ckpt'
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'model.ckpt'
                target.write_bytes(b'old model')
                with self.assertRaises(DownloadCancelled):
                    download_file(url, target, cancelled=lambda: True)
                self.assertEqual(target.read_bytes(), b'old model')
                progress = []
                download_file(url, target, progress=lambda done, total: progress.append((done, total)))
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual(progress[-1], (len(payload), len(payload)))
                self.assertEqual(list(Path(directory).glob('*.part')), [])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
