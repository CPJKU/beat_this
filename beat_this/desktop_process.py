"""Qt JSON worker transport; frozen windowless EXEs communicate through temp files."""
import codecs
import json
from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Signal
from beat_this.desktop_models import app_root
from beat_this.desktop_i18n import LANGUAGE


class JsonWorker(QProcess):
    event_received = Signal(dict)
    log_received = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.buffer = ''
        self.out_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.err_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.frozen = bool(getattr(sys, 'frozen', False))
        self.folder = None
        self.positions = {'events': 0, 'errors': 0}
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.drain)
        self.readyReadStandardOutput.connect(self.drain)
        self.readyReadStandardError.connect(self.drain)
        self.started.connect(self._started)
        self.finished.connect(self._finished)
        self.setWorkingDirectory(str(app_root()))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('BPM_FINDER_LANGUAGE', LANGUAGE)
        self.setProcessEnvironment(environment)

    def start_request(self, request):
        self.payload = (json.dumps(request, ensure_ascii=False) + '\n').encode('utf-8')
        if self.frozen:
            self.folder = tempfile.TemporaryDirectory(prefix='bpm-worker-')
            root = Path(self.folder.name)
            (root / 'request.json').write_bytes(self.payload)
            self.start(sys.executable, ['--worker', str(root)])
            self.timer.start()
        else:
            executable = str(Path(sys.executable).with_name('python.exe'))
            self.start(executable, ['-u', '-X', 'utf8', '-m', 'beat_this.desktop_worker'])

    def _started(self):
        if not self.frozen:
            self.write(self.payload)
            self.closeWriteChannel()

    def _file_bytes(self, name):
        path = Path(self.folder.name) / (name + '.jsonl' if name == 'events' else name + '.log')
        if not path.exists():
            return b''
        with path.open('rb') as handle:
            handle.seek(self.positions[name])
            data = handle.read()
            self.positions[name] = handle.tell()
            return data

    def drain(self):
        if self.frozen and self.folder:
            output, errors = self._file_bytes('events'), self._file_bytes('errors')
        else:
            output, errors = bytes(self.readAllStandardOutput()), bytes(self.readAllStandardError())
        self.buffer += self.out_decoder.decode(output)
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():
                try:
                    self.event_received.emit(json.loads(line))
                except (ValueError, TypeError):
                    self.log_received.emit(line)
        if errors:
            self.log_received.emit(self.err_decoder.decode(errors))

    def _finished(self, *args):
        self.timer.stop()
        self.drain()
        if self.folder:
            self.folder.cleanup()
            self.folder = None


def frozen_worker_main(folder):
    root = Path(folder)
    # PyInstaller --windowed deliberately has no sys.std*; don't rely on them.
    previous = sys.stdin, sys.stdout, sys.stderr
    try:
        with (root / 'request.json').open(encoding='utf-8') as incoming, \
             (root / 'events.jsonl').open('w', encoding='utf-8', buffering=1) as outgoing, \
             (root / 'errors.log').open('w', encoding='utf-8', buffering=1) as errors:
            sys.stdin, sys.stdout, sys.stderr = incoming, outgoing, errors
            print('Starting isolated analysis worker…', file=sys.stderr, flush=True)
            from beat_this.desktop_worker import main
            result = main()
            print('Analysis worker finished.', file=sys.stderr, flush=True)
            return result
    finally:
        sys.stdin, sys.stdout, sys.stderr = previous
