"""Model management and installation diagnostics dialogs."""
from beat_this.desktop_i18n import tr
import os
from pathlib import Path
import threading
import uuid

from PySide6.QtCore import Qt, QThread, Signal, QUrl, QProcess, QItemSelectionModel
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QLineEdit,
    QProgressBar, QFileDialog, QPlainTextEdit)

from beat_this.desktop_models import (MODEL_CATALOG, MODEL_SOURCE, MODEL_DOCS, find_model,
    model_cache, model_label, custom_download_path, download_file, DownloadCancelled)
from beat_this.desktop_process import JsonWorker


class DownloadThread(QThread):
    progress = Signal(int, int)
    completed = Signal(str)
    failed = Signal(str)
    def __init__(self, url, target, parent):
        super().__init__(parent)
        self.url, self.target = url, target
        self.cancel_event = threading.Event()

    def run(self):
        try:
            download_file(self.url, self.target, self.progress.emit, self.cancel_event.is_set)
            self.completed.emit(str(self.target))
        except DownloadCancelled:
            self.failed.emit(tr('ההורדה בוטלה. המודל הקיים נשמר.'))
        except Exception as exc:
            self.failed.emit(tr('לא ניתן להוריד את המודל: {0}', exc))


class ModelDialog(QDialog):
    def __init__(self, current='final0', parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('BPM Finder — מודלים והורדות'))
        self.resize(880, 710)
        self.selection = None
        self.thread = self.worker = None
        self.closing = False
        self.cancelled = False
        self.pending = self.destination = self.pending_selection = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        title = QLabel(tr('בחירת מודל זיהוי פעימות'))
        title.setStyleSheet('font-size: 23px; font-weight: 700;')
        layout.addWidget(title)
        hint = QLabel(tr('לשימוש רגיל בחרו ״דיוק גבוה״. לניתוח קל ומהיר יותר בחרו ״מודל קטן״.\nאפשר להשתמש רק ב־checkpoints תואמים למנוע Beat This!.'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.research = QCheckBox(tr('הצגת מודלי מחקר והשוואות נוספות'))
        self.research.setChecked(current in MODEL_CATALOG and MODEL_CATALOG[current].research)
        self.research.toggled.connect(self.refresh)
        layout.addWidget(self.research)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr('מודל'), tr('גודל הורדה משוער'), tr('זמינות')])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.selected)
        self.table.cellDoubleClicked.connect(lambda *_: self.use_selected())
        layout.addWidget(self.table, 1)
        self.details = QLabel()
        self.details.setTextFormat(Qt.PlainText)
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        buttons = QHBoxLayout()
        self.use_btn = QPushButton(tr('שימוש במודל'))
        self.use_btn.setObjectName('primary')
        self.use_btn.clicked.connect(self.use_selected)
        self.download_btn = QPushButton(tr('הורדה למחשב'))
        self.download_btn.clicked.connect(self.download_selected)
        self.local_btn = QPushButton(tr('טעינת מודל מהמחשב…'))
        self.local_btn.clicked.connect(self.local_model)
        for item in (self.use_btn, self.download_btn, self.local_btn):
            buttons.addWidget(item)
        layout.addLayout(buttons)
        url_row = QHBoxLayout()
        self.url = QLineEdit()
        self.url.setLayoutDirection(Qt.LeftToRight)
        self.url.setPlaceholderText('https://…/compatible-model.ckpt')
        self.url.setAccessibleName(tr('קישור ישיר למודל תואם'))
        self.url_btn = QPushButton(tr('הורדת מודל מקישור'))
        self.url_btn.clicked.connect(self.download_url)
        url_row.addWidget(self.url)
        url_row.addWidget(self.url_btn)
        layout.addLayout(url_row)
        sources = QHBoxLayout()
        for text, url in [(tr('מאגר ההורדות הרשמי'), MODEL_SOURCE), (tr('הסבר על המודלים והמקור'), MODEL_DOCS)]:
            item = QPushButton(text)
            item.clicked.connect(lambda checked=False, link=url: QDesktopServices.openUrl(QUrl(link)))
            sources.addWidget(item)
        folder = QPushButton(tr('פתיחת תיקיית המודלים'))
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(model_cache()))))
        sources.addWidget(folder)
        layout.addLayout(sources)
        self.status = QLabel(tr('בחרו מודל מהרשימה.'))
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        bottom = QHBoxLayout()
        self.cancel_btn = QPushButton(tr('ביטול הורדה'))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        bottom.addWidget(self.cancel_btn)
        bottom.addStretch()
        close = QPushButton(tr('סגירה'))
        close.clicked.connect(self.close)
        bottom.addWidget(close)
        layout.addLayout(bottom)
        self.refresh()
        for row, key in enumerate(self.keys):
            if key == current:
                self.table.setCurrentCell(row, 0, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                break

    def refresh(self):
        previous = self.current_key() if hasattr(self, 'keys') else None
        self.keys = [key for key, info in MODEL_CATALOG.items() if self.research.isChecked() or not info.research]
        self.table.setRowCount(len(self.keys))
        for row, key in enumerate(self.keys):
            info = MODEL_CATALOG[key]
            for column, text in enumerate([info.title, f'{info.size_mb:g} MB', tr('זמין במחשב') if find_model(key) else tr('זמין להורדה')]):
                self.table.setItem(row, column, QTableWidgetItem(text))
        if self.keys:
            row = self.keys.index(previous) if previous in self.keys else 0
            self.table.setCurrentCell(row, 0, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        self.selected()

    def current_key(self):
        row = self.table.currentRow()
        return self.keys[row] if 0 <= row < len(self.keys) else None

    def selected(self):
        key = self.current_key()
        if not key:
            return
        info = MODEL_CATALOG[key]
        self.details.setText(info.description + tr('\nמזהה טכני: {0}', key))
        self.use_btn.setText(tr('שימוש במודל') if find_model(key) else tr('הורדה ושימוש במודל'))
        self.download_btn.setText(tr('הורדה מחדש / תיקון') if find_model(key) else tr('הורדה למחשב'))

    def set_busy(self, busy):
        for item in (self.table, self.use_btn, self.download_btn, self.local_btn, self.url,
                     self.url_btn, self.research):
            item.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def use_selected(self):
        if self.thread or self.worker:
            return
        key = self.current_key()
        if key and find_model(key):
            self.selection = key
            self.accept()
        elif key:
            self.download_selected(use=True)

    def download_selected(self, checked=False, use=False):
        key = self.current_key()
        if key:
            self.begin_download(MODEL_CATALOG[key].url, model_cache() / f'{key}.ckpt', key, use)

    def download_url(self):
        url = self.url.text().strip()
        if url:
            target = custom_download_path(url)
            self.begin_download(url, target, str(target), True)

    def local_model(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('בחירת מודל Beat This! תואם'), '', tr('Checkpoint (*.ckpt *.pt *.pth);;כל הקבצים (*)'))
        if path:
            self.pending_selection = path
            self.pending = self.destination = None
            self.use_after = True
            self.verify(path)

    def begin_download(self, url, destination, selection, use):
        if self.thread or self.worker:
            return
        self.cancelled = False
        self.destination = Path(destination)
        self.pending = self.destination.with_name(self.destination.name + f'.{uuid.uuid4().hex}.pending')
        self.pending_selection, self.use_after = selection, use
        self.set_busy(True)
        self.status.setText(tr('מתחיל הורדה…'))
        self.progress.setRange(0, 0)
        self.thread = DownloadThread(url, self.pending, self)
        self.thread.progress.connect(self.download_progress)
        self.thread.completed.connect(self.downloaded)
        self.thread.failed.connect(self.download_failed)
        self.thread.finished.connect(self.thread_finished)
        self.thread.start()

    def download_progress(self, done, total):
        self.progress.setRange(0, total or 0)
        self.progress.setValue(done)
        self.status.setText(tr('מוריד מודל… {0:.1f}', done / 1024 ** 2) + (tr(' מתוך {0:.1f} MB', total / 1024 ** 2) if total else ' MB'))

    def downloaded(self, path):
        if self.closing or self.cancelled or (self.thread and self.thread.cancel_event.is_set()):
            Path(path).unlink(missing_ok=True)
            return
        self.verify(path)

    def download_failed(self, message):
        self.status.setText(message)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def thread_finished(self):
        thread, self.thread = self.thread, None
        if thread:
            thread.deleteLater()
        if self.closing:
            self.close()
        elif not self.worker:
            self.set_busy(False)

    def verify(self, path):
        self.cancelled = False
        self.set_busy(True)
        self.status.setText(tr('בודק שהמודל תקין ותואם למנוע…'))
        self.progress.setRange(0, 0)
        self.validated = False
        self.worker = JsonWorker(self)
        self.worker.event_received.connect(self.verify_event)
        self.worker.finished.connect(self.verify_finished)
        self.worker.errorOccurred.connect(self.verify_error)
        self.worker.start_request(dict(action='verify_model', model=str(path)))

    def verify_error(self, error):
        if error == QProcess.FailedToStart:
            self.status.setText(tr('לא ניתן להפעיל את בדיקת המודל.'))
            self.verify_finished()

    def verify_event(self, event):
        if event['event'] == 'verified' and not self.cancelled and not self.closing:
            self.validated = True
        elif event['event'] == 'fatal':
            self.status.setText(tr('המודל אינו תקין או אינו תואם ל־Beat This!.\n') + event.get('error', ''))

    def verify_finished(self, *args):
        worker, self.worker = self.worker, None
        if worker:
            worker.deleteLater()
        if self.validated and not self.closing and not self.cancelled:
            try:
                if self.pending:
                    os.replace(self.pending, self.destination)
                self.status.setText(tr('המודל נבדק ומוכן לשימוש.'))
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
                self.refresh()
                if self.use_after:
                    self.selection = self.pending_selection
                    self.accept()
            except OSError as exc:
                self.status.setText(tr('לא ניתן לשמור את המודל: {0}', exc))
        elif not self.validated and not self.closing and not self.status.text().startswith(tr('המודל אינו')):
            self.status.setText(tr('בדיקת המודל לא הושלמה. המודל הקיים לא שונה.'))
        if self.pending:
            self.pending.unlink(missing_ok=True)
            self.pending = None
        self.set_busy(False)
        if self.closing and not self.thread:
            self.close()

    def cancel(self):
        self.cancelled = True
        if self.thread:
            self.thread.cancel_event.set()
            self.status.setText(tr('מבטל הורדה…'))
        if self.worker:
            self.validated = False
            self.worker.kill()
        self.cancel_btn.setEnabled(False)

    def reject(self):
        self.close()

    def closeEvent(self, event):
        if self.thread or self.worker:
            self.closing = True
            self.cancel()
            event.ignore()
        else:
            event.accept()


class DiagnosticsDialog(QDialog):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('BPM Finder — בדיקת תקינות ותיקון'))
        self.resize(760, 480)
        self.model = model
        self.worker = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr('בדיקת מנוע הניתוח, DBN, קריאת שמע, תיקיית המודלים והמודל שנבחר.')))
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setStyleSheet('font-family: "Segoe UI"; font-size: 14px;')
        layout.addWidget(self.report)
        actions = QHBoxLayout()
        self.retry_btn = QPushButton(tr('בדיקה חוזרת'))
        self.retry_btn.clicked.connect(self.run)
        actions.addWidget(self.retry_btn)
        self.fix_btn = QPushButton(tr('הורדה מחדש / תיקון מודלים'))
        self.fix_btn.clicked.connect(self.repair)
        actions.addWidget(self.fix_btn)
        close = QPushButton(tr('סגירה'))
        close.clicked.connect(self.close)
        actions.addWidget(close)
        layout.addLayout(actions)
        self.run()

    def run(self):
        self.report.setPlainText(tr('הבדיקה פועלת…'))
        self.retry_btn.setEnabled(False)
        self.fix_btn.setEnabled(False)
        self.worker = JsonWorker(self)
        self.worker.event_received.connect(self.on_event)
        self.worker.log_received.connect(self.report.appendPlainText)
        self.worker.finished.connect(self.worker_finished)
        self.worker.errorOccurred.connect(self.error)
        self.worker.start_request(dict(action='doctor', model=self.model))

    def on_event(self, event):
        if event['event'] == 'diagnostics':
            lines = []
            for check in event['checks']:
                lines.append((tr('תקין') if check['ok'] else tr('נדרש תיקון')) + ' — ' + check['title'] + '\n' + check['detail'])
            self.report.setPlainText('\n\n'.join(lines))
        elif event['event'] == 'fatal':
            self.report.appendPlainText(event.get('error', tr('שגיאה')))

    def error(self, error):
        if error == QProcess.FailedToStart:
            self.report.appendPlainText(tr('מנוע הבדיקה לא נפתח. חלצו שוב את חבילת Windows המלאה.'))
            self.worker_finished()

    def worker_finished(self, *args):
        worker, self.worker = self.worker, None
        if worker:
            worker.deleteLater()
        self.retry_btn.setEnabled(True)
        self.fix_btn.setEnabled(True)

    def repair(self):
        ModelDialog(self.model, self).exec()

    def closeEvent(self, event):
        if self.worker:
            self.worker.kill()
            self.worker.waitForFinished(3000)
        event.accept()

    def reject(self):
        self.close()
