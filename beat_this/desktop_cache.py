"""Persistent analysis history and cache controls."""
from beat_this.desktop_i18n import tr
from datetime import datetime
import json
from pathlib import Path
import tempfile

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from beat_this.analysis_cache import AnalysisCache
from beat_this.desktop_core import commit_outputs
from beat_this.desktop_models import data_dir, model_label
from beat_this.desktop_tempo import TempoDialog


class CacheDialog(QDialog):
    PAGE_SIZE = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('BPM Finder — היסטוריה ומטמון'))
        self.resize(1090, min(680, self.screen().availableGeometry().height() - 70))
        self.setMinimumSize(820, 540)
        self.page, self.rows, self.cache, self.selected_source = 0, [], None, None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        title = QLabel(tr('היסטוריית ניתוחים ומטמון'))
        title.setStyleSheet('font-size: 23px; font-weight: 700;')
        layout.addWidget(title)
        note = QLabel(tr('תוצאות הניתוח נשמרות בין הפעלות. שיר זהה עם אותו מודל והגדרות ייטען מהמטמון. קובצי השמע אינם מועתקים למטמון.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr('חיפוש לפי שם שיר, נתיב או מודל…'))
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(180)
        self.search.textChanged.connect(lambda: self.search_timer.start())
        self.search_timer.timeout.connect(self.search_changed)
        layout.addWidget(self.search)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([tr('שיר'), 'BPM', tr('מודל'), tr('נותח לאחרונה'), tr('שימוש אחרון'), tr('שליפות'), 'NPY'])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 7):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        self.table.currentCellChanged.connect(lambda *_: self.selection_changed())
        self.table.cellDoubleClicked.connect(lambda *_: self.show_graph())
        layout.addWidget(self.table, 1)
        pages = QHBoxLayout()
        self.previous_btn = self.button(tr('הקודמים'), lambda: self.change_page(-1), pages)
        self.page_label = QLabel()
        pages.addWidget(self.page_label)
        self.next_btn = self.button(tr('הבאים'), lambda: self.change_page(1), pages)
        pages.addStretch()
        self.button(tr('רענון'), self.reload, pages)
        layout.addLayout(pages)
        self.details = QLabel()
        self.details.setTextFormat(Qt.PlainText)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        self.graph_btn = self.button(tr('גרף BPM ושינויים'), self.show_graph, actions)
        self.save_btn = self.button(tr('שמירת פעימות…'), self.save_beats, actions)
        self.add_btn = self.button(tr('הוספת השיר לרשימה'), self.add_source, actions)
        self.add_btn.setToolTip(tr('הוספה לחלון הראשי; הניתוח ישתמש בהגדרות שנבחרו בחלון הראשי.'))
        actions.addStretch()
        self.remove_btn = self.button(tr('מחיקת בחירה'), self.remove_selected, actions)
        layout.addLayout(actions)
        bottom = QHBoxLayout()
        self.clear_btn = self.button(tr('ניקוי כל המטמון'), self.clear_cache, bottom)
        folder = QLabel(str(data_dir() / 'analysis-cache'))
        folder.setTextFormat(Qt.PlainText)
        folder.setLayoutDirection(Qt.LeftToRight)
        folder.setWordWrap(True)
        folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        folder.setStyleSheet('color: #58677b; font-size: 12px;')
        bottom.addWidget(folder, 1)
        self.button(tr('סגירה'), self.close, bottom)
        layout.addLayout(bottom)
        self.reload()

    @staticmethod
    def button(text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def search_changed(self):
        self.page = 0
        self.reload()

    def change_page(self, amount):
        self.page = max(0, self.page + amount)
        self.reload()

    def reload(self):
        try:
            self.cache = AnalysisCache()
            rows = self.cache.list_entries(self.search.text().strip(), self.PAGE_SIZE + 1, self.page * self.PAGE_SIZE)
            stats = self.cache.stats()
            self.rows = rows[:self.PAGE_SIZE]
            self.table.setRowCount(len(self.rows))
            for row, entry in enumerate(self.rows):
                values = [Path(entry['source_path']).name,
                          f'{entry["bpm"]:.2f}' if entry['bpm'] is not None else '—',
                          model_label(entry['model_ref']),
                          datetime.fromtimestamp(entry['analyzed_at']).strftime('%d/%m/%y %H:%M'),
                          datetime.fromtimestamp(entry['last_used_at']).strftime('%d/%m/%y %H:%M'),
                          str(entry['hits']), tr('כן') if entry['has_raw'] else '—']
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setToolTip(entry['source_path'] + '\n' + model_label(entry['model_ref']))
                    self.table.setItem(row, col, item)
            self.summary.setText(tr('{0} תוצאות שמורות · {1} שליפות מהמטמון · {2:.2f} MB של תוצאות', stats['count'], stats['hits'], stats['bytes'] / 1024 ** 2) +
                                 ('' if self.rows else tr(' · אין תוצאות לחיפוש הזה') if self.search.text().strip() else tr(' · ניתוח חדש יופיע כאן לאחר שיסתיים')))
            self.previous_btn.setEnabled(self.page > 0)
            self.next_btn.setEnabled(len(rows) > self.PAGE_SIZE)
            self.page_label.setText(tr('עמוד {0}', self.page + 1))
            self.clear_btn.setText(tr('ניקוי כל המטמון'))
        except Exception as exc:
            self.cache, self.rows = None, []
            self.table.setRowCount(0)
            self.summary.setText(tr('לא ניתן לקרוא את המטמון. ניתן לנסות שוב או לאפס אותו. {0}', exc))
            self.previous_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.clear_btn.setText(tr('איפוס המטמון'))
        self.selection_changed()

    def selected(self):
        selected = [index.row() for index in self.table.selectionModel().selectedRows()]
        row = self.table.currentRow()
        if row not in selected:
            row = selected[0] if selected else -1
        return self.rows[row] if 0 <= row < len(self.rows) else None

    def selection_changed(self):
        entry = self.selected()
        for button in (self.graph_btn, self.save_btn, self.add_btn, self.remove_btn):
            button.setEnabled(entry is not None)
        if entry:
            try:
                settings = json.loads(entry['settings_json'])
                if not isinstance(settings, dict):
                    settings = {}
            except (TypeError, ValueError):
                settings = {}
            self.details.setText(tr('{0}\n{1} פעימות · {2} פעימות ראשיות · DBN: ', entry['source_path'], entry['beat_count'], entry['downbeat_count']) +
                                 (tr('כן') if settings.get('dbn') else tr('לא')) + ' · Float16: ' + (tr('כן') if settings.get('float16') else tr('לא')))
            self.add_btn.setEnabled(Path(entry['source_path']).is_file())
        else:
            self.details.setText(tr('בחרו תוצאה להצגת גרף, לשמירת הפעימות או להוספת השיר לרשימת הניתוח.'))

    def read_selected(self):
        selected = self.selected()
        if not selected or self.cache is None:
            return None
        try:
            entry = self.cache.get(selected['key'])
            if entry is None:
                raise ValueError(tr('התוצאה חסרה או פגומה. ניתן להוסיף את שיר המקור לרשימה ולנתח אותו מחדש.'))
            return entry
        except Exception as exc:
            QMessageBox.warning(self, tr('לא ניתן לטעון תוצאה'), str(exc))
            return None

    def show_graph(self):
        entry = self.read_selected()
        if entry is None:
            return
        try:
            source = Path(entry.source_path)
            export_base = Path(entry.output_path) if entry.output_path else source.with_suffix('.beats')
            if not export_base.parent.is_dir():
                export_base = Path.home() / 'Documents' / source.with_suffix('.beats').name
            with tempfile.TemporaryDirectory(prefix='bpm-history-') as folder:
                beat_file = Path(folder) / source.with_suffix('.beats').name
                beat_file.write_text(entry.beats_text, encoding='utf-8')
                TempoDialog(beat_file, source if source.is_file() else None, self, export_base=export_base).exec()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, tr('לא ניתן להציג גרף קצב'), str(exc))

    def save_beats(self):
        entry = self.read_selected()
        if entry is None:
            return
        default = entry.output_path or str(Path(entry.source_path).with_suffix('.beats'))
        path, _ = QFileDialog.getSaveFileName(self, tr('שמירת הפעימות מהמטמון'), default, 'Beats (*.beats)')
        if path:
            temporary = None
            try:
                target = Path(path)
                if target.suffix.lower() != '.beats':
                    target = target.with_name(target.name + '.beats')
                    if target.exists() and QMessageBox.question(self, tr('קובץ קיים'), tr('קובץ הפעימות כבר קיים. להחליף אותו?'),
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                        return
                if target.resolve() == Path(entry.source_path).resolve():
                    raise ValueError(tr('יש לבחור שם פלט שונה מקובץ השמע.'))
                with tempfile.NamedTemporaryFile(dir=target.parent, mode='w', encoding='utf-8', suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(entry.beats_text)
                commit_outputs(temporary, target)
            except OSError as exc:
                QMessageBox.warning(self, tr('לא ניתן לשמור'), str(exc))
            except ValueError as exc:
                QMessageBox.warning(self, tr('בדיקת נתיב'), str(exc))
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def add_source(self):
        entry = self.selected()
        if entry and Path(entry['source_path']).is_file():
            self.selected_source = entry['source_path']
            self.accept()

    def remove_selected(self):
        keys = [self.rows[index.row()]['key'] for index in self.table.selectionModel().selectedRows()]
        if not keys or self.cache is None:
            return
        if QMessageBox.question(self, tr('מחיקה מהמטמון'), tr('למחוק {0} תוצאות מהמטמון? קובצי השמע והפלט הקיימים יישארו במקומם.', len(keys)),
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            self.cache.remove(keys)
            self.page = 0
            self.reload()
        except Exception as exc:
            QMessageBox.warning(self, tr('לא ניתן למחוק'), str(exc))

    def clear_cache(self):
        if QMessageBox.question(self, tr('ניקוי המטמון'), tr('למחוק את כל תוצאות הניתוח השמורות במטמון? קובצי השמע והפלט הקיימים יישארו במקומם.'),
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            if self.cache is None:
                # Explicit recovery of a damaged database; no other files are touched.
                folder = data_dir() / 'analysis-cache'
                for suffix in ('', '-wal', '-shm'):
                    (folder / ('results.sqlite3' + suffix)).unlink(missing_ok=True)
            else:
                self.cache.clear()
            self.page = 0
            self.reload()
        except Exception as exc:
            QMessageBox.warning(self, tr('לא ניתן לנקות'), str(exc))
