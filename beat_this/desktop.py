"""Simple native RTL desktop interface for Beat This!"""
from beat_this.desktop_i18n import tr
from beat_this.desktop_i18n import LANGUAGE, save_language
from dataclasses import asdict
from pathlib import Path
import sys

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QSettings, QTimer, QUrl, QTranslator, QLibraryInfo
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QScrollArea, QLayout,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from beat_this.desktop_core import AUDIO_EXTENSIONS, MODEL_NAMES, Options, build_tasks
from beat_this.desktop_models import MODEL_CATALOG, model_label, find_model
from beat_this.desktop_model_dialog import ModelDialog, DiagnosticsDialog
from beat_this.desktop_process import JsonWorker
from beat_this.desktop_tempo import TempoDialog
from beat_this.desktop_cache import CacheDialog

ROOT = Path(__file__).resolve().parent.parent
STYLE = '''
QWidget { font-family: "Segoe UI"; font-size: 14px; color: #202c3b; }
QMainWindow, QWidget#root { background: #f3f5f8; }
QFrame#card { background: white; border: 1px solid #dce2ea; border-radius: 10px; }
QLabel#title { font-size: 29px; font-weight: 700; }
QLabel#subtitle, QLabel#hint { color: #58677b; }
QLabel#section { font-size: 16px; font-weight: 600; }
QPushButton { background: #fff; border: 1px solid #ccd5e0; border-radius: 6px;
    padding: 8px 15px; min-height: 20px; }
QPushButton:hover { background: #edf2f9; border-color: #9cacc1; }
QPushButton:focus { border: 2px solid #2865d3; }
QPushButton#primary { background: #245ac7; color: white; border-color: #245ac7; font-weight: 600; }
QPushButton#primary:hover { background: #194aac; }
QPushButton:disabled { color: #8b96a5; background: #edf0f4; border-color: #dce2ea; }
QLineEdit, QComboBox, QSpinBox { background: white; border: 1px solid #ccd5e0;
    border-radius: 5px; padding: 6px 8px; min-height: 21px; selection-background-color: #245ac7; }
QComboBox::drop-down { border: none; width: 25px; }
QComboBox QAbstractItemView { background: white; color: #202c3b;
    selection-background-color: #245ac7; selection-color: white; }
QComboBox QAbstractItemView::item:selected { background: #245ac7; color: white; }
QTableWidget { border: 1px solid #e0e5ed; border-radius: 5px; background: white;
    alternate-background-color: #f8fafc; gridline-color: #eef1f5; selection-background-color: #e2edff;
    selection-color: #18345c; }
QHeaderView::section { background: #f4f6fa; border: none; border-bottom: 1px solid #e0e5ed;
    padding: 9px; font-weight: 600; }
QProgressBar { background: #e5eaf1; border: none; border-radius: 4px; max-height: 9px; min-height: 9px; }
QProgressBar::chunk { background: #245ac7; border-radius: 4px; }
QPlainTextEdit { background: #f8fafc; border: 1px solid #dce2ea; border-radius: 5px; font-family: Consolas; font-size: 12px; }
QCheckBox { spacing: 7px; padding: 3px; }
QCheckBox::indicator { width: 17px; height: 17px; }
'''


def button(text, callback, primary=False):
    item = QPushButton(text)
    item.clicked.connect(callback)
    if primary:
        item.setObjectName('primary')
    return item


def label(text, name=None):
    item = QLabel(text)
    if name:
        item.setObjectName(name)
    return item


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('BPM Finder · Beat This!')
        self.resize(980, min(960, self.screen().availableGeometry().height() - 70))
        self.setMinimumSize(820, 700)
        self.settings = QSettings('BeatThis', 'BPMFinder')
        self.inputs = []
        self.tasks = []
        self.results = {}
        self.process = None
        self.buffer = ''
        self.cancelled = False
        self.got_done = False
        self.completed_count = 0
        self.cleanup_files = {}
        self.run_warning = ''
        self.cache_warning = ''
        self._build()
        self._restore()
        pending = self.settings.value('language_restart_inputs', [], type=list)
        self.settings.remove('language_restart_inputs')
        if pending:
            self.add_paths(pending)
        self.setAcceptDrops(True)

    def _build(self):
        root = QWidget(objectName='root')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        layout = QVBoxLayout(root)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(14)
        top = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(3)
        titles.addWidget(label('BPM Finder', 'title'))
        titles.addWidget(label(tr('בחירת מוזיקה, זיהוי פעימות ושמירת התוצאות'), 'subtitle'))
        top.addLayout(titles)
        top.addStretch()
        badge = label('Beat This!  /  1.1')
        badge.setLayoutDirection(Qt.LeftToRight)
        badge.setStyleSheet('color: #63748c; font-size: 12px;')
        top.addWidget(badge)
        top.addWidget(label(tr('שפה')))
        self.language = QComboBox()
        self.language.setLayoutDirection(Qt.LeftToRight)
        self.language.addItem('English', 'en')
        self.language.addItem('עברית', 'he')
        self.language.setCurrentIndex(self.language.findData(LANGUAGE))
        self.language.currentIndexChanged.connect(self.change_language)
        top.addWidget(self.language)
        layout.addLayout(top)

        card = QFrame(objectName='card')
        body = QVBoxLayout(card)
        body.setContentsMargins(18, 14, 18, 16)
        body.setSpacing(12)
        row = QHBoxLayout()
        row.addWidget(label(tr('1  קובצי שמע'), 'section'))
        row.addStretch()
        self.count = label(tr('לא נבחרו קבצים'), 'hint')
        row.addWidget(self.count)
        body.addLayout(row)
        actions = QHBoxLayout()
        self.add_files_btn = button(tr('הוספת קבצים'), self.add_files)
        self.add_folder_btn = button(tr('הוספת תיקייה'), self.add_folder)
        self.remove_btn = button(tr('הסרת בחירה'), self.remove_selected)
        self.clear_btn = button(tr('ניקוי רשימה'), self.clear_inputs)
        for item in (self.add_files_btn, self.add_folder_btn, self.remove_btn, self.clear_btn):
            actions.addWidget(item)
        actions.addStretch()
        self.tempo_btn = button(tr('גרף BPM ושינויים'), self.open_tempo)
        self.tempo_btn.setEnabled(False)
        self.tempo_btn.setToolTip(tr('לאחר ניתוח: גרף BPM לאורך השיר ורשימת שינויים בזמן. אפשר גם ללחוץ פעמיים על שורת תוצאה.'))
        actions.addWidget(self.tempo_btn)
        body.addLayout(actions)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([tr('קובץ / תיקייה'), tr('מצב'), tr('BPM משוער'), tr('פעימות'), tr('פעימות ראשיות')])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(160)
        self.table.cellDoubleClicked.connect(self.open_result)
        body.addWidget(self.table)
        self.empty_hint = label(tr('אפשר לגרור לכאן קבצים ותיקיות. לחיצה כפולה על תוצאה פותחת גרף BPM ושינויי קצב.'), 'hint')
        self.empty_hint.setWordWrap(True)
        body.addWidget(self.empty_hint)
        layout.addWidget(card, 1)

        self.options_card = QFrame(objectName='card')
        options = QVBoxLayout(self.options_card)
        options.setContentsMargins(18, 14, 18, 14)
        options.setSpacing(10)
        options.addWidget(label(tr('2  הגדרות ניתוח ושמירה'), 'section'))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        self.model = QComboBox()
        for key, info in MODEL_CATALOG.items():
            if not info.research:
                self.model.addItem(info.title, key)
        self.model.setToolTip(tr('בחרו תצורת זיהוי מוכנה. לניהול, הורדות, קובץ מקומי או קישור: מודלים והורדות.'))
        self.model.currentIndexChanged.connect(self.update_model_hint)
        grid.addWidget(label(tr('תצורת זיהוי')), 0, 0)
        grid.addWidget(self.model, 0, 1)
        model_file = button(tr('מודלים והורדות…'), self.choose_model)
        grid.addWidget(model_file, 0, 2)
        self.output_mode = QComboBox()
        self.output_mode.addItems([tr('ליד קובצי המקור'), tr('בתיקייה אחרת'), tr('לקובץ מסוים (קלט יחיד)')])
        self.output_mode.currentIndexChanged.connect(self._output_mode_changed)
        grid.addWidget(label(tr('שמירת פלט')), 1, 0)
        grid.addWidget(self.output_mode, 1, 1, 1, 2)
        self.output = QLineEdit()
        self.output.setLayoutDirection(Qt.LeftToRight)
        self.output.setPlaceholderText(tr('בחירת מיקום שמירה'))
        self.output_browse = button(tr('בחירה…'), self.choose_output)
        grid.addWidget(self.output, 2, 1)
        grid.addWidget(self.output_browse, 2, 2)
        grid.setColumnStretch(1, 1)
        options.addLayout(grid)
        self.model_hint = label('', 'hint')
        self.model_hint.setWordWrap(True)
        options.addWidget(self.model_hint)
        cache_row = QHBoxLayout()
        self.use_cache = QCheckBox(tr('זכור ניתוחי שירים במטמון'))
        self.use_cache.setToolTip(tr('שומר תוצאות בין הפעלות. שיר זהה עם אותו מודל והגדרות ייטען מהמטמון. שינוי בתוכן השמע או בניתוח דורש חישוב חדש.'))
        cache_row.addWidget(self.use_cache)
        cache_row.addStretch()
        self.history_btn = button(tr('היסטוריה ומטמון…'), self.open_history)
        cache_row.addWidget(self.history_btn)
        self.reanalyze_btn = button(tr('ניתוח מחדש'), self.reanalyze)
        self.reanalyze_btn.setToolTip(tr('ניתוח חדש של כל הקבצים ברשימה, גם אם יש מטמון או פלט קיים. מעדכן את המטמון כשהוא מופעל.'))
        self.reanalyze_btn.setEnabled(False)
        cache_row.addWidget(self.reanalyze_btn)
        options.addLayout(cache_row)
        self.advanced_btn = button(tr('הגדרות מתקדמות  +'), self.toggle_advanced)
        self.advanced_btn.setCheckable(True)
        options.addWidget(self.advanced_btn, 0, Qt.AlignRight)
        self.advanced = QWidget()
        adv = QGridLayout(self.advanced)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setVerticalSpacing(8)
        adv.setHorizontalSpacing(12)
        adv.setRowMinimumHeight(0, 38)
        adv.setRowMinimumHeight(1, 38)
        self.device = QComboBox()
        self.device.addItem(tr('אוטומטי · GPU 0 אם זמין'), 0)
        self.device.addItem(tr('CPU · מעבד בלבד'), -1)
        self.device.addItem(tr('כרטיס גרפי לפי מספר'), -2)
        self.device.currentIndexChanged.connect(lambda: self.gpu.setEnabled(self.device.currentData() == -2))
        self.device.setMinimumHeight(36)
        self.gpu = QSpinBox()
        self.gpu.setRange(0, 63)
        self.gpu.setEnabled(False)
        self.gpu.setMinimumHeight(36)
        self.gpu.setPrefix('GPU ')
        self.gpu.setLayoutDirection(Qt.LeftToRight)
        self.suffix = QLineEdit('.beats')
        self.suffix.setMinimumHeight(36)
        self.suffix.setLayoutDirection(Qt.LeftToRight)
        adv.addWidget(label(tr('אמצעי חישוב')), 0, 0)
        adv.addWidget(self.device, 0, 1)
        adv.addWidget(self.gpu, 0, 2)
        adv.addWidget(label(tr('סיומת פלט')), 1, 0)
        adv.addWidget(self.suffix, 1, 1, 1, 2)
        self.checks = {}
        definitions = [
            ('append', tr('הוספת סיומת לשם הקיים'), tr('למשל song.mp3.beats במקום song.beats.')),
            ('skip_existing', tr('דילוג על פלט שכבר קיים'), tr('קבצים שכבר נותחו לא ייכתבו מחדש.')),
            ('touch_first', tr('יצירת קובץ פלט ריק מראש'), tr('אפשרות התוכנה לעבודה במקביל. מומלץ לשלב עם דילוג על פלט קיים.')),
            ('dbn', tr('עיבוד נוסף באמצעות DBN'), tr('עיבוד זמנים נוסף באמצעות madmom. כלול בחבילת Windows.')),
            ('float16', tr('דיוק מופחת · Float16'), tr('אפשרות האצה בכרטיסים תומכים; אם נכשל, בטלו אפשרות זו.')),
            ('activations', tr('שמירת נתונים גולמיים · NPY'), tr('שומר גם קובץ numpy של תחזיות המודל.')),
        ]
        for index, (key, title, hint) in enumerate(definitions):
            box = QCheckBox(title)
            box.setMinimumHeight(30)
            box.setToolTip(hint)
            self.checks[key] = box
            adv.addWidget(box, 2 + index // 2, (index % 2) * 2, 1, 2)
            adv.setRowMinimumHeight(2 + index // 2, 30)
        self.advanced.hide()
        options.addWidget(self.advanced)
        self.download_hint = label(tr('בהפעלה הראשונה המודל יורד מהאינטרנט. הפלט כולל זמני פעימות; BPM הוא אומדן מחושב.'), 'hint')
        self.download_hint.setText(tr('התצורות הרגילה והמהירה כלולות בחבילת Windows. הפלט מכיל פעימות ו־BPM משוער.'))
        self.download_hint.setWordWrap(True)
        options.addWidget(self.download_hint)
        layout.addWidget(self.options_card)

        self.status = label(tr('מוכן · הוסיפו קבצים כדי להתחיל'))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        bottom = QHBoxLayout()
        self.start_btn = button(tr('התחלת ניתוח'), self.start, True)
        self.start_btn.setMinimumWidth(165)
        self.start_btn.setEnabled(False)
        self.stop_btn = button(tr('עצירה'), self.cancel)
        self.stop_btn.setEnabled(False)
        self.open_btn = button(tr('פתיחת תיקיית הפלט'), self.open_output)
        self.open_btn.setEnabled(False)
        self.tempo_btn.setEnabled(False)
        self.log_btn = button(tr('פרטי ריצה'), self.toggle_log)
        self.diagnostics_btn = button(tr('בדיקת תקינות ותיקון'), self.diagnostics)
        bottom.addWidget(self.start_btn)
        bottom.addWidget(self.stop_btn)
        bottom.addStretch()
        bottom.addWidget(self.open_btn)
        bottom.addWidget(self.diagnostics_btn)
        bottom.addWidget(self.log_btn)
        layout.addLayout(bottom)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLayoutDirection(Qt.LeftToRight)
        self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(140)
        self.log.hide()
        layout.addWidget(self.log)

    def _restore(self):
        self.select_model(self.settings.value('model', 'final0'))
        self.output.setText(self.settings.value('output', ''))
        self.output_mode.setCurrentIndex(int(self.settings.value('output_mode', 0)))
        self.suffix.setText(self.settings.value('suffix', '.beats'))
        self.device.setCurrentIndex(int(self.settings.value('device', 0)))
        self.gpu.setValue(int(self.settings.value('gpu', 0)))
        self.use_cache.setChecked(self.settings.value('use_cache', True, type=bool))
        for key, check in self.checks.items():
            check.setChecked(self.settings.value(key, False, type=bool))
        self._output_mode_changed()

    def _save(self):
        for key, value in [('model', self.model.currentData()), ('output', self.output.text()),
                           ('output_mode', self.output_mode.currentIndex()), ('suffix', self.suffix.text()),
                           ('device', self.device.currentIndex()), ('gpu', self.gpu.value())]:
            self.settings.setValue(key, value)
        self.settings.setValue('use_cache', self.use_cache.isChecked())
        for key, check in self.checks.items():
            self.settings.setValue(key, check.isChecked())

    def toggle_advanced(self):
        visible = not self.advanced.isVisible()
        self.advanced.setVisible(visible)
        self.advanced_btn.setText(tr('הגדרות מתקדמות  −') if visible else tr('הגדרות מתקדמות  +'))
        if visible:
            available = self.screen().availableGeometry().height() - 70
            self.resize(self.width(), min(max(self.height(), 960), available))

    def toggle_log(self):
        self.log.setVisible(not self.log.isVisible())

    def _output_mode_changed(self):
        enabled = self.output_mode.currentIndex() > 0
        self.output.setEnabled(enabled)
        self.output_browse.setEnabled(enabled)

    def _last_dir(self):
        return self.settings.value('last_dir', str(Path.home() / 'Music'))

    def add_files(self):
        filters = tr('קובצי שמע (') + ' '.join('*' + ext for ext in sorted(AUDIO_EXTENSIONS)) + tr(');;כל הקבצים (*)')
        files, _ = QFileDialog.getOpenFileNames(self, tr('בחירת קובצי שמע'), self._last_dir(), filters)
        self.add_paths(files)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr('בחירת תיקיית מוזיקה'), self._last_dir())
        if folder:
            self.add_paths([folder])

    def choose_model(self):
        dialog = ModelDialog(self.model.currentData() or 'final0', self)
        if dialog.exec() and dialog.selection:
            self.select_model(dialog.selection)
        self.update_model_hint()

    def select_model(self, key):
        index = self.model.findData(key)
        if index < 0:
            self.model.addItem(model_label(key), key)
            index = self.model.count() - 1
        self.model.setCurrentIndex(index)
        self.update_model_hint()

    def update_model_hint(self):
        if not hasattr(self, 'model_hint'):
            return
        key = self.model.currentData()
        if not key:
            return
        ready = tr('זמין במחשב') if find_model(key) else tr('יורד אוטומטית בניתוח הראשון')
        info = MODEL_CATALOG.get(key)
        self.model_hint.setText(ready + (' · ' + info.description.split('.')[0] if info else tr(' · מודל אישי')))

    def diagnostics(self):
        DiagnosticsDialog(self.model.currentData(), self).exec()

    def open_history(self):
        if self.process:
            return
        dialog = CacheDialog(self)
        if dialog.exec() and dialog.selected_source:
            self.add_paths([dialog.selected_source])

    def choose_output(self):
        if self.output_mode.currentIndex() == 2:
            value, _ = QFileDialog.getSaveFileName(self, tr('שמירת פלט'), self.output.text(), tr('Beats (*.beats);;כל הקבצים (*)'))
        else:
            value = QFileDialog.getExistingDirectory(self, tr('תיקיית פלט'), self.output.text() or self._last_dir())
        if value:
            self.output.setText(value)

    def add_paths(self, paths):
        if self.process:
            return
        for value in paths:
            path = str(Path(value).resolve())
            if Path(path).exists() and path not in self.inputs:
                self.inputs.append(path)
                folder = Path(path) if Path(path).is_dir() else Path(path).parent
                self.settings.setValue('last_dir', str(folder))
        self._render_inputs()

    def _render_inputs(self):
        self.tasks = []
        self.results = {}
        self.open_btn.setEnabled(False)
        self.tempo_btn.setEnabled(False)
        self.table.setRowCount(len(self.inputs))
        for row, path in enumerate(self.inputs):
            self._fill_row(row, path, tr('תיקייה · כולל תתי־תיקיות') if Path(path).is_dir() else tr('ממתין'))
        self.count.setText(tr('{0} פריטים נבחרו', len(self.inputs)) if self.inputs else tr('לא נבחרו קבצים'))
        self.start_btn.setEnabled(bool(self.inputs))
        self.reanalyze_btn.setEnabled(bool(self.inputs))
        self.status.setText(tr('מוכן לניתוח') if self.inputs else tr('מוכן · הוסיפו קבצים כדי להתחיל'))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def _fill_row(self, row, path, state):
        self.table.setItem(row, 0, QTableWidgetItem(Path(path).name))
        self.table.item(row, 0).setToolTip(path)
        for column, text in enumerate([state, '—', '—', '—'], 1):
            self.table.setItem(row, column, QTableWidgetItem(text))

    def remove_selected(self):
        selected = {item.row() for item in self.table.selectedItems()}
        if not selected:
            return
        if self.tasks:
            self.inputs = [task['source'] for index, task in enumerate(self.tasks) if index not in selected]
            self._render_inputs()
            return
        self.inputs = [p for i, p in enumerate(self.inputs) if i not in selected]
        self._render_inputs()

    def clear_inputs(self):
        self.inputs = []
        self._render_inputs()

    def get_options(self):
        gpu = self.gpu.value() if self.device.currentData() == -2 else self.device.currentData()
        return Options(model=self.model.currentData() or 'final0',
                       output=self.output.text().strip() if self.output_mode.currentIndex() else '',
                       output_is_file=self.output_mode.currentIndex() == 2,
                       suffix=self.suffix.text().strip(), gpu=gpu, use_cache=self.use_cache.isChecked(),
                       **{key: check.isChecked() for key, check in self.checks.items()})

    def _busy(self, busy):
        self.language.setEnabled(not busy)
        self.options_card.setEnabled(not busy)
        self.diagnostics_btn.setEnabled(not busy)
        for item in (self.add_files_btn, self.add_folder_btn, self.remove_btn, self.clear_btn):
            item.setEnabled(not busy)
        self.start_btn.setEnabled(not busy and bool(self.inputs))
        self.reanalyze_btn.setEnabled(not busy and bool(self.inputs))
        self.stop_btn.setEnabled(busy)
        self.tempo_btn.setEnabled(not busy and any(r.get('status') in ('success', 'cached', 'skipped') for r in self.results.values()))

    def change_language(self):
        selected = self.language.currentData()
        if selected == LANGUAGE or self.process:
            return
        try:
            self._save()
            save_language(selected)
            self.settings.setValue('language_restart_inputs', list(self.inputs))
            self.settings.sync()
            arguments = [] if getattr(sys, 'frozen', False) else ['-m', 'beat_this.desktop']
            restart = QProcess()
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert('BPM_FINDER_LANGUAGE', selected)
            restart.setProcessEnvironment(environment)
            restart.setProgram(sys.executable)
            restart.setArguments(arguments)
            restart.setWorkingDirectory(str(ROOT))
            started = restart.startDetached()
            if not started:
                self.settings.remove('language_restart_inputs')
                QMessageBox.warning(self, tr('לא ניתן לשנות שפה'), tr('לא ניתן להפעיל מחדש את התוכנה. השפה החדשה תחול בפתיחה הבאה.'))
                return
            self.close()
        except OSError as exc:
            self.settings.remove('language_restart_inputs')
            self.language.blockSignals(True)
            self.language.setCurrentIndex(self.language.findData(LANGUAGE))
            self.language.blockSignals(False)
            QMessageBox.warning(self, tr('לא ניתן לשנות שפה'), str(exc))

    def start(self):
        self._start(False)

    def reanalyze(self):
        self._start(True)

    def _start(self, force):
        if self.process:
            return
        try:
            options = self.get_options()
            options.force_reanalyze = force
            if force:
                options.skip_existing = False
            if self.output_mode.currentIndex() and not options.output:
                raise ValueError(tr('יש לבחור מיקום שמירה.'))
            tasks = build_tasks(self.inputs, options)
        except ValueError as exc:
            QMessageBox.warning(self, tr('בדיקת הגדרות'), str(exc))
            return
        self._save()
        self.tasks = [dict(source=str(t.source), output=str(t.output), skip=t.skip) for t in tasks]
        self.results = {}
        self.cleanup_files = {}
        self.run_warning = ''
        self.cache_warning = ''
        self.table.setRowCount(len(tasks))
        for row, task in enumerate(tasks):
            self._fill_row(row, str(task.source), tr('ממתין'))
        self.count.setText(tr('{0} קובצי שמע', len(tasks)))
        self.cancelled = self.got_done = False
        self.completed_count = 0
        self.buffer = ''
        self.log.clear()
        self.open_btn.setEnabled(False)
        self.status.setText(tr('מכין ניתוח חדש…') if force else tr('בודק תוצאות שמורות ומכין ניתוח…'))
        self.progress.setRange(0, 0)
        self._busy(True)
        self.process = JsonWorker(self)
        self.process.event_received.connect(self._event)
        self.process.log_received.connect(lambda text: self.log.appendPlainText(text.rstrip()))
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.process.start_request(dict(inputs=self.inputs, options=asdict(options)))

    def _event(self, event):
        name = event['event']
        if name == 'plan':
            self.tasks = event['tasks']
            self.table.setRowCount(len(self.tasks))
            for row, task in enumerate(self.tasks):
                self._fill_row(row, task['source'], tr('ממתין'))
            self.count.setText(tr('{0} קובצי שמע', len(self.tasks)))
        elif name == 'loading':
            device = event['device']
            self.status.setText(tr('טוען מודל · {0} · הורדת המודל בפעם הראשונה עשויה להימשך כמה דקות', device))
        elif name == 'cache_check':
            self.status.setText(event['message'])
            if 'index' in event:
                self.table.item(event['index'], 1).setText(tr('בודק מטמון…'))
        elif name == 'cache_miss':
            self.table.item(event['index'], 1).setText(tr('בניתוח…'))
        elif name == 'download':
            done, total = event.get('done', 0), event.get('total', 0)
            self.status.setText(tr('מוריד את תצורת הזיהוי… {0:.1f} MB', done / 1024 ** 2))
            self.progress.setRange(0, total or 0)
            self.progress.setValue(done)
        elif name == 'processing':
            row = event['index']
            self.status.setText(tr('מנתח {0} מתוך {1} · {2}', row + 1, len(self.tasks), Path(event['source']).name))
            self.table.item(row, 1).setText(tr('בניתוח…'))
            self.table.scrollToItem(self.table.item(row, 0))
            self.progress.setRange(0, len(self.tasks))
        elif name == 'result':
            row = event['index']
            self.results[row] = event
            states = {'success': tr('הושלם'), 'cached': tr('נטען מהמטמון'), 'skipped': tr('דולג · פלט קיים'), 'error': tr('שגיאה')}
            self.table.item(row, 1).setText(states[event['status']])
            if event['status'] == 'error':
                self.table.item(row, 1).setToolTip(event.get('error', ''))
                self.log.appendPlainText(f'{self.tasks[row]["source"]}\n{event.get("error", "")}')
            if event['status'] in ('success', 'cached'):
                values = [f'{event["bpm"]:.2f}' if event.get('bpm') else '—', str(event['beats']), str(event['downbeats'])]
                for col, text in enumerate(values, 2):
                    self.table.item(row, col).setText(text)
                self.open_btn.setEnabled(True)
            elif event['status'] == 'skipped':
                self.open_btn.setEnabled(True)
            self.table.item(row, 0).setToolTip(self.tasks[row]['source'] + '\n→ ' + event.get('output', ''))
            self.completed_count += 1
            self.progress.setRange(0, len(self.tasks))
            self.progress.setValue(self.completed_count)
        elif name == 'cleanup':
            self.cleanup_files[event['path']] = event.get('empty', False)
        elif name == 'cleaned':
            self.cleanup_files.pop(event['path'], None)
        elif name == 'warning':
            self.run_warning = event['message']
            self.status.setText(event['message'])
            self.log.appendPlainText(event['message'])
        elif name == 'cache_warning':
            self.cache_warning = event['message']
            self.status.setText(event['message'])
            self.log.appendPlainText(event['message'])
        elif name == 'done':
            self.got_done = True
            self.status.setText(tr('סיום · {0} הושלמו · {1} דולגו · {2} נכשלו', event['success'], event['skipped'], event['failed']))
            if event.get('cached'):
                self.status.setText(self.status.text() + tr(' · {0} מהמטמון', event['cached']))
            if self.run_warning:
                self.status.setText(self.status.text() + tr(' · בוצע באמצעות CPU'))
            if self.cache_warning:
                self.status.setText(self.status.text() + tr(' · המטמון דורש בדיקה בפרטי הריצה'))
                self.log.show()
            if event['failed']:
                self.log.show()
        elif name == 'fatal':
            self.status.setText(tr('הניתוח לא התחיל · פירוט השגיאה מופיע למטה'))
            self.log.appendPlainText(event.get('error', tr('שגיאה לא ידועה')))
            self.log.show()
            self.got_done = True

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self.status.setText(tr('לא ניתן להפעיל את מנוע הניתוח. בדקו את התקנת Python.'))
            self.log.appendPlainText(self.process.errorString())
            self.log.show()
            self._release_process()

    def _finished(self, exit_code, exit_status):
        for path, empty_only in self.cleanup_files.items():
            try:
                file = Path(path)
                if file.exists() and (not empty_only or file.stat().st_size == 0):
                    file.unlink()
            except OSError as exc:
                self.log.appendPlainText(str(exc))
        if self.cancelled:
            self.status.setText(tr('הניתוח נעצר · {0} מתוך {1} קבצים טופלו', self.completed_count, len(self.tasks)))
            for row in range(len(self.tasks)):
                if row not in self.results:
                    self.table.item(row, 1).setText(tr('נעצר'))
        elif not self.got_done:
            self.status.setText(tr('מנוע הניתוח הסתיים באופן לא צפוי · ראו פרטי ריצה'))
            self.log.show()
        self.progress.setRange(0, max(1, len(self.tasks)))
        self.progress.setValue(self.completed_count)
        self._release_process()

    def _release_process(self):
        process, self.process = self.process, None
        if process:
            process.deleteLater()
        self._busy(False)

    def cancel(self):
        if self.process:
            self.cancelled = True
            self.status.setText(tr('עוצר ניתוח…'))
            self.stop_btn.setEnabled(False)
            self.process.kill()

    def open_result(self, row, column):
        self.show_tempo(row)

    def open_tempo(self):
        row = self.table.currentRow()
        if row < 0:
            row = next((i for i, r in self.results.items() if r.get('status') in ('success', 'cached', 'skipped')), -1)
        self.show_tempo(row)

    def show_tempo(self, row):
        if self.process:
            return
        result = self.results.get(row)
        if result and result.get('status') in ('success', 'cached', 'skipped'):
            try:
                dialog = TempoDialog(result['output'], self.tasks[row]['source'], self)
                dialog.exec()
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, tr('לא ניתן להציג גרף קצב'), str(exc))
        else:
            QMessageBox.information(self, tr('בחירת תוצאה'), tr('בחרו שורת קובץ שנותח בהצלחה או דולג כי כבר יש לו פלט.'))

    def open_output(self):
        for result in self.results.values():
            if result.get('status') in ('success', 'cached', 'skipped'):
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(result['output']).parent)))
                return

    def dragEnterEvent(self, event):
        if not self.process and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_paths([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def closeEvent(self, event):
        if self.process:
            answer = QMessageBox.question(self, tr('סגירת התוכנה'), tr('ניתוח עדיין פועל. לעצור ולסגור?'),
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.cancel()
            self.process.waitForFinished(3000)
        self._save()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('BPM Finder')
    app.setLayoutDirection(Qt.RightToLeft if LANGUAGE == 'he' else Qt.LeftToRight)
    if LANGUAGE == 'he':
        translator = QTranslator(app)
        if translator.load('qtbase_he', QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
            app.installTranslator(translator)
        app.language_translator = translator
    app.setStyle('Fusion')
    app.setFont(QFont('Segoe UI', 10))
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
