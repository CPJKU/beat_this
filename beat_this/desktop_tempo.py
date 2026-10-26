"""Interactive native tempo timeline and timestamped change report."""
from beat_this.desktop_i18n import tr
from bisect import bisect_left
import math
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QFont
from PySide6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QScrollArea, QFrame, QLayout)

from beat_this.tempo_analysis import (TempoSettings, analyze_tempo, load_beats,
    export_curve_csv, export_changes_csv)

KINDS = {'abrupt': tr('שינוי חד'), 'acceleration': tr('האצה הדרגתית'), 'deceleration': tr('האטה הדרגתית')}
COLORS = {'abrupt': '#bb6412', 'acceleration': '#218269', 'deceleration': '#b64261'}
PRESETS = [TempoSettings(), TempoSettings(3, 1.5, 1.5, 3), TempoSettings(8, 5, 5, 6)]


def time_label(seconds, precise=False):
    seconds = max(0, seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(int(minutes), 60)
    tail = f'{sec:04.1f}' if precise else f'{int(sec):02d}'
    return (f'{hours:02d}:' if hours else '') + f'{minute:02d}:{tail}'


class TempoChart(QWidget):
    cursor_changed = Signal(float, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis = None
        self.start, self.end = 0.0, 1.0
        self.cursor = None
        self.highlight = None
        self.setMinimumHeight(280)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(tr('גרף BPM לאורך השיר'))
        self.setAccessibleDescription(tr('זמן משמאל לימין. רשימת כל שינויי הקצב זמינה גם בטבלה. גלגלת לזום, לחיצה כפולה לכל השיר, פלוס ומינוס לזום מהמקלדת.'))
        self.setToolTip(tr('גלגלת: התקרבות והתרחקות · לחיצה כפולה: כל השיר · בחירת שינוי בטבלה: התמקדות בזמן השינוי'))

    def set_analysis(self, analysis):
        self.analysis = analysis
        self.times = [p.time for p in analysis.points]
        self.highlight = None
        self.show_all()

    def show_all(self):
        self.start, self.end = 0.0, max(1.0, self.analysis.duration if self.analysis else 1.0)
        self.update()

    def focus_change(self, index):
        if not self.analysis or not 0 <= index < len(self.analysis.changes):
            return
        event = self.analysis.changes[index]
        self.highlight = index
        self.start = max(0, event.start - 8)
        self.end = min(max(self.analysis.duration, 1), event.end + 8)
        self.update()

    def plot_rect(self):
        return QRectF(66, 72, max(1, self.width() - 94), max(1, self.height() - 122))

    def x(self, time):
        rect = self.plot_rect()
        return rect.left() + (time - self.start) / max(.001, self.end - self.start) * rect.width()

    def at_x(self, x):
        rect = self.plot_rect()
        return self.start + max(0, min(1, (x - rect.left()) / rect.width())) * (self.end - self.start)

    def bpm_at(self, time):
        if not self.analysis or not self.times:
            return None
        index = bisect_left(self.times, time)
        candidates = [self.analysis.points[i] for i in (index - 1, index) if 0 <= i < len(self.times)]
        nearest = min(candidates, key=lambda p: abs(p.time - time))
        return nearest.bpm

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#ffffff'))
        painter.setPen(QColor('#dce2ea'))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 8, 8)
        rect = self.plot_rect()
        painter.setPen(QColor('#24364e'))
        painter.setFont(QFont('Segoe UI', 11, QFont.Bold))
        painter.drawText(QRectF(16, 10, self.width() - 32, 23), Qt.AlignRight, tr('BPM לאורך השיר'))
        painter.setFont(QFont('Segoe UI', 9))
        legend = tr('קו כחול: קצב מוחלק   |   כתום: שינוי חד   |   ירוק: האצה   |   אדום: האטה')
        painter.setPen(QColor('#58677b'))
        painter.drawText(QRectF(16, 36, self.width() - 32, 22), Qt.AlignRight, legend)
        painter.setFont(QFont('Segoe UI', 9))
        visible = [p.bpm for p in self.analysis.points if p.bpm is not None and self.start <= p.time <= self.end] if self.analysis else []
        if not visible:
            painter.drawText(rect, Qt.AlignCenter, tr('אין מספיק פעימות להצגת קצב בקטע הזה'))
            return
        low, high = min(visible), max(visible)
        padding = max(2, (high - low) * .14)
        low, high = max(0, low - padding), high + padding
        def y(value):
            return rect.bottom() - (value - low) / max(1, high - low) * rect.height()
        painter.setPen(QColor('#e6ebf2'))
        for index in range(5):
            value = low + (high - low) * index / 4
            pos = y(value)
            painter.drawLine(QPointF(rect.left(), pos), QPointF(rect.right(), pos))
            painter.setPen(QColor('#58677b'))
            painter.drawText(QRectF(2, pos - 9, 56, 18), Qt.AlignRight | Qt.AlignVCenter, f'{value:.1f}')
            painter.setPen(QColor('#e6ebf2'))
        for index in range(7):
            time = self.start + (self.end - self.start) * index / 6
            pos = self.x(time)
            painter.drawLine(QPointF(pos, rect.top()), QPointF(pos, rect.bottom()))
            painter.setPen(QColor('#58677b'))
            painter.drawText(QRectF(pos - 42, rect.bottom() + 8, 84, 20), Qt.AlignCenter, time_label(time))
            painter.setPen(QColor('#e6ebf2'))
        painter.setPen(QColor('#58677b'))
        painter.drawText(QRectF(8, rect.top() - 24, 48, 20), Qt.AlignCenter, 'BPM')
        painter.save()
        painter.setClipRect(rect)
        for index, change in enumerate(self.analysis.changes):
            if change.end < self.start or change.start > self.end:
                continue
            color = QColor(COLORS[change.kind])
            if change.kind != 'abrupt':
                shade = QColor(color)
                shade.setAlpha(27 if self.highlight != index else 50)
                painter.fillRect(QRectF(self.x(change.start), rect.top(), max(1, self.x(change.end) - self.x(change.start)), rect.height()), shade)
            painter.setPen(QPen(color, 2.5 if self.highlight == index else 1.3, Qt.DashLine))
            painter.drawLine(QPointF(self.x(change.start), rect.top()), QPointF(self.x(change.start), rect.bottom()))
            if change.end != change.start:
                painter.drawLine(QPointF(self.x(change.end), rect.top()), QPointF(self.x(change.end), rect.bottom()))
        path = QPainterPath()
        drawing = False
        for point in self.analysis.points:
            if point.bpm is None:
                drawing = False
                continue
            pos = QPointF(self.x(point.time), y(point.bpm))
            if drawing:
                path.lineTo(pos)
            else:
                path.moveTo(pos)
                drawing = True
        painter.setPen(QPen(QColor('#245ac7'), 2.3))
        painter.drawPath(path)
        if self.cursor is not None and self.start <= self.cursor <= self.end:
            painter.setPen(QPen(QColor('#52657f'), 1, Qt.DotLine))
            painter.drawLine(QPointF(self.x(self.cursor), rect.top()), QPointF(self.x(self.cursor), rect.bottom()))
        painter.restore()
        for index, change in enumerate(self.analysis.changes):
            if self.start <= change.start <= self.end:
                x = min(rect.right() - 10, max(rect.left() + 10, self.x(change.start)))
                painter.setBrush(QColor(COLORS[change.kind]))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(x, rect.top() + 11), 9, 9)
                painter.setPen(Qt.white)
                painter.drawText(QRectF(x - 9, rect.top() + 2, 18, 18), Qt.AlignCenter, str(index + 1))

    def mouseMoveEvent(self, event):
        if self.plot_rect().contains(event.position()):
            self.cursor = self.at_x(event.position().x())
            self.cursor_changed.emit(self.cursor, self.bpm_at(self.cursor))
            self.update()

    def mousePressEvent(self, event):
        self.setFocus()
        self.mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.show_all()

    def zoom(self, factor, center=None):
        if not self.analysis or self.analysis.duration <= 0:
            return
        center = (self.start + self.end) / 2 if center is None else center
        duration = max(1, self.analysis.duration)
        span = min(duration, max(min(5, duration), (self.end - self.start) * factor))
        fraction = (center - self.start) / max(.001, self.end - self.start)
        self.start = max(0, min(duration - span, center - span * fraction))
        self.end = self.start + span
        self.update()

    def wheelEvent(self, event):
        self.zoom(.8 if event.angleDelta().y() > 0 else 1.25, self.at_x(event.position().x()))
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom(.8)
        elif event.key() == Qt.Key_Minus:
            self.zoom(1.25)
        elif event.key() in (Qt.Key_Home, Qt.Key_0):
            self.show_all()
        else:
            super().keyPressEvent(event)


class TempoDialog(QDialog):
    def __init__(self, beat_path, source=None, parent=None, export_base=None):
        super().__init__(parent)
        self.beat_path = Path(beat_path)
        self.export_base = Path(export_base) if export_base is not None else self.beat_path
        self.beats = load_beats(beat_path)
        self.duration = None
        if source:
            try:
                import soundfile as sf
                self.duration = sf.info(str(source)).duration
            except Exception:
                pass
        self.setWindowTitle(tr('BPM Finder — גרף ושינויי קצב'))
        self.resize(1040, min(870, self.screen().availableGeometry().height() - 70))
        self.setMinimumSize(820, 650)
        self.build(Path(source).name if source else self.beat_path.name)
        self.recalculate()

    def build(self, name):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        root = QWidget()
        scroll.setWidget(root)
        layout = QVBoxLayout(root)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)
        title = QLabel(tr('גרף BPM ושינויי קצב'))
        title.setStyleSheet('font-size: 23px; font-weight: 700;')
        layout.addWidget(title)
        file = QLabel(name)
        file.setTextFormat(Qt.PlainText)
        file.setWordWrap(True)
        layout.addWidget(file)
        controls = QHBoxLayout()
        controls.addWidget(QLabel(tr('רגישות לזיהוי שינויים')))
        self.preset = QComboBox()
        self.preset.addItems([tr('רגילה'), tr('גבוהה — שינויים עדינים'), tr('יציבה — שינויים גדולים'), tr('הגדרה ידנית')])
        self.preset.currentIndexChanged.connect(self.apply_preset)
        controls.addWidget(self.preset)
        self.advanced_btn = QPushButton(tr('רגישות מתקדמת +'))
        self.advanced_btn.clicked.connect(self.toggle_settings)
        controls.addWidget(self.advanced_btn)
        controls.addStretch()
        all_song = QPushButton(tr('כל השיר'))
        all_song.clicked.connect(lambda: self.chart.show_all())
        controls.addWidget(all_song)
        layout.addLayout(controls)
        self.advanced = QWidget()
        grid = QGridLayout(self.advanced)
        grid.setContentsMargins(0, 0, 0, 0)
        self.spins = []
        definitions = [
            (tr('חלון החלקה (שניות)'), 2, 20, 4, tr('חלון ארוך מפחית תנודות; חלון קצר מציג שינויים מהירים יותר.')),
            (tr('שינוי מזערי (BPM)'), .5, 50, 3, tr('השינוי צריך לעבור גם סף זה וגם את הסף באחוזים.')),
            (tr('שינוי מזערי (%)'), 0, 30, 3, tr('שינוי יחסי בקצב, בנוסף לסף ה־BPM.')),
            (tr('זמן אישור (שניות)'), 1, 30, 4, tr('משך המשמש לאימות הקצב משני צדי השינוי ולסינון קפיצות קצרות.')),
        ]
        for index, (text, low, high, default, tooltip) in enumerate(definitions):
            row, col = divmod(index, 2)
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(.5)
            spin.setDecimals(1)
            spin.setValue(default)
            spin.setToolTip(tooltip)
            spin.valueChanged.connect(self.manual_settings)
            grid.addWidget(QLabel(text), row, col * 2)
            grid.addWidget(spin, row, col * 2 + 1)
            self.spins.append(spin)
        self.advanced.hide()
        layout.addWidget(self.advanced)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.stats.setStyleSheet('font-weight: 600; color: #345375;')
        layout.addWidget(self.stats)
        self.chart = TempoChart()
        self.chart.cursor_changed.connect(self.cursor)
        layout.addWidget(self.chart, 1)
        self.cursor_label = QLabel(tr('העבירו את העכבר על הגרף לקריאת הזמן וה־BPM. גלגלת לזום; לחיצה כפולה לכל השיר.'))
        self.cursor_label.setWordWrap(True)
        layout.addWidget(self.cursor_label)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['#', tr('סוג השינוי'), tr('התחלה'), tr('סיום'), tr('BPM לפני'), tr('BPM אחרי'), tr('הפרש')])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(130)
        self.table.setMaximumHeight(190)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.select_change)
        layout.addWidget(self.table)
        note = QLabel(tr('הזמנים והקצב הם אומדן לפי הפעימות שזוהו. שגיאות זיהוי, כפל או חצי קצב עשויים להשפיע. אזור ללא פעימות אינו מחובר בקו. שינוי הרגישות אינו דורש ניתוח מחדש של השמע.'))
        note.setStyleSheet('color: #58677b; font-size: 12px;')
        note.setWordWrap(True)
        layout.addWidget(note)
        exports = QHBoxLayout()
        for text, callback in [(tr('ייצוא שינויי קצב · CSV'), self.save_changes),
                               (tr('ייצוא עקומת BPM · CSV'), self.save_curve),
                               (tr('שמירת הגרף · PNG'), self.save_graph)]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            exports.addWidget(button)
        exports.addStretch()
        close = QPushButton(tr('סגירה'))
        close.clicked.connect(self.close)
        exports.addWidget(close)
        layout.addLayout(exports)
        self.export_status = QLabel('')
        self.export_status.setTextFormat(Qt.PlainText)
        self.export_status.setWordWrap(True)
        layout.addWidget(self.export_status)
        self.recalc_timer = QTimer(self)
        self.recalc_timer.setSingleShot(True)
        self.recalc_timer.setInterval(200)
        self.recalc_timer.timeout.connect(self.recalculate)

    def toggle_settings(self):
        visible = not self.advanced.isVisible()
        self.advanced.setVisible(visible)
        self.advanced_btn.setText(tr('רגישות מתקדמת −') if visible else tr('רגישות מתקדמת +'))

    def apply_preset(self, index):
        if index >= len(PRESETS) or not hasattr(self, 'spins'):
            return
        preset = PRESETS[index]
        for spin, value in zip(self.spins, (preset.window, preset.min_bpm, preset.min_percent, preset.confirmation)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.recalculate()

    def manual_settings(self):
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(3)
        self.preset.blockSignals(False)
        self.recalc_timer.start()

    def recalculate(self):
        self.analysis = analyze_tempo(self.beats, TempoSettings(*(spin.value() for spin in self.spins)), self.duration)
        self.chart.set_analysis(self.analysis)
        values = [p.bpm for p in self.analysis.points if p.bpm is not None]
        if values:
            self.stats.setText(tr('{0} פעימות · טווח קצב מוחלק: \u2066{1:.1f}–{2:.1f} BPM\u2069 · אורך ציר הזמן: {3}', len(self.beats), min(values), max(values), time_label(self.analysis.duration)))
        else:
            self.stats.setText(tr('אין מספיק פעימות ליצירת עקומת קצב. אפשר לנסות חלון החלקה ארוך יותר.'))
        changes = self.analysis.changes
        self.summary.setText(tr('{0} שינויי קצב משוערים · בחרו שורה להתמקדות באזור השינוי.', len(changes)) if changes else tr('לא נמצאו שינויי קצב שעוברים את ספי הרגישות שנבחרו. אפשר לבחור רגישות גבוהה יותר.'))
        self.table.setRowCount(len(changes))
        for row, event in enumerate(changes):
            items = [str(row + 1), KINDS[event.kind], time_label(event.start, True),
                     time_label(event.end, True), f'{event.before:.2f}', f'{event.after:.2f}', f'\u2066{event.delta:+.2f}\u2069']
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                if col != 1:
                    item.setTextAlignment(Qt.AlignCenter)
                item.setToolTip(tr('{0} · {1:.3f}–{2:.3f} שניות', KINDS[event.kind], event.start, event.end))
                self.table.setItem(row, col, item)
        self.cursor_label.setText(tr('העבירו את העכבר על הגרף לקריאת הזמן וה־BPM. גלגלת לזום; לחיצה כפולה לכל השיר.'))

    def cursor(self, time, bpm):
        self.cursor_label.setText(tr('זמן: {0} · ', time_label(time, True)) + (f'{bpm:.2f} BPM' if bpm is not None else tr('אין אומדן קצב באזור הזה')))

    def select_change(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.analysis.changes):
            self.chart.focus_change(row)

    def save(self, suffix, extension, operation):
        default = self.export_base.with_name(self.export_base.stem + suffix + extension)
        path, _ = QFileDialog.getSaveFileName(self, tr('שמירת תוצאות'), str(default), f'{extension[1:].upper()} (*{extension})')
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != extension:
            target = target.with_name(target.name + extension)
            if target.exists() and QMessageBox.question(self, tr('קובץ קיים'), tr('הקובץ כבר קיים. להחליף אותו?'), QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        try:
            operation(target)
            self.export_status.setText(tr('נשמר: {0}', target))
        except Exception as exc:
            QMessageBox.warning(self, tr('לא ניתן לשמור'), str(exc))

    def save_changes(self):
        self.save('-tempo-changes', '.csv', lambda path: export_changes_csv(self.analysis, path))

    def save_curve(self):
        self.save('-tempo-curve', '.csv', lambda path: export_curve_csv(self.analysis, path))

    def save_graph(self):
        def write(path):
            if not self.chart.grab().save(str(path), 'PNG'):
                raise OSError(tr('שמירת תמונת הגרף נכשלה.'))
        self.save('-tempo-chart', '.png', write)
