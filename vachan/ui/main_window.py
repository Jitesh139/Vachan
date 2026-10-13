"""Operator control window: live transcript, suggestion approval, search, history, debug."""

from __future__ import annotations

import html
import logging
import os
import threading
from dataclasses import asdict
from typing import Optional

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QLinearGradient, QPainter, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from vachan.audio.source import LiveAudioSource, list_input_devices
from vachan.core.approval import describe_warning, should_auto_show
from vachan.core.models import ChunkDebugInfo, Suggestion
from vachan.core.paths import LOGS_DIR
from vachan.core.pipeline import PipelineEvents, PipelineWorker
from vachan.core.settings import Settings
from vachan.output.verse_window import THEMES, VerseDisplayWindow
from vachan.ui import theme
from vachan.ui.settings_dialog import STT_FIELDS, SettingsDialog

log = logging.getLogger(__name__)
HINT = (
    "Say e.g. “Turn to John chapter 3 verse 16”, “यूहन्ना 3 अध्याय 16 पद” or "
    "“Romans ke aath adhyay mein 28 pad”, or type in Find a verse."
)


class Bridge(QObject):
    """Moves pipeline callbacks (worker thread) onto the GUI thread."""

    status = Signal(str)
    transcript = Signal(str, str)
    suggestion = Signal(object)
    error = Signal(str)
    debug = Signal(object)


class LevelMeter(QWidget):
    """Rounded segmented audio meter."""

    def __init__(self) -> None:
        super().__init__()
        self.level = 0.0
        self.setFixedSize(120, 10)
        self.setToolTip("Input level")

    def set_level(self, level: float) -> None:
        """Update and repaint if changed."""
        if abs(level - self.level) > 0.005:
            self.level = level
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """Track + gradient fill."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.SURFACE_2))
        p.drawRoundedRect(r, 5, 5)
        if self.level > 0:
            grad = QLinearGradient(r.topLeft(), r.topRight())
            grad.setColorAt(0, QColor(theme.GREEN))
            grad.setColorAt(0.75, QColor(theme.AMBER))
            grad.setColorAt(1, QColor(theme.RED))
            p.setBrush(grad)
            p.drawRoundedRect(QRectF(r.left(), r.top(), r.width() * self.level, r.height()), 5, 5)


def _label(text: str = "", name: str = "", wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    return lbl


def _card(name: str = "Card") -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName(name)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(10)
    return frame, layout


class ControlWindow(QMainWindow):
    """Main operator dashboard."""

    def __init__(
        self, settings, transcriber, parser, bible, output: VerseDisplayWindow, startup_error: str = ""
    ):
        super().__init__()
        self.settings: Settings = settings
        self.transcriber, self.parser, self.bible, self.output = transcriber, parser, bible, output
        self.bridge = Bridge()
        self.audio = LiveAudioSource(settings.input_device, settings.input_gain)
        events = PipelineEvents(
            status=self.bridge.status.emit,
            transcript=self.bridge.transcript.emit,
            suggestion=self.bridge.suggestion.emit,
            error=self.bridge.error.emit,
            debug=self.bridge.debug.emit,
        )
        self.pipeline = (
            PipelineWorker(self.audio, transcriber, parser, bible, settings, events) if bible else None
        )
        self.pending: Optional[Suggestion] = None
        self.on_air: Optional[Suggestion] = None
        self._listening = False
        self._transcript_chars = 0

        self.setWindowTitle("Vachan - Operator")
        self.resize(1280, 800)
        self.setMinimumSize(980, 640)
        self._build()
        self._connect()
        self._set_suggestion(None)
        self.transcript.setFocus()
        if startup_error:
            self._show_error(startup_error)
        self._meter_timer = QTimer(
            self, interval=40, timeout=lambda: self.meter.set_level(self.audio.get_level())
        )
        self._meter_timer.start()
        if self.pipeline:
            threading.Thread(target=self._preload_model, daemon=True).start()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        body = QVBoxLayout()
        body.setContentsMargins(18, 14, 18, 10)
        body.setSpacing(12)
        self.banner, banner_layout = QFrame(), QHBoxLayout()
        self.banner.setObjectName("Banner")
        self.banner.setLayout(banner_layout)
        banner_layout.setContentsMargins(14, 8, 8, 8)
        self.banner_text = _label("", wrap=True)
        banner_close = QPushButton("Dismiss")
        banner_close.setObjectName("Ghost")
        banner_close.clicked.connect(self.banner.hide)
        banner_layout.addWidget(self.banner_text, 1)
        banner_layout.addWidget(banner_close)
        self.banner.hide()
        body.addWidget(self.banner)

        left = QSplitter(Qt.Orientation.Vertical)
        left.setChildrenCollapsible(False)
        left.setHandleWidth(12)
        left.addWidget(self._build_transcript())
        left.addWidget(self._build_debug())
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)
        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setFrameShape(QFrame.Shape.NoFrame)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right.setWidget(self._build_right())
        right.setMinimumWidth(440)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(12)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 4)
        body.addWidget(split, 1)
        outer.addLayout(body, 1)
        outer.addWidget(self._build_footer())
        self.setCentralWidget(root)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("Header")
        row = QHBoxLayout(header)
        row.setContentsMargins(20, 12, 20, 12)
        row.setSpacing(12)
        logo = _label("वचन", "Title")
        logo.setStyleSheet(f"color: {theme.ACCENT}; font-family: '{theme.SERIF_DEVA}'; font-size: 18pt;")
        row.addWidget(logo)
        row.addWidget(_label("Vachan", "Title"))
        self.state_pill = _label("IDLE", "Badge")
        row.addSpacing(6)
        row.addWidget(self.state_pill)
        row.addStretch(1)
        row.addWidget(_label("Input", "Muted"))
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(240)
        self.mic_combo.setMaximumWidth(340)
        self._fill_devices()
        row.addWidget(self.mic_combo)
        self.meter = LevelMeter()
        row.addWidget(self.meter)
        row.addSpacing(6)
        self.btn_output = QPushButton("Output window")
        self.btn_output.setToolTip("Show / hide the broadcast window (F9)")
        row.addWidget(self.btn_output)
        self.btn_settings = QPushButton("Settings")
        row.addWidget(self.btn_settings)
        self.btn_listen = QPushButton("Start listening")
        self.btn_listen.setObjectName("Start")
        self.btn_listen.setToolTip("Start / stop speech recognition (F5)")
        self.btn_listen.setEnabled(self.pipeline is not None)
        row.addWidget(self.btn_listen)
        return header

    def _build_transcript(self) -> QFrame:
        card, lay = _card()
        top = QHBoxLayout()
        top.addWidget(_label("LIVE TRANSCRIPT", "Section"))
        top.addStretch(1)
        self.lang_label = _label("", "Muted")
        top.addWidget(self.lang_label)
        clear = QPushButton("Clear")
        clear.setObjectName("Ghost")
        clear.clicked.connect(self._clear_transcript)
        top.addWidget(clear)
        lay.addLayout(top)
        self.transcript = QTextBrowser()
        self.transcript.document().setMaximumBlockCount(400)  # bounded memory for long services
        self.transcript.setStyleSheet(
            f"font-family: '{theme.UI_FONT}', '{theme.SANS_DEVA}'; font-size: 13pt;"
        )
        self.transcript.setPlaceholderText(
            "Press Start listening. The preacher's words appear here, with detected references highlighted."
        )
        lay.addWidget(self.transcript, 1)
        return card

    def _build_right(self) -> QWidget:
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        self.sug_card, s = _card("Suggestion")
        head = QHBoxLayout()
        head.addWidget(_label("SUGGESTION", "Section"))
        head.addStretch(1)
        self.conf_badge = _label("", "Badge")
        head.addWidget(self.conf_badge)
        s.addLayout(head)
        self.sug_ref = _label("", "RefBig")
        s.addWidget(self.sug_ref)
        self.sug_text = _label("", "VerseText", wrap=True)
        self.sug_text_hi = _label("", "VerseTextHi", wrap=True)
        for lbl in (self.sug_text, self.sug_text_hi):
            lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        s.addWidget(self.sug_text)
        s.addWidget(self.sug_text_hi)
        self.sug_warn = _label("", "Warning", wrap=True)
        s.addWidget(self.sug_warn)
        btns = QHBoxLayout()
        self.btn_show = QPushButton("Show on screen   Enter")
        self.btn_show.setObjectName("Primary")
        self.btn_dismiss = QPushButton("Dismiss   Esc")
        btns.addWidget(self.btn_show, 3)
        btns.addWidget(self.btn_dismiss, 2)
        s.addLayout(btns)
        lay.addWidget(self.sug_card)

        air, a = _card()
        row = QHBoxLayout()
        dot = _label("●")
        dot.setStyleSheet(f"color: {theme.RED}; font-size: 9pt;")
        row.addWidget(dot)
        row.addWidget(_label("ON SCREEN", "Section"))
        self.air_label = _label("Nothing", "Muted")
        row.addWidget(self.air_label, 1)
        self.btn_clear_air = QPushButton("Clear screen   F8")
        row.addWidget(self.btn_clear_air)
        a.addLayout(row)
        lay.addWidget(air)

        ctl, c = _card()
        c.addWidget(_label("FIND A VERSE", "Section"))
        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("John 3:16  ·  यूहन्ना 3:16  ·  Rom 8 28      (Ctrl+F)")
        self.btn_search = QPushButton("Find")
        search_row.addWidget(self.search, 1)
        search_row.addWidget(self.btn_search)
        c.addLayout(search_row)
        self.search_hint = _label("", "Muted")
        self.search_hint.hide()
        c.addWidget(self.search_hint)
        c.addSpacing(4)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        mode_row.addWidget(_label("Mode", "Muted"))
        mode_row.addSpacing(12)
        self.mode_group = QButtonGroup(self)
        for i, (text, tip) in enumerate(
            [("Manual approve", "Every verse waits for Enter (recommended)"),
             ("Auto-show", "Confident, warning-free verses go on screen immediately")]
        ):  # fmt: skip
            b = QPushButton(text)
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.setToolTip(tip)
            radius = (
                "border-top-left-radius: 8px; border-bottom-left-radius: 8px;"
                if i == 0
                else ("border-top-right-radius: 8px; border-bottom-right-radius: 8px;")
            )
            b.setStyleSheet(f"QPushButton#Seg {{ {radius} }}")
            self.mode_group.addButton(b, i)
            mode_row.addWidget(b)
        self.mode_group.button(1 if self.settings.auto_show else 0).setChecked(True)
        mode_row.addStretch(1)
        c.addLayout(mode_row)
        out_row = QHBoxLayout()
        out_row.addWidget(_label("Output", "Muted"))
        out_row.addSpacing(4)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(THEMES))
        self.theme_combo.setCurrentText(self.settings.theme)
        out_row.addWidget(self.theme_combo, 1)
        self.lang_combo = QComboBox()
        for label, value in [
            ("Both languages", "both"),
            ("Main only", "primary"),
            ("Second only", "secondary"),
        ]:
            self.lang_combo.addItem(label, value)
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(self.settings.display_mode)))
        out_row.addWidget(self.lang_combo, 1)
        c.addLayout(out_row)
        lay.addWidget(ctl)

        hist, h = _card()
        h.addWidget(_label("SHOWN THIS SERVICE", "Section"))
        self.history = QListWidget()
        self.history.setToolTip("Double-click to load a verse again")
        self.history.setMinimumHeight(150)
        h.addWidget(self.history, 1)
        lay.addWidget(hist, 1)
        return col

    def _build_debug(self) -> QFrame:
        self.debug_panel, d = _card()
        top = QHBoxLayout()
        title = _label("PIPELINE DEBUG", "Section")
        title.setToolTip("Every audio window: level, Whisper text, parser result. Also in logs/pipeline.log")
        top.addWidget(title)
        top.addStretch(1)
        self.chk_save_audio = QCheckBox("Save audio")
        self.chk_save_audio.setToolTip("Save the exact audio sent to Whisper to debug_audio/ (WAV)")
        self.chk_save_audio.setChecked(self.settings.save_debug_audio)
        top.addWidget(self.chk_save_audio)
        open_logs = QPushButton("Logs")
        open_logs.setToolTip("Open the logs folder")
        open_logs.setObjectName("Ghost")
        open_logs.clicked.connect(lambda: (LOGS_DIR.mkdir(exist_ok=True), os.startfile(LOGS_DIR)))
        top.addWidget(open_logs)
        d.addLayout(top)
        self.debug_log = QPlainTextEdit()
        self.debug_log.setReadOnly(True)
        self.debug_log.setMaximumBlockCount(600)
        self.debug_log.setMinimumHeight(90)
        self.debug_log.setStyleSheet(
            f"font-family: Consolas, '{theme.SANS_DEVA}'; font-size: 9pt;"
            " color: #A9C7A1; background: #0B0D12;"
        )
        d.addWidget(self.debug_log)
        self.debug_panel.hide()
        return self.debug_panel

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("Header")
        row = QHBoxLayout(footer)
        row.setContentsMargins(20, 6, 12, 6)
        self.status = _label("Ready", "Muted")
        row.addWidget(self.status, 1)
        self.engine_label = _label("", "Muted")
        row.addWidget(self.engine_label)
        self.btn_debug = QPushButton("Debug")
        self.btn_debug.setObjectName("Ghost")
        self.btn_debug.setCheckable(True)
        row.addWidget(self.btn_debug)
        return footer

    def _fill_devices(self) -> None:
        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()
        try:
            devices = list_input_devices()
        except Exception as e:
            log.exception("Device listing failed")
            devices = []
            self._show_error(f"Could not list audio devices: {e}")
        self.mic_combo.addItem("Windows default input", "")
        for d in devices:
            self.mic_combo.addItem(d["name"] + ("  (default)" if d["is_default"] else ""), d["name"])
        self.mic_combo.setCurrentIndex(max(0, self.mic_combo.findData(self.settings.input_device)))
        self.mic_combo.blockSignals(False)

    # ------------------------------------------------------------------ wiring
    def _connect(self) -> None:
        b = self.bridge
        b.status.connect(self._on_status)
        b.transcript.connect(self._on_transcript)
        b.suggestion.connect(self._on_suggestion)
        b.error.connect(self._show_error)
        b.debug.connect(self._on_debug)
        self.btn_listen.clicked.connect(self.toggle_listening)
        self.btn_output.clicked.connect(self.toggle_output)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_show.clicked.connect(self.approve)
        self.btn_dismiss.clicked.connect(self.dismiss)
        self.btn_clear_air.clicked.connect(self.clear_screen)
        self.btn_search.clicked.connect(self.manual_search)
        self.search.returnPressed.connect(self.manual_search)
        self.btn_debug.toggled.connect(self.debug_panel.setVisible)
        self.chk_save_audio.toggled.connect(lambda on: self._update_settings(save_debug_audio=on))
        self.mode_group.idToggled.connect(lambda i, on: on and self._update_settings(auto_show=i == 1))
        self.theme_combo.currentTextChanged.connect(self._change_theme)
        self.lang_combo.currentIndexChanged.connect(self._change_display_mode)
        self.mic_combo.currentIndexChanged.connect(self._change_device)
        self.history.itemDoubleClicked.connect(
            lambda item: self._set_suggestion(item.data(Qt.ItemDataRole.UserRole))
        )
        for key, handler in [
            (Qt.Key.Key_Return, self._on_enter), (Qt.Key.Key_Enter, self._on_enter),
            (Qt.Key.Key_Escape, self._on_escape), (Qt.Key.Key_F5, self.toggle_listening),
            (Qt.Key.Key_F8, self.clear_screen), (Qt.Key.Key_F9, self.toggle_output),
            (QKeySequence.StandardKey.Find, self.search.setFocus),
        ]:  # fmt: skip
            QShortcut(QKeySequence(key), self, activated=handler)

    def _update_settings(self, **changes) -> None:
        for k, v in changes.items():
            setattr(self.settings, k, v)
        self._save_settings()

    def _save_settings(self) -> None:
        try:
            self.settings.save()
        except OSError as e:
            self._show_error(f"Could not save settings: {e}")

    # ------------------------------------------------------------------ actions
    def toggle_listening(self) -> None:
        """Start or stop the speech pipeline (F5)."""
        if not self.pipeline:
            return
        if self._listening:
            self.pipeline.stop()
            self._set_listening(False)
            self._on_status("Stopped")
        else:
            self.banner.hide()
            self.pipeline.start()
            self._set_listening(True)
            self._on_status("Starting...")

    def _set_listening(self, on: bool) -> None:
        self._listening = on
        self.btn_listen.setText("Stop listening" if on else "Start listening")
        self.btn_listen.setObjectName("Stop" if on else "Start")
        self.btn_listen.style().unpolish(self.btn_listen)
        self.btn_listen.style().polish(self.btn_listen)
        color = theme.GREEN if on else theme.MUTED
        self.state_pill.setText("LISTENING" if on else "IDLE")
        self.state_pill.setStyleSheet(self._badge_css(color))

    def toggle_output(self) -> None:
        """Show / hide the broadcast window (F9)."""
        self.output.setVisible(not self.output.isVisible())

    def approve(self) -> None:
        """Put the pending suggestion on screen (Enter)."""
        if self.pending is None:
            return
        s = self.pending
        self.output.show_suggestion(s)
        if not self.output.isVisible():
            self.output.show()
        self.on_air = s
        if self.pipeline:
            self.pipeline.set_context(s.reference)
        self.air_label.setText(f"{s.reference.format_reference()}  ·  {s.primary.version}")
        self.air_label.setObjectName("")
        self.air_label.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600;")
        preview = s.primary.text if len(s.primary.text) < 42 else s.primary.text[:40].rsplit(" ", 1)[0] + "…"
        item = QListWidgetItem(f"{s.reference.format_reference()}  ·  {s.primary.version}     {preview}")
        item.setData(Qt.ItemDataRole.UserRole, s)
        self.history.insertItem(0, item)
        if self.history.count() > 200:
            self.history.takeItem(200)
        self._set_suggestion(None, keep_text="Shown on screen. Waiting for the next reference...")

    def dismiss(self) -> None:
        """Drop the pending suggestion (Esc)."""
        if self.pending is not None:
            self._set_suggestion(None, keep_text="Dismissed. Waiting for the next reference...")

    def clear_screen(self) -> None:
        """Blank the broadcast output (F8)."""
        self.output.clear()
        self.on_air = None
        self.air_label.setText("Nothing")
        self.air_label.setStyleSheet(f"color: {theme.MUTED};")

    def _on_enter(self) -> None:
        if self.search.hasFocus():
            self.manual_search()
        else:
            self.approve()

    def _on_escape(self) -> None:
        if self.search.hasFocus() and self.search.text():
            self.search.clear()
        else:
            self.dismiss()

    def manual_search(self) -> None:
        """Typed reference -> suggestion card (never straight to air)."""
        query = self.search.text().strip()
        if not query or not self.pipeline:
            return
        ctx = self.on_air.reference if self.on_air else None
        result = self.parser.parse(query, context_reference=ctx, typed=True)
        if result is None or not result.is_valid:
            why = "is not a real chapter/verse" if result else "was not recognised as a reference"
            self._hint(f"'{query}' {why}. Try e.g. John 3:16 or यूहन्ना 3:16")
            return
        suggestion = self.pipeline.build_suggestion(result)
        if suggestion is None:
            self._hint(f"No text found for {result.reference.format_reference()}")
            return
        self.search_hint.hide()
        self.search.clear()
        self.search.clearFocus()
        self._set_suggestion(suggestion)

    def _hint(self, text: str) -> None:
        self.search_hint.setText(text)
        self.search_hint.setStyleSheet(f"color: {theme.AMBER};")
        self.search_hint.show()

    def open_settings(self) -> None:
        """Edit settings; reload the model if STT options changed."""
        try:
            devices = list_input_devices()
        except Exception:
            devices = []
        versions = self.bible.versions() if self.bible else ["KJV"]
        dialog = SettingsDialog(self.settings, devices, versions, self)
        if not dialog.exec():
            return
        old = asdict(self.settings)
        vars(self.settings).update(asdict(dialog.result_settings))  # shared with pipeline
        self._save_settings()
        self.audio.gain = self.settings.input_gain
        self.audio.requested_device = self.settings.input_device
        self.theme_combo.setCurrentText(self.settings.theme)
        self.chk_save_audio.setChecked(self.settings.save_debug_audio)
        self._fill_devices()
        if any(old[f] != asdict(self.settings)[f] for f in STT_FIELDS):
            self.transcriber.unload()
            self._restart_if_listening()
        elif old["input_device"] != self.settings.input_device:
            self._restart_if_listening()

    def _restart_if_listening(self) -> None:
        if self._listening and self.pipeline:
            self.pipeline.stop()
            self.pipeline.start()

    def _change_device(self) -> None:
        self._update_settings(input_device=self.mic_combo.currentData())
        self.audio.requested_device = self.settings.input_device
        self._restart_if_listening()

    def _change_theme(self, name: str) -> None:
        self.output.set_theme(name)
        self._update_settings(theme=name)

    def _change_display_mode(self) -> None:
        mode = self.lang_combo.currentData()
        self.output.set_display_mode(mode)
        self._update_settings(display_mode=mode)

    # ------------------------------------------------------------------ pipeline events
    def _preload_model(self) -> None:
        try:
            self.transcriber.load(self.bridge.status.emit)
        except Exception as e:
            log.exception("Model preload failed")
            self.bridge.error.emit(str(e))
        self.bridge.status.emit("")

    def _on_status(self, text: str) -> None:
        if text:
            self.status.setText(text)
            if text.startswith(("Listening on", "Audio restored")):
                self.banner.hide()
        t = self.transcriber
        if t.device != "not loaded":
            gpu = t.device == "cuda"
            note = "" if gpu else f"  ·  {t.fallback_reason}" if t.fallback_reason else ""
            self.engine_label.setText(f"Whisper {t.model_name} on {'GPU' if gpu else 'CPU'}{note}")
            self.engine_label.setStyleSheet(f"color: {theme.GREEN if gpu else theme.AMBER};")

    def _show_error(self, text: str) -> None:
        self.banner_text.setText(f"⚠  {text}")
        self.banner.show()
        self.status.setText(text)
        if text.startswith(("Whisper model", "No microphone", "Could not open")) and self._listening:
            self._set_listening(False)

    def _clear_transcript(self) -> None:
        self.transcript.clear()
        self._transcript_chars = 0

    def _on_transcript(self, text: str, language: str) -> None:
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self._transcript_chars > 260 or self.transcript.document().isEmpty():
            if not self.transcript.document().isEmpty():
                cursor.insertBlock()
            self._transcript_chars = 0
        cursor.insertText(("" if self._transcript_chars == 0 else " ") + text)
        self._transcript_chars += len(text)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()
        self.lang_label.setText({"hi": "हिंदी", "en": "English"}.get(language, language))

    def _on_suggestion(self, s: Suggestion) -> None:
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        chip = (
            f"&nbsp;<span style='background-color:{theme.ACCENT_DIM}; color:{theme.ACCENT}; "
            f"font-weight:600;'>&nbsp;{html.escape(s.reference.format_reference())}&nbsp;</span>&nbsp;"
        )
        cursor.insertHtml(chip)
        self.transcript.ensureCursorVisible()
        air = self.on_air
        if s.action == "upgrade" and air and air.is_guess and air.reference.key[:2] == s.reference.key[:2]:
            # the operator already put the guessed verse on air; the spoken verse replaces it
            self.pending = s
            self.approve()
            self.status.setText(f"Upgraded on air: {air.reference.format_reference()} -> "
                                f"{s.reference.format_reference()}")
        elif should_auto_show(s, self.settings):
            self.pending = s
            self.approve()
            self.status.setText(f"Auto-shown {s.reference.format_reference()}")
        else:
            self._set_suggestion(s)

    def _set_suggestion(self, s: Optional[Suggestion], keep_text: str = "") -> None:
        self.pending = s
        self.btn_show.setEnabled(s is not None)
        self.btn_dismiss.setEnabled(s is not None)
        if s is None:
            self.sug_ref.setText("Listening for a reference" if self._listening else "No suggestion")
            self.sug_ref.setStyleSheet(f"color: {theme.MUTED};")
            self.sug_text.setText(keep_text or HINT)
            self.sug_text.setStyleSheet(f"color: {theme.MUTED}; font-size: 10.5pt;")
            self.sug_text_hi.hide()
            self.sug_warn.hide()
            self.conf_badge.hide()
            self._card_state("")
            return
        guess = "  (verse? guessed)" if s.is_guess else ""
        self.sug_ref.setText(f"{s.reference.format_reference()}{guess}  ·  {s.primary.version}")
        self.sug_ref.setStyleSheet("")
        self.sug_text.setStyleSheet("")
        self.sug_text.setText(s.primary.text)
        self.sug_text_hi.setVisible(s.secondary is not None)
        if s.secondary:
            self.sug_text_hi.setText(s.secondary.text)
        warnings = [describe_warning(w) for w in s.warnings]
        self.sug_warn.setText("\n".join(f"⚠  {w}" for w in warnings))
        self.sug_warn.setVisible(bool(warnings))
        conf = s.result.confidence
        color = theme.GREEN if conf >= 0.9 and not warnings else theme.AMBER if conf >= 0.6 else theme.RED
        self.conf_badge.setText(
            f"{conf * 100:.0f}%" if color == theme.GREEN else f"Review  ·  {conf * 100:.0f}%"
        )
        self.conf_badge.setStyleSheet(self._badge_css(color))
        self.conf_badge.show()
        self._card_state("active" if color == theme.GREEN else "warn")

    @staticmethod
    def _badge_css(color: str) -> str:
        border = theme.tint(color, 110)
        return f"background: {theme.tint(color, 34)}; color: {color}; border: 1px solid {border};"

    def _card_state(self, state: str) -> None:
        self.sug_card.setProperty("state", state)
        self.sug_card.style().unpolish(self.sug_card)
        self.sug_card.style().polish(self.sug_card)

    def _on_debug(self, d: ChunkDebugInfo) -> None:
        line = (
            f"[{d.timestamp}] RMS {d.rms:.4f}  {d.device} {d.sample_rate}Hz  lang={d.detected_language} "
            f"stt={d.stt_seconds:.2f}s\n   whisper: {d.raw_whisper_text!r}\n   parser:  {d.parser_output}"
        )
        if d.audio_saved_path:
            line += f"\n   audio:   {d.audio_saved_path}"
        self.debug_log.appendPlainText(line)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop threads and close the output window."""
        if self.pipeline:
            self.pipeline.stop(wait=3)
        self._save_settings()
        self.output.close()
        event.accept()
