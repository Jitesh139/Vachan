"""Main PySide6 operator control window for Vachan.

Features:
- Live audio input selector and smooth RMS audio level meter.
- Start / Stop listening toggle button with status indicators.
- Live rolling transcript panel with Devanagari font support and auto-scroll.
- Reference suggestion card with Show (Enter) and Dismiss (Esc) controls.
- Manual Bible search input box (e.g. 'John 3:16') for fallback.
- Auto-show vs Manual approval mode toggle.
- Theme selector and Language mode (Both / English / Hindi) for the verse window.
- Quick button to show / hide the separate borderless output window.
"""

from typing import Optional
from PySide6.QtCore import QObject, QRect, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon, QKeyEvent, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vachan.audio.source import AudioDeviceManager, LiveAudioSource
from vachan.bible.provider import LocalSqliteBibleProvider
from vachan.core.models import BibleReference, ParseResult
from vachan.core.pipeline import PipelineWorker
from vachan.output.verse_window import THEMES, VerseDisplayWindow
from vachan.parser.parser import ReferenceParser
from vachan.stt.transcriber import WhisperTranscriber


class PipelineBridge(QObject):
    """Qt signal bridge dispatches background thread events safely to the Qt GUI thread."""

    transcript_received = Signal(str)
    verse_detected = Signal(object, str, str)  # ParseResult, text_en, text_hi
    error_occurred = Signal(str)


class ControlWindow(QMainWindow):
    """Operator control dashboard."""

    def __init__(
        self,
        audio_source: LiveAudioSource,
        transcriber: WhisperTranscriber,
        parser: ReferenceParser,
        bible_provider: LocalSqliteBibleProvider,
        verse_window: VerseDisplayWindow,
    ) -> None:
        super().__init__()
        self.audio_source = audio_source
        self.transcriber = transcriber
        self.parser = parser
        self.bible_provider = bible_provider
        self.verse_window = verse_window

        self.pipeline_bridge = PipelineBridge()
        self.pipeline_worker: Optional[PipelineWorker] = None

        self._pending_suggestion: Optional[ParseResult] = None
        self._pending_text_en: str = ""
        self._pending_text_hi: Optional[str] = None
        self.auto_show_mode = False

        self._init_window()
        self._init_ui()
        self._connect_signals()

        # Audio meter poll timer (runs at ~30 FPS on GUI thread)
        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self._update_meter)
        self.meter_timer.start(33)

    def _init_window(self) -> None:
        self.setWindowTitle("Vachan - Operator Dashboard")
        self.resize(1050, 720)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #12131A;
            }
            QWidget {
                color: #ECEFF4;
                font-family: 'Segoe UI', 'Nirmala UI', sans-serif;
                font-size: 14px;
            }
            QFrame#Panel {
                background-color: #1A1C27;
                border: 1px solid #282B3C;
                border-radius: 12px;
            }
            QFrame#CardHighlight {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E2336, stop:1 #22293F);
                border: 1px solid #3D4A6E;
                border-radius: 12px;
            }
            QPushButton {
                background-color: #2E3440;
                color: #ECEFF4;
                border: 1px solid #434C5E;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3B4252;
                border-color: #5E81AC;
            }
            QPushButton#BtnStart {
                background-color: #2E7D32;
                border-color: #388E3C;
                color: #FFFFFF;
                font-size: 15px;
                padding: 10px 24px;
            }
            QPushButton#BtnStart:hover {
                background-color: #388E3C;
            }
            QPushButton#BtnStop {
                background-color: #C62828;
                border-color: #D32F2F;
                color: #FFFFFF;
                font-size: 15px;
                padding: 10px 24px;
            }
            QPushButton#BtnShow {
                background-color: #1976D2;
                border-color: #2196F3;
                color: #FFFFFF;
                font-size: 14px;
                padding: 8px 20px;
            }
            QPushButton#BtnDismiss {
                background-color: #37474F;
                border-color: #455A64;
                color: #B0BEC5;
                font-size: 14px;
                padding: 8px 16px;
            }
            QComboBox, QLineEdit {
                background-color: #1E2230;
                border: 1px solid #3B4252;
                border-radius: 6px;
                padding: 6px 12px;
                color: #ECEFF4;
            }
            QComboBox:hover, QLineEdit:focus {
                border-color: #88C0D0;
            }
            QProgressBar {
                background-color: #1E2230;
                border: 1px solid #2E3440;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:0.8 #8BC34A, stop:1 #FFEB3B);
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #151722;
                border: 1px solid #282B3C;
                border-radius: 8px;
                padding: 12px;
                color: #E5E9F0;
                font-size: 16px;
                line-height: 1.5;
            }
        """)

    def _init_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)

        # ---------------- Top Control Bar ----------------
        top_bar = QFrame(self)
        top_bar.setObjectName("Panel")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 12, 16, 12)
        top_layout.setSpacing(14)

        # Start / Stop Toggle Button
        self.btn_toggle = QPushButton("▶ Start Listening", self)
        self.btn_toggle.setObjectName("BtnStart")
        self.btn_toggle.clicked.connect(self._toggle_listening)
        top_layout.addWidget(self.btn_toggle)

        # Microphone Dropdown
        mic_label = QLabel("Microphone:", self)
        mic_label.setStyleSheet("color: #88C0D0; font-weight: bold;")
        top_layout.addWidget(mic_label)

        self.mic_combo = QComboBox(self)
        self._populate_devices()
        self.mic_combo.currentIndexChanged.connect(self._change_device)
        top_layout.addWidget(self.mic_combo, stretch=2)

        # Audio Meter
        meter_label = QLabel("Level:", self)
        meter_label.setStyleSheet("color: #88C0D0; font-weight: bold;")
        top_layout.addWidget(meter_label)

        self.audio_meter = QProgressBar(self)
        self.audio_meter.setRange(0, 100)
        self.audio_meter.setValue(0)
        self.audio_meter.setFixedWidth(130)
        self.audio_meter.setFixedHeight(18)
        self.audio_meter.setTextVisible(False)
        top_layout.addWidget(self.audio_meter)

        # Output Window Toggle Button
        self.btn_verse_win = QPushButton("📺 Show Verse Screen", self)
        self.btn_verse_win.clicked.connect(self._toggle_verse_window)
        top_layout.addWidget(self.btn_verse_win)

        root_layout.addWidget(top_bar)

        # ---------------- Main Splitter (Left: Transcript, Right: Suggestion & Search) ----------------
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setHandleWidth(8)

        # Left Panel: Live Rolling Transcript
        transcript_panel = QFrame(splitter)
        transcript_panel.setObjectName("Panel")
        t_layout = QVBoxLayout(transcript_panel)
        t_layout.setContentsMargins(16, 16, 16, 16)
        t_layout.setSpacing(10)

        t_title = QLabel("🎙 Live Preacher Transcript (Hindi + English)", transcript_panel)
        t_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #88C0D0;")
        t_layout.addWidget(t_title)

        self.transcript_box = QTextEdit(transcript_panel)
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setFont(QFont("Nirmala UI", 16))
        self.transcript_box.setPlaceholderText("Preacher words will stream in real-time as you speak into the microphone...")
        t_layout.addWidget(self.transcript_box)

        # Status Label
        self.status_label = QLabel("Status: Idle. Click 'Start Listening' to begin speech recognition.", transcript_panel)
        self.status_label.setStyleSheet("color: #81A1C1; font-size: 12px;")
        t_layout.addWidget(self.status_label)

        # Right Panel: Detected Verse Suggestion + Approval + Controls
        right_panel = QFrame(splitter)
        right_panel.setObjectName("Panel")
        r_layout = QVBoxLayout(right_panel)
        r_layout.setContentsMargins(18, 16, 18, 16)
        r_layout.setSpacing(16)

        # Header with Mode Toggle
        mode_header = QHBoxLayout()
        r_title = QLabel("📖 Detected Verse Suggestion", right_panel)
        r_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #EBCB8B;")
        mode_header.addWidget(r_title)
        mode_header.addStretch()

        self.btn_manual_mode = QRadioButton("Manual (Enter)", right_panel)
        self.btn_manual_mode.setChecked(True)
        self.btn_manual_mode.toggled.connect(self._mode_changed)
        mode_header.addWidget(self.btn_manual_mode)

        self.btn_auto_mode = QRadioButton("Auto-Show", right_panel)
        mode_header.addWidget(self.btn_auto_mode)
        r_layout.addLayout(mode_header)

        # Verse Suggestion Card
        self.suggestion_card = QFrame(right_panel)
        self.suggestion_card.setObjectName("CardHighlight")
        sug_card_layout = QVBoxLayout(self.suggestion_card)
        sug_card_layout.setContentsMargins(16, 14, 16, 14)
        sug_card_layout.setSpacing(10)

        self.sug_ref_label = QLabel("No reference detected yet", self.suggestion_card)
        self.sug_ref_label.setStyleSheet("font-size: 17px; font-weight: bold; color: #E5A93C;")
        sug_card_layout.addWidget(self.sug_ref_label)

        self.sug_text_en = QLabel("Speak a reference like 'John 3:16' or 'यूहन्ना 3:16'...", self.suggestion_card)
        self.sug_text_en.setWordWrap(True)
        self.sug_text_en.setStyleSheet("color: #ECEFF4; font-size: 14px; line-height: 1.4;")
        sug_card_layout.addWidget(self.sug_text_en)

        self.sug_text_hi = QLabel("", self.suggestion_card)
        self.sug_text_hi.setWordWrap(True)
        self.sug_text_hi.setStyleSheet("color: #D8DEE9; font-size: 14px; font-family: 'Nirmala UI';")
        sug_card_layout.addWidget(self.sug_text_hi)

        # Action Buttons (Show / Dismiss)
        btn_action_layout = QHBoxLayout()
        self.btn_approve = QPushButton("✔ Show on Screen (Enter)", self.suggestion_card)
        self.btn_approve.setObjectName("BtnShow")
        self.btn_approve.clicked.connect(self._approve_verse)
        btn_action_layout.addWidget(self.btn_approve)

        self.btn_dismiss = QPushButton("✖ Dismiss (Esc)", self.suggestion_card)
        self.btn_dismiss.setObjectName("BtnDismiss")
        self.btn_dismiss.clicked.connect(self._dismiss_verse)
        btn_action_layout.addWidget(self.btn_dismiss)
        sug_card_layout.addLayout(btn_action_layout)

        r_layout.addWidget(self.suggestion_card)

        # Fallback Manual Search
        search_label = QLabel("🔍 Manual Scripture Search Fallback:", right_panel)
        search_label.setStyleSheet("color: #88C0D0; font-weight: bold;")
        r_layout.addWidget(search_label)

        search_box_layout = QHBoxLayout()
        self.search_input = QLineEdit(right_panel)
        self.search_input.setPlaceholderText("Type e.g. 'John 3:16', 'Romans 8:28', or 'यूहन्ना 3:16'...")
        self.search_input.returnPressed.connect(self._perform_manual_search)
        search_box_layout.addWidget(self.search_input)

        btn_search = QPushButton("Lookup", right_panel)
        btn_search.clicked.connect(self._perform_manual_search)
        search_box_layout.addWidget(btn_search)
        r_layout.addLayout(search_box_layout)

        # Settings / Presentation Style
        r_layout.addSpacing(10)
        style_title = QLabel("🎨 Output Screen Presentation Settings:", right_panel)
        style_title.setStyleSheet("color: #88C0D0; font-weight: bold;")
        r_layout.addWidget(style_title)

        style_row = QHBoxLayout()
        theme_lbl = QLabel("Theme:", right_panel)
        style_row.addWidget(theme_lbl)

        self.theme_combo = QComboBox(right_panel)
        for t in THEMES.keys():
            self.theme_combo.addItem(t)
        self.theme_combo.currentTextChanged.connect(self._change_theme)
        style_row.addWidget(self.theme_combo)

        lang_lbl = QLabel("Language:", right_panel)
        style_row.addWidget(lang_lbl)

        self.lang_combo = QComboBox(right_panel)
        self.lang_combo.addItem("Both (Side-by-Side)", "both")
        self.lang_combo.addItem("English Only", "english")
        self.lang_combo.addItem("Hindi Only", "hindi")
        self.lang_combo.currentIndexChanged.connect(self._change_language_mode)
        style_row.addWidget(self.lang_combo)

        r_layout.addLayout(style_row)
        r_layout.addStretch()

        splitter.addWidget(transcript_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([550, 450])
        root_layout.addWidget(splitter)

    def _populate_devices(self) -> None:
        """Enumerate audio devices into combo."""
        devices = AudioDeviceManager.get_input_devices()
        self.mic_combo.clear()
        target_index = 0
        for i, dev in enumerate(devices):
            self.mic_combo.addItem(f"[{dev['index']}] {dev['name']}", dev['index'])
            # Prefer device index 1 as selected by user
            if dev['index'] == 1:
                target_index = i

        if self.mic_combo.count() > 0:
            self.mic_combo.setCurrentIndex(target_index)

    def _change_device(self, idx: int) -> None:
        """Handle microphone selection change."""
        dev_index = self.mic_combo.currentData()
        if dev_index is not None:
            self.audio_source.set_device(dev_index)
            self.status_label.setText(f"Audio device changed to [{dev_index}]")

    def _connect_signals(self) -> None:
        self.pipeline_bridge.transcript_received.connect(self._on_transcript_safe)
        self.pipeline_bridge.verse_detected.connect(self._on_verse_detected_safe)
        self.pipeline_bridge.error_occurred.connect(self._on_error_safe)

    def _toggle_listening(self) -> None:
        """Start or stop speech pipeline."""
        if self.pipeline_worker and self.pipeline_worker._is_running:
            # Stop
            self.pipeline_worker.stop()
            self.pipeline_worker = None
            self.btn_toggle.setText("▶ Start Listening")
            self.btn_toggle.setObjectName("BtnStart")
            self.btn_toggle.setStyle(self.btn_toggle.style())
            self.status_label.setText("Status: Stopped listening.")
        else:
            # Start
            dev_idx = self.mic_combo.currentData()
            self.audio_source.device_index = dev_idx

            self.pipeline_worker = PipelineWorker(
                audio_source=self.audio_source,
                transcriber=self.transcriber,
                parser=self.parser,
                bible_provider=self.bible_provider,
                default_version="KJV",
            )
            # Route callbacks through Qt Signal Bridge
            self.pipeline_worker.on_transcript = lambda t: self.pipeline_bridge.transcript_received.emit(t)
            self.pipeline_worker.on_verse_detected = (
                lambda res, en, hi: self.pipeline_bridge.verse_detected.emit(res, en, hi or "")
            )
            self.pipeline_worker.on_error = lambda err: self.pipeline_bridge.error_occurred.emit(err)

            self.status_label.setText("Status: Initializing faster-whisper model (GPU/CPU)...")
            self.btn_toggle.setText("⏹ Stop Listening")
            self.btn_toggle.setObjectName("BtnStop")
            self.btn_toggle.setStyle(self.btn_toggle.style())

            self.pipeline_worker.start()
            self.status_label.setText(f"Status: Listening live on {self.transcriber.active_device.upper()}...")

    def _update_meter(self) -> None:
        """Refresh volume meter widget."""
        lvl = self.audio_source.get_level()
        pct = int(min(1.0, lvl) * 100)
        self.audio_meter.setValue(pct)

    def _toggle_verse_window(self) -> None:
        """Open or hide output display window."""
        if self.verse_window.isVisible():
            self.verse_window.hide()
            self.btn_verse_win.setText("📺 Show Verse Screen")
        else:
            self.verse_window.show()
            self.btn_verse_win.setText("📺 Hide Verse Screen")

    def _mode_changed(self) -> None:
        self.auto_show_mode = self.btn_auto_mode.isChecked()

    def _change_theme(self, theme_name: str) -> None:
        self.verse_window.apply_theme(theme_name)

    def _change_language_mode(self, idx: int) -> None:
        mode = self.lang_combo.currentData()
        self.verse_window.set_display_mode(mode)

    @Slot(str)
    def _on_transcript_safe(self, text: str) -> None:
        """Append live recognized speech to the transcript panel."""
        self.transcript_box.append(f"• {text}")
        # Auto-scroll to bottom
        sb = self.transcript_box.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.status_label.setText(f"Status: Active | Device: {self.transcriber.active_device.upper()} | Model: {self.transcriber.model_size}")

    @Slot(object, str, str)
    def _on_verse_detected_safe(self, parse_res: ParseResult, text_en: str, text_hi: str) -> None:
        """Update suggestion card when a reference is detected."""
        if not parse_res.reference:
            return

        ref = parse_res.reference
        ref_formatted = ref.format_reference(default_version="KJV")

        self._pending_suggestion = parse_res
        self._pending_text_en = text_en
        self._pending_text_hi = text_hi

        self.sug_ref_label.setText(f"✨ {ref_formatted} (Confidence: {parse_res.confidence * 100:.0f}%)")
        self.sug_text_en.setText(text_en)
        self.sug_text_hi.setText(text_hi if text_hi else "")

        # Highlight detected reference chip in transcript
        self.transcript_box.append(f"<span style='color: #E5A93C; font-weight: bold;'>[DETECTED REFERENCE: {ref_formatted}]</span>")
        sb = self.transcript_box.verticalScrollBar()
        sb.setValue(sb.maximum())

        # If in auto-show mode, push immediately to output
        if self.auto_show_mode and parse_res.confidence >= 0.80:
            self._approve_verse()

    @Slot(str)
    def _on_error_safe(self, error_message: str) -> None:
        self.status_label.setText(f"Notice: {error_message}")

    def _approve_verse(self) -> None:
        """Push the currently pending suggestion to the output screen."""
        if not self._pending_suggestion or not self._pending_suggestion.reference:
            return

        ref = self._pending_suggestion.reference
        ref_text = ref.format_reference(default_version="KJV")

        self.verse_window.set_verse(
            reference_text=ref_text,
            text_en=self._pending_text_en,
            text_hi=self._pending_text_hi,
        )
        if not self.verse_window.isVisible():
            self.verse_window.show()
            self.btn_verse_win.setText("📺 Hide Verse Screen")

        # Update context for relative follow-up citations
        if self.pipeline_worker:
            self.pipeline_worker.set_context_reference(ref)

        self.status_label.setText(f"Approved and sent to display: {ref_text}")

    def _dismiss_verse(self) -> None:
        """Dismiss current candidate suggestion."""
        self._pending_suggestion = None
        self._pending_text_en = ""
        self._pending_text_hi = None
        self.sug_ref_label.setText("Suggestion dismissed")
        self.sug_text_en.setText("Waiting for next spoken citation...")
        self.sug_text_hi.setText("")

    def _perform_manual_search(self) -> None:
        """Search manual input box."""
        query = self.search_input.text().strip()
        if not query:
            return

        res = self.parser.parse(query)
        if res and res.is_valid and res.reference:
            ref = res.reference
            text_en = self.bible_provider.get_verse_text(
                ref.book_id, ref.chapter, ref.verse_start, ref.verse_end, version="KJV"
            )
            text_hi = self.bible_provider.get_verse_text(
                ref.book_id, ref.chapter, ref.verse_start, ref.verse_end, version="HIN_IRV"
            )

            if text_en:
                self._on_verse_detected_safe(res, text_en, text_hi or "")
                self.search_input.clear()
        else:
            self.status_label.setText(f"Manual search: No valid citation found in '{query}'")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Keyboard shortcuts: Enter approves, Esc dismisses."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._approve_verse()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self._dismiss_verse()
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """Clean shutdown of background threads and windows."""
        if self.pipeline_worker:
            self.pipeline_worker.stop()
        self.verse_window.close()
        event.accept()
