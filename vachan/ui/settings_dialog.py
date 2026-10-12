"""Settings dialog."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vachan.core.settings import DEFAULT_PROMPT, Settings
from vachan.output.verse_window import THEMES

MODELS = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]
STT_FIELDS = ("device", "model_gpu", "model_cpu")  # changing these needs a model reload


def _combo(items: list[tuple[str, str]], current: str) -> QComboBox:
    box = QComboBox()
    for label, value in items:
        box.addItem(label, value)
    index = box.findData(current)
    box.setCurrentIndex(max(0, index))
    return box


def _section(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("Section")
    label.setContentsMargins(0, 14, 0, 2)
    return label


class SettingsDialog(QDialog):
    """Edits a copy of Settings; `result_settings` holds the accepted values."""

    def __init__(self, settings: Settings, devices: list[dict], versions: list[str], parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Vachan Settings")
        self.setMinimumWidth(560)
        self.result_settings = settings
        s = settings
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)

        form.addRow(_section("Audio input"))
        self.device = _combo(
            [("Windows default input", "")] + [(d["name"], d["name"]) for d in devices], s.input_device
        )
        form.addRow("Microphone / mixer", self.device)
        self.gain = QDoubleSpinBox(minimum=0.5, maximum=8.0, singleStep=0.25, value=s.input_gain, suffix=" x")
        form.addRow("Input gain", self.gain)
        self.silence = QDoubleSpinBox(
            decimals=4, minimum=0.0, maximum=0.05, singleStep=0.0005, value=s.silence_rms
        )
        self.silence.setToolTip("Windows quieter than this RMS are skipped without running Whisper")
        form.addRow("Silence threshold", self.silence)

        form.addRow(_section("Speech recognition"))
        self.stt_device = _combo(
            [("Automatic (GPU if available)", "auto"), ("NVIDIA GPU (CUDA)", "cuda"), ("CPU only", "cpu")],
            s.device,
        )
        form.addRow("Run Whisper on", self.stt_device)
        self.model_gpu = _combo([(m, m) for m in MODELS], s.model_gpu)
        form.addRow("Model on GPU", self.model_gpu)
        self.model_cpu = _combo([(m, m) for m in MODELS], s.model_cpu)
        form.addRow("Model on CPU", self.model_cpu)
        self.language = _combo(
            [("Hindi + English (auto per sentence)", "auto"), ("Hindi", "hi"), ("English", "en")], s.language
        )
        form.addRow("Language", self.language)
        self.window = QDoubleSpinBox(
            minimum=4.0, maximum=12.0, singleStep=0.5, value=s.window_sec, suffix=" s"
        )
        form.addRow("Audio window", self.window)
        self.step = QDoubleSpinBox(minimum=0.5, maximum=4.0, singleStep=0.25, value=s.step_sec, suffix=" s")
        form.addRow("Update every", self.step)
        self.prompt = QPlainTextEdit(s.initial_prompt)
        self.prompt.setFixedHeight(70)
        self.prompt.setToolTip("Vocabulary hint for Whisper. Never include chapter/verse numbers.")
        form.addRow("Vocabulary prompt", self.prompt)

        form.addRow(_section("Bible & display"))
        self.primary = _combo([(v, v) for v in versions], s.primary_version)
        form.addRow("Main translation", self.primary)
        self.secondary = _combo([("None", "")] + [(v, v) for v in versions], s.secondary_version)
        form.addRow("Second translation", self.secondary)
        self.theme = _combo([(t, t) for t in THEMES], s.theme)
        form.addRow("Output theme", self.theme)

        form.addRow(_section("Approval"))
        self.threshold = QSpinBox(
            minimum=50, maximum=100, value=round(s.auto_show_threshold * 100), suffix=" %"
        )
        form.addRow("Auto-show minimum confidence", self.threshold)
        self.cooldown = QSpinBox(minimum=5, maximum=600, value=round(s.duplicate_cooldown_sec), suffix=" s")
        form.addRow("Ignore repeats for", self.cooldown)
        self.save_audio = QCheckBox("Save every audio window sent to Whisper (debug_audio/)")
        self.save_audio.setChecked(s.save_debug_audio)
        form.addRow("", self.save_audio)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            lambda: self.prompt.setPlainText(DEFAULT_PROMPT)
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setObjectName("Primary")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 20)
        layout.addLayout(form)
        layout.addSpacing(12)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        self.result_settings = replace(
            self.result_settings,
            input_device=self.device.currentData(),
            input_gain=self.gain.value(),
            silence_rms=self.silence.value(),
            device=self.stt_device.currentData(),
            model_gpu=self.model_gpu.currentData(),
            model_cpu=self.model_cpu.currentData(),
            language=self.language.currentData(),
            window_sec=self.window.value(),
            step_sec=self.step.value(),
            initial_prompt=self.prompt.toPlainText().strip(),
            primary_version=self.primary.currentData(),
            secondary_version=self.secondary.currentData(),
            theme=self.theme.currentData(),
            auto_show_threshold=self.threshold.value() / 100,
            duplicate_cooldown_sec=float(self.cooldown.value()),
            save_debug_audio=self.save_audio.isChecked(),
        )
        self.accept()
