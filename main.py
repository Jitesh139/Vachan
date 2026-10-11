"""Vachan - Windows desktop app for live church streaming.

Composition Root:
Instantiates the architectural adapters and wires them together:
1. LiveAudioSource (Audio Input)
2. WhisperTranscriber (STT)
3. ReferenceParser (NLP/Regex Cues)
4. LocalSqliteBibleProvider (Offline Scripture SQLite)
5. VerseDisplayWindow (Broadcast Output Window)
6. ControlWindow (Operator Dashboard)
"""

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vachan.audio.source import LiveAudioSource
from vachan.bible.provider import LocalSqliteBibleProvider
from vachan.output.verse_window import VerseDisplayWindow
from vachan.parser.parser import ReferenceParser
from vachan.stt.transcriber import WhisperTranscriber
from vachan.ui.main_window import ControlWindow


def main() -> int:
    """Application entrypoint and composition root."""
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Vachan")

    # 1. Instantiate Adapters
    # Using device index 1 (Microphone Array - Intel SST) as chosen by the user
    audio_source = LiveAudioSource(device_index=1, sample_rate=16000, chunk_duration_sec=2.5)

    # Using 'large-v3' as chosen by user, with auto CUDA/CPU detection & fallback
    transcriber = WhisperTranscriber(model_size="large-v3", device="auto", compute_type="default")

    parser = ReferenceParser()
    bible_provider = LocalSqliteBibleProvider()

    # 2. Instantiate Windows
    verse_window = VerseDisplayWindow(theme_name="Midnight")
    control_window = ControlWindow(
        audio_source=audio_source,
        transcriber=transcriber,
        parser=parser,
        bible_provider=bible_provider,
        verse_window=verse_window,
    )

    # 3. Show Windows
    control_window.show()
    # Also show the verse output window so operator can position or test it right away
    verse_window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
