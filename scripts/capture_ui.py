"""Capture screenshots of the ControlWindow and VerseDisplayWindow."""

import os
import sys
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vachan.audio.source import LiveAudioSource
from vachan.bible.provider import LocalSqliteBibleProvider
from vachan.output.verse_window import VerseDisplayWindow
from vachan.parser.parser import ReferenceParser
from vachan.stt.transcriber import WhisperTranscriber
from vachan.ui.main_window import ControlWindow


def capture():
    app = QApplication(sys.argv)

    artifacts_dir = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    audio_source = LiveAudioSource(device_index=None)
    transcriber = WhisperTranscriber(model_size="tiny", device="cpu")
    parser = ReferenceParser()
    bible_provider = LocalSqliteBibleProvider()

    verse_win = VerseDisplayWindow(theme_name="Midnight")
    verse_win.set_verse(
        reference_text="John 3:16 · KJV",
        text_en="For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life.",
        text_hi="क्योंकि परमेश्वर ने जगत से ऐसा प्रेम रखा कि उसने अपना एकलौता पुत्र दे दिया, ताकि जो कोई उस पर विश्वास करे वह नाश न हो, परन्तु अनन्त जीवन पाए।",
    )
    verse_win.resize(1100, 680)
    verse_win.show()

    control_win = ControlWindow(
        audio_source=audio_source,
        transcriber=transcriber,
        parser=parser,
        bible_provider=bible_provider,
        verse_window=verse_win,
    )
    control_win.transcript_box.append("• Good morning church, welcome to Sunday service.")
    control_win.transcript_box.append("• Let us open our Bibles together to John 3:16.")
    control_win.transcript_box.append("<span style='color: #E5A93C; font-weight: bold;'>[DETECTED REFERENCE: John 3:16 (KJV)]</span>")
    control_win.show()

    def do_grab():
        # Grab screenshots
        pix_ctrl = control_win.grab()
        pix_ctrl.save(str(artifacts_dir / "control_window.png"))

        pix_verse = verse_win.grab()
        pix_verse.save(str(artifacts_dir / "verse_window.png"))

        print("Screenshots captured successfully.")
        app.quit()

    QTimer.singleShot(500, do_grab)
    app.exec()


if __name__ == "__main__":
    capture()
