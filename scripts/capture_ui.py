"""Render both windows with realistic content and save PNG screenshots.

    python scripts/capture_ui.py --scale 1.25 [--theme "Royal Blue"] [--out docs/screenshots]
The scale emulates Windows display scaling (100% / 125% / 150%) via QT_SCALE_FACTOR.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    """Build the UI offscreen-free, populate it, grab PNGs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--theme", default="Midnight")
    ap.add_argument("--mode", default="both")
    ap.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    args = ap.parse_args()
    os.environ["QT_SCALE_FACTOR"] = str(args.scale)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from vachan.bible.provider import LocalSqliteBibleProvider
    from vachan.core.models import ChunkDebugInfo
    from vachan.core.settings import Settings
    from vachan.output.verse_window import VerseDisplayWindow
    from vachan.parser import ReferenceParser
    from vachan.stt.transcriber import WhisperTranscriber
    from vachan.ui import theme
    from vachan.ui.main_window import ControlWindow

    app = QApplication(sys.argv)
    theme.load_fonts()
    app.setStyleSheet(theme.STYLESHEET)
    settings = Settings(theme=args.theme, display_mode=args.mode)
    parser, bible = ReferenceParser(), LocalSqliteBibleProvider()
    output = VerseDisplayWindow(args.theme, args.mode)
    win = ControlWindow(settings, WhisperTranscriber(settings), parser, bible, output)
    win.settings.save = lambda *a: None  # never touch the user's settings.json

    def suggest(text: str, ctx=None):
        return win.pipeline.build_suggestion(parser.parse(text, context_reference=ctx))

    win._set_listening(True)
    win._on_status("Listening on Microphone (USB Audio Mixer) - Whisper large-v3-turbo on GPU")
    win.transcriber.device, win.transcriber.model_name = "cuda", "large-v3-turbo"
    win._on_status("")
    for line, lang in [
        ("Good morning church, it is such a blessing to be together today.", "en"),
        ("आज हम परमेश्वर के प्रेम के बारे में बात करेंगे।", "hi"),
        ("Turn with me to John chapter 3 verse 16.", "en"),
    ]:
        win._on_transcript(line, lang)
    first = suggest("Turn to John chapter 3 verse 16")
    win._on_suggestion(first)
    win.approve()
    win._on_transcript("अब रोमियों के आठ अध्याय में 28 पद देखेंगे।", "hi")
    second = suggest("रोमियों के आठ अध्याय में 28 पद देखेंगे")
    win._on_suggestion(second)
    win._on_transcript("And the next verse in the NIV says", "en")
    third_text = "Psalm 23 verse 1 in the NIV"
    third = suggest(third_text)
    win._set_suggestion(third)
    win.btn_debug.setChecked(True)
    mic = "Microphone (USB Audio Mixer)"
    match = "MATCH Psalms 23:1 (conf 1.00)"
    win._on_debug(ChunkDebugInfo("10:42:07.512", 0.0412, 7.0, mic, 48000, third_text, "en", 0.34, "", match))
    win.resize(1280, 800)
    win.show()
    output.resize(1280, 720)
    output.show()
    output.show_suggestion(first)

    def grab() -> None:
        args.out.mkdir(parents=True, exist_ok=True)
        tag = f"{round(args.scale * 100)}"
        win.grab().save(str(args.out / f"control_{tag}.png"))
        name = args.theme.split()[0].lower()
        output.grab().save(str(args.out / f"output_{name}_{args.mode}_{tag}.png"))
        print("saved", args.out)
        app.quit()

    QTimer.singleShot(1200, grab)
    app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
