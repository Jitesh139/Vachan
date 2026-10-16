"""Psalm 91:2 in English, Hindi and Hinglish, N times, through the real UI and real Whisper.

    python scripts/repeat_ui_test.py --runs 10 [--profile auto]
Plays the test_speech samples in real time through ControlWindow (offscreen), presses Show
on every suggestion and checks that the card, the output window and the expected reference
all agree. Run scripts/test_speech.py first so the samples exist.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PSALM_CASES = (8, 9, 10)  # indexes into test_speech.SAMPLES: en, hi, Hinglish "Psalms 91 verse 2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--profile", default="auto")
    args = ap.parse_args()

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from scripts.test_speech import SAMPLES, read_wav, synthesize, write_wav
    from vachan.audio.source import FileAudioSource
    from vachan.bible.provider import LocalSqliteBibleProvider
    from vachan.core.paths import APP_DIR
    from vachan.core.settings import Settings
    from vachan.output.verse_window import VerseDisplayWindow
    from vachan.parser import ReferenceParser
    from vachan.stt.profiles import ProfileTranscriber
    from vachan.ui.main_window import ControlWindow

    pause = np.zeros(16000 * 5, np.float32)
    clips = [read_wav(synthesize(*SAMPLES[i][:2], i)) for i in PSALM_CASES]
    audio = np.concatenate([np.concatenate([c, pause]) for _ in range(args.runs) for c in clips])
    wav = APP_DIR / "tmp" / "repeat_psalm91.wav"
    write_wav(wav, audio)
    expected = [SAMPLES[i][2] for _ in range(args.runs) for i in PSALM_CASES]

    app = QApplication(sys.argv)
    settings = Settings(profile=args.profile, duplicate_cooldown_sec=0)  # every repeat must show
    settings.save = lambda *a: None  # leave the user's settings.json alone
    stt = ProfileTranscriber(settings)
    stt.load(lambda m: print(f"[stt] {m}", flush=True))
    output = VerseDisplayWindow()
    win = ControlWindow(settings, stt, ReferenceParser(), LocalSqliteBibleProvider(), output)
    win.audio = win.pipeline.audio = FileAudioSource(wav, loop=False)
    seen: list[tuple[str, str]] = []
    start = time.monotonic()

    def on_suggestion(_s) -> None:
        card = win.sug_ref.text().split("  ·  ")[0]
        win.approve()
        on_air = win.output.current.reference.format_reference() if win.output.current else ""
        seen.append((card, on_air))
        print(f"{time.monotonic() - start:6.1f}s  card {card!r}  output {on_air!r}", flush=True)

    win.bridge.suggestion.connect(lambda s: QTimer.singleShot(0, lambda: on_suggestion(s)))
    QTimer.singleShot(200, win.toggle_listening)
    QTimer.singleShot(int(len(audio) / 16000 * 1000) + 8000, app.quit)
    app.exec()
    win.close()

    agree = sum(c == o for c, o in seen)
    correct = sum(c == e for (c, _), e in zip(seen, expected))
    ok = len(seen) == len(expected) and agree == correct == len(expected)
    print(f"\nsuggestions {len(seen)}/{len(expected)}, card == output {agree}, correct {correct}  "
          f"({stt.profile}: Whisper {stt.model_name} on {stt.device})")  # fmt: skip
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
