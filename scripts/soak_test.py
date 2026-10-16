"""Soak test: the real UI + real Whisper on looped audio for N minutes.

    python scripts/soak_test.py --minutes 30
Builds a ~70 s sermon-like loop from the test_speech samples (run that first),
feeds it through the full app in Auto-show mode, and every 30 s records memory
(RSS), the worst UI-thread stall, suggestions and errors. Report: logs/soak_report.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_loop(out: Path) -> Path:
    """Concatenate the synthesized samples with pauses into one WAV."""
    from scripts.test_speech import SAMPLE_DIR, read_wav, write_wav

    clips = sorted(SAMPLE_DIR.glob("sample_*.wav"))
    if not clips:
        sys.exit("Run python scripts/test_speech.py first to create the speech samples.")
    pause = np.zeros(16000 * 4, np.float32)
    write_wav(out, np.concatenate([np.concatenate([read_wav(c), pause]) for c in clips]))
    return out


def main() -> int:
    """Run the soak and write the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    args = ap.parse_args()

    import psutil
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from vachan.audio.source import FileAudioSource
    from vachan.bible.provider import LocalSqliteBibleProvider
    from vachan.core.paths import APP_DIR, LOGS_DIR
    from vachan.core.settings import Settings
    from vachan.output.verse_window import VerseDisplayWindow
    from vachan.parser import ReferenceParser
    from vachan.stt.profiles import ProfileTranscriber
    from vachan.ui import theme
    from vachan.ui.main_window import ControlWindow

    loop_wav = build_loop(APP_DIR / "tmp" / "soak_loop.wav")
    app = QApplication(sys.argv)
    theme.load_fonts()
    app.setStyleSheet(theme.STYLESHEET)
    settings = Settings(auto_show=True, duplicate_cooldown_sec=20)
    settings.save = lambda *a: None  # leave the user's settings.json alone
    output = VerseDisplayWindow()
    win = ControlWindow(
        settings, ProfileTranscriber(settings), ReferenceParser(), LocalSqliteBibleProvider(), output
    )
    win.audio = win.pipeline.audio = FileAudioSource(loop_wav, loop=True)
    win.btn_debug.setChecked(True)
    win.show()
    output.show()

    proc = psutil.Process()
    stats = {"suggestions": 0, "errors": [], "ticks": 0, "worst_stall": 0.0, "rows": []}
    win.bridge.suggestion.connect(lambda s: stats.__setitem__("suggestions", stats["suggestions"] + 1))
    win.bridge.error.connect(lambda e: stats["errors"].append(e))
    win.bridge.debug.connect(lambda d: stats.__setitem__("ticks", stats["ticks"] + 1))
    start = time.monotonic()
    last_beat = [time.monotonic()]
    window_stall = [0.0]

    def heartbeat() -> None:  # a 50 ms timer: lateness = UI thread blocked
        now = time.monotonic()
        window_stall[0] = max(window_stall[0], now - last_beat[0] - 0.05)
        last_beat[0] = now

    def sample() -> None:
        minute = (time.monotonic() - start) / 60
        rss = proc.memory_info().rss / 1e6
        stats["worst_stall"] = max(stats["worst_stall"], window_stall[0])
        row = (
            f"{minute:6.1f} min  RSS {rss:7.1f} MB  stall {window_stall[0] * 1000:5.0f} ms  "
            f"ticks {stats['ticks']:5d}  suggestions {stats['suggestions']:4d}  errors {len(stats['errors'])}"
        )
        print(row, flush=True)
        stats["rows"].append((minute, rss, row))
        window_stall[0] = 0.0
        if minute >= args.minutes - 0.01:
            win.close()
            app.quit()

    beat = QTimer(interval=50, timeout=heartbeat)
    beat.start()
    sampler = QTimer(interval=30_000, timeout=sample)
    sampler.start()
    QTimer.singleShot(500, win.toggle_listening)
    app.exec()

    rows = stats["rows"]
    warm = next((r for r in rows if r[0] >= 5), rows[0])  # model + caches settle first
    growth = rows[-1][1] - warm[1]
    verdict = stats["suggestions"] > 0 and not stats["errors"] and stats["worst_stall"] < 0.5 and growth < 50
    report = "\n".join(
        [r[2] for r in rows]
        + [
            "",
            f"duration {rows[-1][0]:.1f} min, pipeline ticks {stats['ticks']}, "
            f"suggestions {stats['suggestions']}",
            f"memory after warm-up {warm[1]:.1f} MB -> end {rows[-1][1]:.1f} MB (growth {growth:+.1f} MB)",
            f"worst UI stall {stats['worst_stall'] * 1000:.0f} ms, errors {len(stats['errors'])}"
            + (f": {stats['errors'][:3]}" if stats["errors"] else ""),
            f"RESULT: {'PASS' if verdict else 'FAIL'}",
        ]
    )
    LOGS_DIR.mkdir(exist_ok=True)
    (LOGS_DIR / "soak_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
