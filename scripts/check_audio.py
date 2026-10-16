"""Check the audio Vachan sent to Whisper: level, clipping, noise.

    python scripts/check_audio.py                 # newest 10 files in debug_audio/
    python scripts/check_audio.py a.wav b.wav

Tick "Save audio" in the Debug panel first; every window sent to Whisper is then kept in
debug_audio/ (after Input gain). Thresholds are rules of thumb for Whisper, not exact limits.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vachan.core.paths import DEBUG_AUDIO_DIR  # noqa: E402

FRAME = 320  # 20 ms at 16 kHz


def _db(x: float) -> float:
    return float(20 * np.log10(max(float(x), 1e-6)))


def analyze(audio: np.ndarray) -> dict:
    """Levels in dBFS; speech = loud frames (90th percentile), noise = quiet frames (10th)."""
    n = len(audio) // FRAME * FRAME
    frames = np.sqrt(np.mean(audio[:n].reshape(-1, FRAME) ** 2, axis=1)) if n else np.zeros(1)
    speech, noise = _db(np.percentile(frames, 90)), _db(np.percentile(frames, 10))
    clipped = float(np.mean(np.abs(audio) >= 0.999) * 100) if len(audio) else 0.0
    problems = []
    if clipped > 0.1:
        problems.append("clipping: lower Input gain")
    if speech < -35:
        problems.append("too quiet: raise Input gain or move the mic closer")
    if speech - noise < 15:
        problems.append("noisy: little difference between speech and background")
    return {
        "speech_dbfs": round(speech, 1),
        "noise_dbfs": round(noise, 1),
        "snr_db": round(speech - noise, 1),
        "peak_dbfs": round(_db(float(np.max(np.abs(audio))) if len(audio) else 0.0), 1),
        "clipped_pct": round(clipped, 2),
        "verdict": "; ".join(problems) or "ok",
    }


def read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def main() -> int:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    paths = [Path(p) for p in sys.argv[1:]] or sorted(DEBUG_AUDIO_DIR.glob("*.wav"))[-10:]
    if not paths:
        print(f"No WAV files in {DEBUG_AUDIO_DIR}. Tick 'Save audio' in the Debug panel and speak again.")
        return 1
    print("file | speech dBFS | noise dBFS | SNR dB | peak dBFS | clipped % | verdict")
    for p in paths:
        a = analyze(read(p))
        print(
            f"{p.name} | {a['speech_dbfs']} | {a['noise_dbfs']} | {a['snr_db']} | "
            f"{a['peak_dbfs']} | {a['clipped_pct']} | {a['verdict']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
