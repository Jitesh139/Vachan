"""scripts/check_audio.py: level, clipping and noise of the audio sent to Whisper."""

import numpy as np

from scripts.check_audio import analyze

SR = 16000
rng = np.random.default_rng(1)


def speech_like(level, seconds=3.0):
    t = np.arange(int(SR * seconds)) / SR
    env = (np.sin(2 * np.pi * 2 * t) > 0).astype(np.float32)  # 0.25 s syllables, 0.25 s gaps
    return (level * env * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_good_level():
    a = analyze(speech_like(0.3) + rng.normal(0, 0.002, 3 * SR).astype(np.float32))
    assert a["verdict"] == "ok" and a["clipped_pct"] == 0 and a["snr_db"] > 30


def test_too_quiet():
    a = analyze(speech_like(0.01) + rng.normal(0, 0.001, 3 * SR).astype(np.float32))
    assert "quiet" in a["verdict"]


def test_clipping():
    a = analyze(np.clip(speech_like(3.0), -1, 1))
    assert "clipping" in a["verdict"] and a["clipped_pct"] > 1


def test_noisy():
    a = analyze(speech_like(0.1) + rng.normal(0, 0.05, 3 * SR).astype(np.float32))
    assert "noisy" in a["verdict"]
