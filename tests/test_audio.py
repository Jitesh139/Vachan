"""Ring buffer stream clock (no audio device needed)."""

import numpy as np

from vachan.audio.source import _RingAudio


def test_read_since_uses_stream_clock_and_resamples():
    ring = _RingAudio(buffer_sec=2.0)
    ring._reset(48000)
    ring._push(np.full(48000, 0.5, np.float32))  # 1 s at 48 kHz
    audio, now = ring.read_since(0.5)
    assert now == 1.0 and len(audio) == 8000  # 0.5 s at 16 kHz
    ring._push(np.zeros(96000, np.float32))
    audio, now = ring.read_since(0.0)
    assert now == 3.0 and len(audio) == 32000  # older audio fell out of the 2 s ring
    assert len(ring.read_since(now)[0]) == 0
