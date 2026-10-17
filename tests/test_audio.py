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


def test_live_source_falls_back_to_mme_when_wasapi_refuses(monkeypatch):
    import sounddevice as sd

    from vachan.audio.source import LiveAudioSource

    apis = [{"name": "MME", "default_input_device": 0}, {"name": "Windows WASAPI", "default_input_device": 1}]
    devs = [
        {"name": "Line (USB Audio CODEC", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100.0},
        {"name": "Line (USB Audio CODEC)", "hostapi": 1, "max_input_channels": 2, "default_samplerate": 48000.0},
    ]
    opened = []

    class FakeStream:
        def __init__(self, device, channels, **kw):
            if device == 1 or channels == 2:
                raise sd.PortAudioError("Invalid number of channels")
            opened.append((device, channels))

        def start(self):
            pass

    monkeypatch.setattr(sd, "_terminate", lambda: None)
    monkeypatch.setattr(sd, "_initialize", lambda: None)
    monkeypatch.setattr(sd, "query_hostapis", lambda: apis)
    monkeypatch.setattr(sd, "query_devices", lambda i=None: devs if i is None else devs[i])
    monkeypatch.setattr(sd, "InputStream", FakeStream)
    monkeypatch.setattr(sd, "WasapiSettings", lambda **kw: kw)
    src = LiveAudioSource("Line (USB Audio CODEC)")
    src.start()
    assert opened == [(0, 1)] and src.sample_rate == 44100
