"""Audio adapters implementing the AudioSource port.

The PortAudio callback only copies mono samples into a ring buffer (no resampling or
allocation-heavy work on the audio thread). The pipeline pulls the latest window and
it is resampled to 16 kHz there.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger(__name__)
TARGET_SR = 16000


class _RingAudio:
    """Thread-safe mono ring buffer with level meter and 16 kHz read-out."""

    def __init__(self, buffer_sec: float = 30.0) -> None:
        self.device_name = ""
        self.sample_rate = TARGET_SR
        self.gain = 1.0
        self.overflows = 0
        self._buffer_sec = buffer_sec
        self._lock = threading.Lock()
        self._reset(TARGET_SR)

    def _reset(self, sample_rate: int) -> None:
        with self._lock:
            self.sample_rate = sample_rate
            self._ring = np.zeros(int(sample_rate * self._buffer_sec), np.float32)
            self._pos = 0
            self._filled = 0
            self._total = 0  # samples received since start (stream clock)
            self._level = 0.0
            self._last_block = time.monotonic()

    def _push(self, mono: np.ndarray) -> None:
        n = len(mono)
        with self._lock:
            size = len(self._ring)
            if n >= size:
                self._ring[:] = mono[-size:]
                self._pos, self._filled = 0, size
            else:
                end = self._pos + n
                if end <= size:
                    self._ring[self._pos : end] = mono
                else:
                    split = size - self._pos
                    self._ring[self._pos :] = mono[:split]
                    self._ring[: n - split] = mono[split:]
                self._pos = end % size
                self._filled = min(size, self._filled + n)
            self._total += n
            rms = float(np.sqrt(np.mean(mono * mono))) if n else 0.0
            # dBFS-style meter: -60 dB -> 0, 0 dB -> 1
            level = max(0.0, min(1.0, 1 + 20 * math.log10(max(rms * self.gain, 1e-6)) / 60))
            self._level = max(level, self._level * 0.85)
            self._last_block = time.monotonic()

    def get_level(self) -> float:
        """Meter level 0..1 (log scale, with fall-off)."""
        with self._lock:
            return self._level if time.monotonic() - self._last_block < 0.5 else 0.0

    def seconds_since_audio(self) -> float:
        """Seconds since the last block arrived."""
        with self._lock:
            return time.monotonic() - self._last_block

    def read_latest(self, seconds: float) -> np.ndarray:
        """Latest `seconds` of audio, gain applied, as 16 kHz mono float32."""
        with self._lock:
            raw = self._slice(int(seconds * self.sample_rate))
        return self._to_16k(raw)

    def read_since(self, t: float) -> tuple[np.ndarray, float]:
        """Audio from stream time `t` (seconds of audio since start) to now, and now."""
        with self._lock:
            now = self._total / self.sample_rate
            raw = self._slice(self._total - int(t * self.sample_rate))
        return self._to_16k(raw), now

    def _slice(self, n: int) -> np.ndarray:
        """Newest n native-rate samples; caller holds the lock."""
        n = min(self._filled, max(0, n))
        start = (self._pos - n) % len(self._ring)
        if start + n <= len(self._ring):
            return self._ring[start : start + n].copy()
        return np.concatenate((self._ring[start:], self._ring[: (start + n) % len(self._ring)]))

    def _to_16k(self, audio: np.ndarray) -> np.ndarray:
        sr = self.sample_rate
        if sr != TARGET_SR and len(audio):
            g = math.gcd(sr, TARGET_SR)
            audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)
        return np.clip(audio * self.gain, -1.0, 1.0).astype(np.float32)


def list_input_devices() -> list[dict]:
    """Input devices of the preferred host API (WASAPI, else MME), default first."""
    import sounddevice as sd

    apis = sd.query_hostapis()
    api_index = next((i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None)
    if api_index is None:
        api_index = next((i for i, a in enumerate(apis) if a["name"] == "MME"), 0)
    default = apis[api_index]["default_input_device"]
    devices = []
    for idx, d in enumerate(sd.query_devices()):
        if d["hostapi"] == api_index and d["max_input_channels"] > 0:
            devices.append({"index": idx, "name": d["name"], "is_default": idx == default})
    return sorted(devices, key=lambda d: not d["is_default"])


class LiveAudioSource(_RingAudio):
    """Microphone / mixer input via sounddevice, opened at the device's native rate."""

    def __init__(self, device_name: str = "", gain: float = 1.0) -> None:
        super().__init__()
        self.requested_device = device_name
        self.gain = gain
        self._stream = None
        self.stream_error = ""

    def _resolve(self) -> Optional[int]:
        devices = list_input_devices()
        for d in devices:
            if d["name"] == self.requested_device:
                return d["index"]
        if self.requested_device:
            log.warning("Input '%s' not found; using default", self.requested_device)
        return devices[0]["index"] if devices else None

    def start(self) -> None:
        """Open the input stream; raises RuntimeError with a readable message."""
        import sounddevice as sd

        self.stop()
        index = self._resolve()
        if index is None:
            raise RuntimeError("No microphone / audio input found. Plug one in and press Start.")
        info = sd.query_devices(index)
        sr = int(info["default_samplerate"])
        channels = min(2, int(info["max_input_channels"]))
        self._reset(sr)
        self.device_name, self.stream_error = info["name"], ""
        try:
            self._stream = sd.InputStream(
                device=index,
                samplerate=sr,
                channels=channels,
                dtype="float32",
                blocksize=int(sr * 0.05),
                callback=self._callback,
                finished_callback=self._finished,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise RuntimeError(f"Could not open '{info['name']}': {e}") from e
        log.info("Audio: %s @ %d Hz, %d ch", self.device_name, sr, channels)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status.input_overflow:
            self.overflows += 1
        self._push(indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0].copy())

    def _finished(self) -> None:
        self.stream_error = f"Audio input '{self.device_name}' stopped (unplugged?)"

    def stop(self) -> None:
        """Close the stream if open."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception as e:  # device already gone
                log.debug("Stream close: %s", e)


class FileAudioSource(_RingAudio):
    """Plays a 16-bit PCM WAV file in real time (looped) - for tests and soak runs."""

    def __init__(self, path: Path, loop: bool = True) -> None:
        super().__init__()
        with wave.open(str(path), "rb") as wf:
            sr, ch = wf.getframerate(), wf.getnchannels()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
        self._audio = (pcm.reshape(-1, ch).mean(axis=1) / 32768.0).astype(np.float32)
        self._file_sr, self._loop = sr, loop
        self.device_name = f"file:{Path(path).name}"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Begin real-time playback into the ring buffer."""
        self._reset(self._file_sr)
        self._stop.clear()
        self._thread = threading.Thread(target=self._play, daemon=True)
        self._thread.start()

    def _play(self) -> None:
        block = int(self._file_sr * 0.05)
        pos, t_next = 0, time.monotonic()
        while not self._stop.is_set():
            if pos >= len(self._audio):
                if not self._loop:
                    self._push(np.zeros(block, np.float32))  # trailing silence
                    t_next += 0.05
                    time.sleep(max(0.0, t_next - time.monotonic()))
                    continue
                pos = 0
            self._push(self._audio[pos : pos + block])
            pos += block
            t_next += 0.05
            time.sleep(max(0.0, t_next - time.monotonic()))

    def stop(self) -> None:
        """Stop playback."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
