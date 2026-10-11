"""Audio capture and streaming adapter implementing the AudioSource protocol.

Uses sounddevice to capture low-latency microphone audio on a background worker thread.
Computes RMS volume levels for the UI audio level meter.
"""

import math
import threading
from typing import Callable, List, Optional
import numpy as np
import sounddevice as sd


class AudioDeviceManager:
    """Manages audio device enumeration and metadata."""

    @staticmethod
    def get_input_devices() -> List[dict]:
        """List all available input devices."""
        devices = sd.query_devices()
        input_devs = []
        default_in = sd.default.device[0]

        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devs.append({
                    "index": idx,
                    "name": dev.get("name", "Unknown"),
                    "channels": dev.get("max_input_channels", 1),
                    "default_samplerate": dev.get("default_samplerate", 16000),
                    "is_default": idx == default_in,
                })
        return input_devs


class LiveAudioSource:
    """Streams real-time microphone audio chunks with rolling RMS meter."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_duration_sec: float = 2.5,
    ) -> None:
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_samples = int(self.sample_rate * self.chunk_duration_sec)

        self._stream: Optional[sd.InputStream] = None
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        self._is_running = False
        self._current_level = 0.0
        self._level_lock = threading.Lock()

        # Buffer for accumulating audio until chunk_samples is reached
        self._buffer: List[np.ndarray] = []
        self._accumulated_samples = 0
        self._buffer_lock = threading.Lock()

    def set_device(self, device_index: Optional[int]) -> None:
        """Switch audio input device."""
        was_running = self._is_running
        cb = self._callback
        if was_running:
            self.stop()
        self.device_index = device_index
        if was_running and cb:
            self.start(cb)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Invoked by sounddevice on high-priority audio thread."""
        if not self._is_running:
            return

        # Flatten to mono 1D float32 array
        audio_mono = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()

        # Compute RMS volume level (0.0 to 1.0)
        rms = float(np.sqrt(np.mean(np.square(audio_mono)))) if len(audio_mono) > 0 else 0.0
        # Normalize and smooth
        normalized_level = min(1.0, max(0.0, rms * 8.0))
        with self._level_lock:
            self._current_level = 0.7 * self._current_level + 0.3 * normalized_level

        # Accumulate samples for STT
        with self._buffer_lock:
            self._buffer.append(audio_mono)
            self._accumulated_samples += len(audio_mono)

            if self._accumulated_samples >= self.chunk_samples:
                full_chunk = np.concatenate(self._buffer)
                # Keep rolling overlap of last 0.5s
                overlap_samples = int(self.sample_rate * 0.5)
                if len(full_chunk) > overlap_samples:
                    self._buffer = [full_chunk[-overlap_samples:]]
                    self._accumulated_samples = overlap_samples
                else:
                    self._buffer = []
                    self._accumulated_samples = 0

                if self._callback:
                    # Dispatch to transcription callback
                    self._callback(full_chunk)

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """Start streaming from microphone."""
        if self._is_running:
            return

        self._callback = callback
        self._is_running = True
        with self._buffer_lock:
            self._buffer = []
            self._accumulated_samples = 0

        self._stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop audio streaming."""
        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._level_lock:
            self._current_level = 0.0

    def get_level(self) -> float:
        """Return current audio level for UI meter."""
        with self._level_lock:
            return self._current_level
