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
    """Streams real-time microphone audio chunks with rolling RMS meter and normalization."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_duration_sec: float = 7.0,
        step_duration_sec: float = 2.0,
        input_gain: float = 1.0,
        enable_normalization: bool = True,
    ) -> None:
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.target_sample_rate = 16000
        self.channels = 1
        self.chunk_duration_sec = chunk_duration_sec
        self.step_duration_sec = step_duration_sec
        self.window_samples = int(self.sample_rate * self.chunk_duration_sec)
        self.step_samples = int(self.sample_rate * self.step_duration_sec)
        self.input_gain = input_gain
        self.enable_normalization = enable_normalization

        self.active_device_index: Optional[int] = device_index
        self.active_device_name: str = "Unknown"
        self.active_sample_rate: int = sample_rate

        self._stream: Optional[sd.InputStream] = None
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        self._is_running = False
        self._current_level = 0.0
        self._level_lock = threading.Lock()

        # Rolling buffer for accumulating audio until chunk_samples is reached
        self._buffer: List[np.ndarray] = []
        self._accumulated_samples = 0
        self._samples_since_step = 0
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

    def set_gain(self, gain: float) -> None:
        """Update input gain multiplier (e.g. 1.0 to 4.0)."""
        self.input_gain = max(0.1, min(10.0, float(gain)))

    def _convert_to_mono_float32(self, indata: np.ndarray) -> np.ndarray:
        """Ensure mono float32 normalized to [-1.0, 1.0], averaging stereo channels."""
        if indata.dtype == np.int16:
            data = indata.astype(np.float32) / 32768.0
        elif indata.dtype == np.int32:
            data = indata.astype(np.float32) / 2147483648.0
        elif indata.dtype != np.float32:
            data = indata.astype(np.float32)
        else:
            data = indata.copy()

        # Stereo / multichannel to mono: average all channels
        if data.ndim > 1 and data.shape[1] > 1:
            audio_mono = np.mean(data, axis=1)
        else:
            audio_mono = data.flatten()

        return audio_mono

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio array using scipy.signal.resample_poly."""
        if orig_sr == target_sr or len(audio) == 0:
            return audio
        try:
            from scipy.signal import resample_poly
            gcd = math.gcd(int(orig_sr), int(target_sr))
            up = target_sr // gcd
            down = orig_sr // gcd
            return resample_poly(audio, up, down).astype(np.float32)
        except Exception:
            # Fallback linear interpolation
            num_samples = int(len(audio) * target_sr / orig_sr)
            return np.interp(
                np.linspace(0.0, len(audio), num_samples, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply gain and normalize quiet speech for reliable Whisper recognition."""
        if len(audio) == 0:
            return audio

        # Apply input gain
        if self.input_gain != 1.0:
            audio = audio * self.input_gain

        # Peak normalization for quiet mic input
        if self.enable_normalization:
            peak = float(np.max(np.abs(audio)))
            if 0.005 < peak < 0.6:  # Audible speech but quiet
                norm_factor = min(4.0, 0.8 / peak)
                audio = audio * norm_factor

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Invoked by sounddevice on audio thread."""
        if not self._is_running:
            return

        # Flatten and average stereo channels to mono float32 [-1, 1]
        audio_mono = self._convert_to_mono_float32(indata)

        # Compute RMS volume level (0.0 to 1.0)
        rms = float(np.sqrt(np.mean(np.square(audio_mono)))) if len(audio_mono) > 0 else 0.0
        normalized_level = min(1.0, max(0.0, rms * 8.0 * self.input_gain))
        with self._level_lock:
            self._current_level = 0.7 * self._current_level + 0.3 * normalized_level

        # Accumulate samples for rolling STT window
        with self._buffer_lock:
            self._buffer.append(audio_mono)
            self._accumulated_samples += len(audio_mono)
            self._samples_since_step += len(audio_mono)

            # Trigger when step_samples reached and we have at least 2.0s of speech
            min_initial_samples = int(self.sample_rate * 2.0)
            if self._samples_since_step >= self.step_samples and self._accumulated_samples >= min_initial_samples:
                full_audio = np.concatenate(self._buffer)
                if len(full_audio) > self.window_samples:
                    window_audio = full_audio[-self.window_samples:]
                    # Retain window_samples in buffer
                    self._buffer = [window_audio]
                    self._accumulated_samples = self.window_samples
                else:
                    window_audio = full_audio

                self._samples_since_step = 0

                if self._callback:
                    # Resample if stream sample rate is not 16000
                    if self.sample_rate != self.target_sample_rate:
                        ready_chunk = self._resample(window_audio, self.sample_rate, self.target_sample_rate)
                    else:
                        ready_chunk = window_audio.copy()

                    # Apply mic gain and normalization
                    ready_chunk = self._normalize_audio(ready_chunk)

                    # Dispatch to background transcription worker
                    self._callback(ready_chunk)

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """Start streaming from microphone."""
        if self._is_running:
            return

        self._callback = callback
        self._is_running = True
        self.window_samples = int(self.sample_rate * self.chunk_duration_sec)
        self.step_samples = int(self.sample_rate * self.step_duration_sec)
        with self._buffer_lock:
            self._buffer = []
            self._accumulated_samples = 0
            self._samples_since_step = 0

        # Query device info for channels and default samplerate
        dev_channels = 1
        req_sr = self.sample_rate
        try:
            dev_info = sd.query_devices(self.device_index)
            self.active_device_name = dev_info.get("name", "Default Device")
            max_in = dev_info.get("max_input_channels", 1)
            # Use 1 or 2 channels (average in callback)
            dev_channels = min(2, max_in)
        except Exception:
            self.active_device_name = "Default Microphone"

        self.channels = dev_channels
        self.active_device_index = self.device_index

        try:
            self._stream = sd.InputStream(
                device=self.device_index,
                channels=dev_channels,
                samplerate=req_sr,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=int(req_sr * 0.1),  # 100ms blocks
            )
            self._stream.start()
            self.active_sample_rate = int(self._stream.samplerate)
        except Exception as e:
            # Fallback to device default samplerate if 16000 fails on device
            try:
                def_sr = int(dev_info.get("default_samplerate", 44100))
                self.sample_rate = def_sr
                self.chunk_samples = int(self.sample_rate * self.chunk_duration_sec)
                self._stream = sd.InputStream(
                    device=self.device_index,
                    channels=dev_channels,
                    samplerate=def_sr,
                    dtype="float32",
                    callback=self._audio_callback,
                    blocksize=int(def_sr * 0.1),
                )
                self._stream.start()
                self.active_sample_rate = int(self._stream.samplerate)
            except Exception:
                self._is_running = False
                raise e

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
