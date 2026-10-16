"""Speed/accuracy profiles: each is an ordered list of engines, the first that loads wins.

    NVIDIA GPU (faster-whisper CUDA) -> Intel Arc iGPU (OpenVINO) -> CPU (faster-whisper int8)

Model sizes come from docs/BENCHMARKS.md: the laptop default must answer within ~3 s
of the end of a phrase. "custom" ignores profiles and uses the device/model settings.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from vachan.core.models import Transcript
from vachan.core.settings import Settings
from vachan.stt.openvino_transcriber import OpenVinoTranscriber, openvino_devices
from vachan.stt.transcriber import ModelUnavailable, WhisperTranscriber, add_cuda_dll_dirs, speech_spans

log = logging.getLogger(__name__)
OV_REPO = "OpenVINO/whisper-{}-int8-ov"

# (engine, model, device); engine "fw" = faster-whisper, "ov" = OpenVINO
PROFILES: dict[str, list[tuple[str, str, str]]] = {
    "accurate": [("fw", "large-v3-turbo", "cuda"), ("ov", "large-v3-turbo", "GPU"), ("fw", "medium", "cpu")],
    "balanced": [("fw", "large-v3-turbo", "cuda"), ("ov", "small", "GPU"), ("fw", "small", "cpu")],
    "fast": [("fw", "small", "cuda"), ("ov", "small", "GPU"), ("ov", "base", "CPU"), ("fw", "base", "cpu")],
}
PROFILE_NAMES = ["auto", *PROFILES, "custom"]


def has_nvidia() -> bool:
    try:
        import ctranslate2

        add_cuda_dll_dirs()
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve(profile: str) -> str:
    """'auto' -> accurate with an NVIDIA or Intel Arc GPU (large-v3-turbo: 1.2 s per 5 s on the
    Arc iGPU), balanced on a CPU-only machine (medium on CPU is slower than real time)."""
    if profile == "auto":
        return "accurate" if has_nvidia() or "GPU" in openvino_devices() else "balanced"
    return profile if profile in PROFILES or profile == "custom" else "balanced"


def build(settings: Settings, engine: str, model: str, device: str):
    if engine == "ov":
        return OpenVinoTranscriber(settings, OV_REPO.format(model), device)
    return WhisperTranscriber(settings, force=(model, device))


class ProfileTranscriber:
    """Transcriber port over the first engine of the chosen profile that loads on this machine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.profile = ""
        self.device, self.model_name, self.fallback_reason = "not loaded", "", ""
        self._t = None

    def load(self, progress: Optional[Callable[[str], None]] = None) -> None:
        if self._t is not None:
            return
        self.profile = resolve(self.settings.profile)
        if self.profile == "custom":
            self._use(WhisperTranscriber(self.settings), progress)
            return
        skipped = []
        for engine, model, device in PROFILES[self.profile]:
            if device == "cuda" and not has_nvidia():
                skipped.append("no NVIDIA GPU")
                continue
            if engine == "ov" and device not in openvino_devices():
                skipped.append(f"no OpenVINO {device}")
                continue
            try:
                self._use(build(self.settings, engine, model, device), progress)
            except ModelUnavailable:
                raise  # offline: the next engine would need a download too
            except Exception as e:
                log.exception("%s %s on %s failed; trying the next engine", engine, model, device)
                skipped.append(f"{device} failed: {e}")
                continue
            self.fallback_reason = self.fallback_reason or ", ".join(skipped)
            return
        raise RuntimeError(f"No speech engine could load ({'; '.join(skipped)})")

    def _use(self, t, progress) -> None:
        t.load(progress)
        self._t = t
        self._sync()

    def _sync(self) -> None:
        self.device, self.model_name = self._t.device, self._t.model_name
        self.fallback_reason = self._t.fallback_reason or self.fallback_reason

    def unload(self) -> None:
        if self._t is not None:
            self._t.unload()
        self._t = None
        self.device, self.model_name, self.fallback_reason = "not loaded", "", ""

    def speech_spans(self, audio: np.ndarray, min_silence: float = 0.5) -> list[tuple[float, float]]:
        return speech_spans(audio, min_silence)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        self.load()
        result = self._t.transcribe(audio, language)
        self._sync()  # faster-whisper may have switched from CUDA to CPU after an error
        return result
