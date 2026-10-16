"""OpenVINO GenAI adapter implementing the Transcriber port (Intel CPU / integrated GPU).

Optional: needs `pip install -r requirements-openvino.txt`. On an Intel Core Ultra laptop
the Arc iGPU ran Whisper several times faster than faster-whisper on the CPU (see
docs/BENCHMARKS.md). The NPU is not used: it produced garbage ("!!!!") or failed to run
with a prompt in the benchmark.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from vachan.core.models import Transcript
from vachan.core.paths import APP_DIR
from vachan.core.settings import Settings
from vachan.stt.transcriber import MAX_NEW_TOKENS, ModelUnavailable, speech_spans

log = logging.getLogger(__name__)
CACHE_DIR = APP_DIR / "models" / "openvino_cache"  # compiled GPU kernels: later loads are fast


def openvino_devices() -> list[str]:
    """OpenVINO devices on this machine (e.g. ['CPU', 'GPU', 'NPU']); [] if not installed."""
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception:  # not installed / driver problem
        return []


class OpenVinoTranscriber:
    """Whisper via openvino_genai.WhisperPipeline on 'GPU' or 'CPU'."""

    def __init__(self, settings: Settings, repo: str, ov_device: str) -> None:
        self.settings, self.repo, self.ov_device = settings, repo, ov_device
        self.device = "not loaded"
        self.model_name = repo.split("/")[-1].removeprefix("whisper-").removesuffix("-ov")
        self.fallback_reason = ""
        self._pipe = None
        self._lock = threading.Lock()
        self._last_language = "en"

    def load(self, progress: Optional[Callable[[str], None]] = None) -> None:
        """Download (first run) and compile the model; idempotent."""
        progress = progress or (lambda msg: None)
        with self._lock:
            if self._pipe is not None:
                return
            import openvino_genai as ov_genai
            from huggingface_hub import snapshot_download

            try:
                path = snapshot_download(self.repo, local_files_only=True)
            except Exception:
                progress(f"Downloading Whisper '{self.model_name}' (first run only)...")
                try:
                    path = snapshot_download(self.repo)
                except Exception as e:
                    raise ModelUnavailable(
                        f"Whisper model '{self.repo}' is not downloaded and the download failed ({e}). "
                        "Connect to the internet once and press Start again."
                    ) from e
            progress(f"Loading Whisper '{self.model_name}' on Intel {self.ov_device} (OpenVINO)...")
            t0 = time.perf_counter()
            pipe = ov_genai.WhisperPipeline(path, self.ov_device, CACHE_DIR=str(CACHE_DIR))
            pipe.generate(np.zeros(16000, np.float32).tolist(), language="<|en|>", max_new_tokens=4)
            self._pipe, self.device = pipe, f"openvino-{self.ov_device.lower()}"
            secs = time.perf_counter() - t0
            log.info("Whisper %s on OpenVINO %s ready in %.1fs", self.repo, self.ov_device, secs)
            progress(f"Whisper '{self.model_name}' ready on Intel {self.ov_device}")

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            self.device = "not loaded"

    def speech_spans(self, audio: np.ndarray, min_silence: float = 0.5) -> list[tuple[float, float]]:
        return speech_spans(audio, min_silence)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        """Transcribe one segment; language None = detect, restricted to Hindi/English."""
        self.load()
        t0 = time.perf_counter()
        if self.settings.language in ("hi", "en"):
            language = self.settings.language
        text, detected = self._generate(audio, language)
        if language is None:
            if detected == "ur":  # Urdu is acoustically Hindi but written in Arabic script
                detected = "hi"
                text, _ = self._generate(audio, "hi")
            elif detected not in ("hi", "en"):
                detected = self._last_language
                text, _ = self._generate(audio, detected)
            language = self._last_language = detected
        return Transcript(text, language, time.perf_counter() - t0)

    def _generate(self, audio: np.ndarray, language: Optional[str]) -> tuple[str, str]:
        kwargs = {"language": f"<|{language}|>"} if language else {}
        result = self._pipe.generate(
            audio.astype(np.float32, copy=False).tolist(),
            task="transcribe",
            initial_prompt=self.settings.initial_prompt or None,
            max_new_tokens=MAX_NEW_TOKENS,
            **kwargs,
        )
        detected = (result.languages[0] if result.languages else "").strip("<|>") or (language or "en")
        return str(result).strip(), detected
