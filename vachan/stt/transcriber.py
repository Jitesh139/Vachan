"""faster-whisper adapter implementing the Transcriber port.

- GPU first (float16), verified with a warm-up inference: on Windows a missing
  cuBLAS/cuDNN DLL only fails at the first transcribe, not at model load.
- Falls back to CPU int8 with a smaller model, and again at runtime on CUDA errors
  (e.g. GPU out of memory).
- Downloads the model on first use with progress; clear error when offline.
- Language: Hindi or English per window (detection restricted to those two; Urdu
  probability counts as Hindi because they are acoustically the same).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from vachan.core.models import Transcript
from vachan.core.settings import Settings

log = logging.getLogger(__name__)
Progress = Optional[Callable[[str], None]]


class ModelUnavailable(RuntimeError):
    """The Whisper model is not on disk and could not be downloaded."""


def add_cuda_dll_dirs() -> None:
    """Make pip-installed CUDA libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12) loadable."""
    import importlib.util

    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue
        for root in spec.submodule_search_locations if spec else []:
            bin_dir = Path(root) / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


class WhisperTranscriber:
    """Transcriber backed by faster-whisper / CTranslate2."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "not loaded"
        self.model_name = ""
        self.fallback_reason = ""  # why we are not on the GPU, shown in the UI
        self._model = None
        self._lock = threading.Lock()
        self._last_language = "en"

    # ------------------------------------------------------------------ loading
    def load(self, progress: Progress = None) -> None:
        """Load the best available model; safe to call repeatedly from any thread."""
        with self._lock:
            if self._model is None:
                self._load(progress or (lambda msg: None))

    def unload(self) -> None:
        """Drop the model so the next load() applies changed settings."""
        with self._lock:
            self._model = None
            self.device = "not loaded"

    def _load(self, progress: Callable[[str], None]) -> None:
        import ctranslate2

        want_gpu = self.settings.device in ("auto", "cuda")
        if want_gpu:
            add_cuda_dll_dirs()
            try:
                gpus = ctranslate2.get_cuda_device_count()
            except Exception as e:  # driver too old / no CUDA runtime
                gpus, self.fallback_reason = 0, f"CUDA unavailable: {e}"
            if gpus:
                for compute in ("float16", "int8_float16"):
                    try:
                        self._try_load(self.settings.model_gpu, "cuda", compute, progress)
                        self.fallback_reason = ""
                        return
                    except ModelUnavailable:
                        raise
                    except Exception as e:
                        log.warning("GPU load failed (%s): %s", compute, e)
                        self.fallback_reason = f"GPU failed: {e}"
            elif not self.fallback_reason:
                self.fallback_reason = "No NVIDIA GPU detected"
        self._try_load(self.settings.model_cpu, "cpu", "int8", progress)

    def _try_load(self, name: str, device: str, compute: str, progress: Callable) -> None:
        from faster_whisper import WhisperModel

        path = self._ensure_downloaded(name, progress)
        progress(f"Loading Whisper '{name}' on {device.upper()} ({compute})...")
        t0 = time.perf_counter()
        # more threads is slower on hybrid P/E-core CPUs (measured: 11 threads 2.5x slower than 8)
        threads = min(8, max(1, (os.cpu_count() or 4) // 2))
        model = WhisperModel(path, device=device, compute_type=compute, cpu_threads=threads)
        # warm-up: forces cuBLAS/cuDNN to load now, so failures trigger the CPU fallback
        segments, _ = model.transcribe(np.zeros(16000, np.float32), language="en")
        list(segments)
        self._model, self.device, self.model_name = model, device, name
        log.info("Whisper %s on %s/%s ready in %.1fs", name, device, compute, time.perf_counter() - t0)
        progress(f"Whisper '{name}' ready on {device.upper()}")

    def _ensure_downloaded(self, name: str, progress: Callable[[str], None]) -> str:
        from faster_whisper.utils import _MODELS, download_model

        try:
            return download_model(name, local_files_only=True)
        except Exception:
            pass  # not cached yet
        repo = _MODELS.get(name, name)
        progress(f"Downloading Whisper '{name}' (first run only)...")
        result: dict = {}

        def worker() -> None:
            try:
                result["path"] = download_model(name)
            except Exception as e:  # network errors surface below
                result["error"] = e

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        total = self._repo_size(repo)
        cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--"))
        while thread.is_alive():
            thread.join(1.0)
            done = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) if cache.exists() else 0
            pct = f" {min(99, done * 100 // total)}%" if total else ""
            progress(f"Downloading Whisper '{name}'{pct} ({done / 1e6:.0f} MB)")
        if "error" in result:
            raise ModelUnavailable(
                f"Whisper model '{name}' is not downloaded and the download failed "
                f"({result['error']}). Connect to the internet once and press Start again."
            )
        return result["path"]

    @staticmethod
    def _repo_size(repo: str) -> int:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(repo, files_metadata=True)
            return sum(s.size or 0 for s in info.siblings if s.rfilename in ("model.bin",))
        except Exception:
            return 0

    # ------------------------------------------------------------------ inference
    def has_speech(self, audio: np.ndarray) -> bool:
        """Silero VAD (bundled with faster-whisper); a few ms per window on CPU."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        return bool(get_speech_timestamps(audio, VadOptions(min_speech_duration_ms=250)))

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Transcribe one window; on a CUDA runtime error switch to CPU and retry once."""
        self.load()
        try:
            return self._transcribe(audio)
        except RuntimeError as e:
            if self.device != "cuda":
                raise
            log.exception("CUDA error during transcription; falling back to CPU")
            with self._lock:
                self._model = None
                self.fallback_reason = f"GPU error ({e}); switched to CPU"
                self._try_load(self.settings.model_cpu, "cpu", "int8", lambda m: None)
            return self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        language = self._pick_language(audio)
        segments, _ = self._model.transcribe(
            audio.astype(np.float32, copy=False),
            language=language,
            beam_size=self.settings.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
            initial_prompt=self.settings.initial_prompt or None,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=[0.0, 0.2, 0.4],
        )
        text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6)
        return Transcript(text.strip(), language, time.perf_counter() - t0)

    def _pick_language(self, audio: np.ndarray) -> str:
        if self.settings.language in ("hi", "en"):
            return self.settings.language
        _, _, probs = self._model.detect_language(audio, vad_filter=True)
        p = dict(probs)
        hindi, english = p.get("hi", 0.0) + p.get("ur", 0.0), p.get("en", 0.0)
        if abs(hindi - english) >= 0.15:  # hysteresis: keep the last language when unsure
            self._last_language = "hi" if hindi > english else "en"
        return self._last_language
