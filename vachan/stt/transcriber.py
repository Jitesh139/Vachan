"""Speech-to-text transcriber implementing the Transcriber protocol using faster-whisper.

Features:
- Configurable model (default large-v3, small, medium, etc.)
- GPU float16 execution with automatic fallback to CPU/int8
- Automatic multi-lingual language detection (Hindi + English)
- Non-blocking worker queue processing
"""

import logging
import threading
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """faster-whisper transcriber with GPU auto-detection and CPU fallback."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.active_device = "cpu"
        self.active_compute_type = "int8"

        self._model = None
        self._lock = threading.Lock()
        self._init_error: Optional[str] = None

    def load_model(self) -> None:
        """Load or lazy-initialize the faster-whisper model."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            from faster_whisper import WhisperModel
            import ctranslate2

            # Determine best device
            can_cuda = False
            if self.requested_device in ("auto", "cuda"):
                try:
                    cuda_types = ctranslate2.get_supported_compute_types("cuda")
                    can_cuda = len(cuda_types) > 0
                except Exception:
                    can_cuda = False

            if can_cuda:
                target_dev = "cuda"
                target_comp = "float16" if self.requested_compute_type in ("default", "auto") else self.requested_compute_type
            else:
                target_dev = "cpu"
                target_comp = "int8" if self.requested_compute_type in ("default", "auto") else self.requested_compute_type

            try:
                logger.info(f"Loading faster-whisper model '{self.model_size}' on {target_dev} ({target_comp})...")
                self._model = WhisperModel(
                    self.model_size,
                    device=target_dev,
                    compute_type=target_comp,
                    cpu_threads=4,
                )
                self.active_device = target_dev
                self.active_compute_type = target_comp
                self._init_error = None
            except Exception as e:
                logger.warning(f"Failed to load on {target_dev}: {e}. Falling back to CPU int8...")
                # Fallback to CPU int8 if CUDA fails
                try:
                    self._model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=4,
                    )
                    self.active_device = "cpu"
                    self.active_compute_type = "int8"
                    self._init_error = None
                except Exception as cpu_err:
                    self._init_error = str(cpu_err)
                    raise

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe a mono float32 audio numpy array. Language is auto-detected."""
        if self._model is None:
            self.load_model()

        if self._model is None:
            return ""

        # Ensure float32 between -1.0 and 1.0
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Transcribe with faster-whisper without forcing language
        # beam_size=1 or 2 for low latency in live stream chunks
        segments, info = self._model.transcribe(
            audio_data,
            beam_size=2,
            language=None,  # Auto-detect language every chunk (code-switching Hindi+English)
            task="transcribe",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )

        texts = []
        for segment in segments:
            clean = segment.text.strip()
            if clean:
                texts.append(clean)

        return " ".join(texts)
