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
    """faster-whisper transcriber with GPU auto-detection, CPU fallback, and Bible biasing."""

    DEFAULT_INITIAL_PROMPT = (
        "John 3:16, Romans 8:28, Psalm 23, Matthew chapter 5 verse 7, "
        "यूहन्ना 3:16, रोमियों 8:28, भजन संहिता 23"
    )

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = False,
        language: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt if initial_prompt is not None else self.DEFAULT_INITIAL_PROMPT
        self.condition_on_previous_text = condition_on_previous_text
        self.language = language
        self.active_device = "cpu"
        self.active_compute_type = "int8"

        self._model = None
        self._lock = threading.Lock()
        self._init_error: Optional[str] = None
        self.last_detected_language = "unknown"
        self._language_history: list[str] = []

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

        self.last_detected_language = "unknown"

    def transcribe_with_info(self, audio_data: np.ndarray, initial_prompt: Optional[str] = None) -> tuple[str, str]:
        """Transcribe a mono float32 audio numpy array and return (text, detected_language)."""
        if self._model is None:
            self.load_model()

        if self._model is None:
            return "", "unknown"

        # Ensure float32 between -1.0 and 1.0
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        prompt = initial_prompt or self.initial_prompt

        # Transcribe with faster-whisper
        segments, info = self._model.transcribe(
            audio_data,
            beam_size=self.beam_size,
            language=self.language,  # None for auto-detect
            task="transcribe",
            vad_filter=self.vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                threshold=0.5,
                min_speech_duration_ms=250,
                speech_pad_ms=200,
            ),
            initial_prompt=prompt,
            condition_on_previous_text=self.condition_on_previous_text,
        )

        texts = []
        for segment in segments:
            clean = segment.text.strip()
            if clean:
                texts.append(clean)

        raw_lang = getattr(info, "language", "unknown") or "unknown"
        lang_prob = getattr(info, "language_probability", 1.0) or 1.0

        # Stabilize language auto-detection between short chunks
        if raw_lang in ("en", "hi", "ur"):
            self._language_history.append(raw_lang)
            if len(self._language_history) > 6:
                self._language_history.pop(0)
            detected_lang = raw_lang
        elif self._language_history and (lang_prob < 0.70 or raw_lang not in ("en", "hi", "ur")):
            # Fall back to majority language from recent stable window
            detected_lang = max(set(self._language_history), key=self._language_history.count)
        else:
            detected_lang = raw_lang

        self.last_detected_language = detected_lang
        return " ".join(texts), detected_lang

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe a mono float32 audio numpy array. Language is auto-detected."""
        text, _ = self.transcribe_with_info(audio_data)
        return text
