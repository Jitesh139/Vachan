"""Background pipeline worker that coordinates Audio -> STT -> Parser -> Bible lookup.

Decoupled from PySide6 GUI to prevent any UI freezing.
Uses standard Python threading and thread-safe queues.
"""

import queue
import threading
from typing import Callable, Optional
import numpy as np

from vachan.audio.source import LiveAudioSource
from vachan.bible.provider import LocalSqliteBibleProvider
from vachan.core.models import BibleReference, ParseResult
from vachan.parser.parser import ReferenceParser
from vachan.stt.transcriber import WhisperTranscriber


class PipelineWorker:
    """Orchestrates live audio capture, transcription, parsing, and scripture lookup."""

    def __init__(
        self,
        audio_source: LiveAudioSource,
        transcriber: WhisperTranscriber,
        parser: ReferenceParser,
        bible_provider: LocalSqliteBibleProvider,
        default_version: str = "KJV",
    ) -> None:
        self.audio_source = audio_source
        self.transcriber = transcriber
        self.parser = parser
        self.bible_provider = bible_provider
        self.default_version = default_version

        # Threading & queues
        self._audio_queue: queue.Queue = queue.Queue(maxsize=15)
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Tracking context for follow-up verses
        self._context_reference: Optional[BibleReference] = None
        self._context_lock = threading.Lock()

        # Callbacks dispatched to UI
        self.on_transcript: Optional[Callable[[str], None]] = None
        self.on_verse_detected: Optional[Callable[[ParseResult, str, Optional[str]], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def start(self) -> None:
        """Start background processing pipeline."""
        if self._is_running:
            return

        self._is_running = True
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()

        # Start live microphone capture feeding the queue
        self.audio_source.start(self._on_audio_chunk)

    def stop(self) -> None:
        """Stop background processing pipeline."""
        self._is_running = False
        self.audio_source.stop()
        # Wake worker loop with sentinel
        self._audio_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._worker_thread = None

    def set_context_reference(self, ref: Optional[BibleReference]) -> None:
        """Update context reference for follow-ups."""
        with self._context_lock:
            self._context_reference = ref

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Incoming audio chunk from sounddevice."""
        if not self._is_running:
            return
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            # Drop older frames if pipeline falls behind
            try:
                _ = self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(chunk)
            except Exception:
                pass

    def _process_loop(self) -> None:
        """Continuous worker loop running STT, parsing, and Bible retrieval."""
        while self._is_running:
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None or not self._is_running:
                break

            try:
                # 1. Transcribe audio chunk
                transcript = self.transcriber.transcribe(chunk)
                if not transcript or not transcript.strip():
                    continue

                if self.on_transcript:
                    self.on_transcript(transcript)

                # 2. Parse scripture citation
                with self._context_lock:
                    ctx = self._context_reference

                parse_res = self.parser.parse(transcript, context_reference=ctx)

                if parse_res and parse_res.is_valid and parse_res.reference:
                    ref = parse_res.reference
                    v_code = ref.version or self.default_version

                    # 3. Lookup English & Hindi texts
                    text_primary = self.bible_provider.get_verse_text(
                        ref.book_id, ref.chapter, ref.verse_start, ref.verse_end, version=v_code
                    )

                    # Lookup Hindi text for stacked display
                    text_hindi = self.bible_provider.get_verse_text(
                        ref.book_id, ref.chapter, ref.verse_start, ref.verse_end, version="HIN_IRV"
                    )

                    if text_primary:
                        if self.on_verse_detected:
                            self.on_verse_detected(parse_res, text_primary, text_hindi)

            except Exception as e:
                if self.on_error:
                    self.on_error(str(e))
