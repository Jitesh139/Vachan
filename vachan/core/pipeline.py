"""Background pipeline worker that coordinates Audio -> STT -> Parser -> Bible lookup.

Decoupled from PySide6 GUI to prevent any UI freezing.
Uses standard Python threading and thread-safe queues.
"""

import logging
from datetime import datetime
from pathlib import Path
import queue
import threading
from typing import Callable, Optional
import wave
import numpy as np

from vachan.audio.source import LiveAudioSource
from vachan.bible.provider import LocalSqliteBibleProvider
from vachan.core.models import BibleReference, ChunkDebugInfo, ParseResult
from vachan.parser.parser import ReferenceParser
from vachan.stt.transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)


class PipelineWorker:
    """Orchestrates live audio capture, transcription, parsing, and scripture lookup."""

    def __init__(
        self,
        audio_source: LiveAudioSource,
        transcriber: WhisperTranscriber,
        parser: ReferenceParser,
        bible_provider: LocalSqliteBibleProvider,
        default_version: str = "KJV",
        logs_dir: Optional[Path] = None,
        debug_audio_dir: Optional[Path] = None,
        save_debug_audio: bool = False,
    ) -> None:
        self.audio_source = audio_source
        self.transcriber = transcriber
        self.parser = parser
        self.bible_provider = bible_provider
        self.default_version = default_version
        self.logs_dir = logs_dir or Path("logs")
        self.debug_audio_dir = debug_audio_dir or Path("debug_audio")
        self.save_debug_audio = save_debug_audio

        # Ensure directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.save_debug_audio:
            self.debug_audio_dir.mkdir(parents=True, exist_ok=True)

        # Threading & queues
        self._audio_queue: queue.Queue = queue.Queue(maxsize=15)
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Tracking context for follow-up verses
        self._context_reference: Optional[BibleReference] = None
        self._context_lock = threading.Lock()

        # Rolling transcript buffer (25-30s window)
        self.buffer_window_seconds: float = 25.0
        self._rolling_transcript_buffer: list[dict] = []
        self._last_matched_ref_tuple: Optional[tuple] = None

        # Callbacks dispatched to UI
        self.on_transcript: Optional[Callable[[str], None]] = None
        self.on_verse_detected: Optional[Callable[[ParseResult, str, Optional[str]], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_debug_chunk: Optional[Callable[[ChunkDebugInfo], None]] = None

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

    def _save_chunk_wav(self, audio: np.ndarray, sample_rate: int) -> str:
        """Save raw audio chunk to debug_audio/ folder as 16-bit PCM WAV."""
        try:
            self.debug_audio_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = self.debug_audio_dir / f"chunk_{timestamp_str}.wav"
            clamped = np.clip(audio, -1.0, 1.0)
            int16_data = (clamped * 32767).astype(np.int16)
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16_data.tobytes())
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save debug audio: {e}")
            return ""

    def _log_debug_info(self, info: ChunkDebugInfo) -> None:
        """Append chunk telemetry to logs/pipeline.log."""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_dir / "pipeline.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[{info.timestamp}] RMS={info.rms:.5f} | Len={info.chunk_length_sec:.2f}s | "
                    f"SR={info.sample_rate}Hz | Ch={info.channels} | Lang={info.detected_language}\n"
                    f"  Whisper Text:       \"{info.raw_whisper_text}\"\n"
                    f"  Passed to Parser:   \"{info.text_passed_to_parser}\"\n"
                    f"  Parser Result:      {info.parser_output}\n"
                )
                if info.audio_saved_path:
                    f.write(f"  Debug Audio Saved:  {info.audio_saved_path}\n")
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write to pipeline log: {e}")

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
                # Telemetry inputs
                sr = getattr(self.audio_source, "sample_rate", 16000)
                ch = getattr(self.audio_source, "channels", 1)
                chunk_len_sec = round(len(chunk) / sr, 2) if sr > 0 else 0.0
                rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) > 0 else 0.0
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # Save debug WAV if enabled
                saved_wav_path = None
                if self.save_debug_audio:
                    saved_wav_path = self._save_chunk_wav(chunk, sr)

                # 1. Transcribe audio chunk
                if hasattr(self.transcriber, "transcribe_with_info"):
                    transcript, detected_lang = self.transcriber.transcribe_with_info(chunk)
                else:
                    transcript = self.transcriber.transcribe(chunk)
                    detected_lang = getattr(self.transcriber, "last_detected_language", "unknown")

                # Text passed to parser
                text_to_parse = transcript.strip() if transcript else ""

                # 2. Update rolling text buffer & parse
                raw_chunk_text = transcript.strip() if transcript else ""
                now_epoch = time.time()

                if raw_chunk_text:
                    if self.on_transcript:
                        self.on_transcript(raw_chunk_text)

                    # Append to rolling transcript buffer
                    self._rolling_transcript_buffer.append({"text": raw_chunk_text, "time": now_epoch})

                # Prune buffer older than buffer_window_seconds (25.0s)
                cutoff_time = now_epoch - self.buffer_window_seconds
                self._rolling_transcript_buffer = [
                    item for item in self._rolling_transcript_buffer if item["time"] >= cutoff_time
                ]

                # Combine buffer text
                combined_buffer_text = " ".join(item["text"] for item in self._rolling_transcript_buffer).strip()

                parse_res = None
                parser_output_str = "no match: empty transcript"

                if combined_buffer_text:
                    with self._context_lock:
                        ctx = self._context_reference

                    parse_res = self.parser.parse(combined_buffer_text, context_reference=ctx)

                    if parse_res and parse_res.is_valid and parse_res.reference:
                        ref = parse_res.reference
                        ref_tuple = (ref.book_id, ref.chapter, ref.verse_start, ref.verse_end)
                        ref_formatted = ref.format_reference(default_version=self.default_version)

                        # Check if duplicate of immediately previous match
                        if ref_tuple == self._last_matched_ref_tuple:
                            parser_output_str = f"already suggested: {ref_formatted}"
                            # Do not re-emit duplicate suggestion
                            parse_res = None
                        else:
                            self._last_matched_ref_tuple = ref_tuple
                            parser_output_str = f"{ref_formatted} (conf={parse_res.confidence:.2f})"

                            # CONSUME the matched portion from rolling buffer so it isn't suggested twice
                            raw_matched = parse_res.raw_match
                            matched_idx = combined_buffer_text.lower().find(raw_matched.lower())
                            if matched_idx != -1:
                                remainder_text = combined_buffer_text[matched_idx + len(raw_matched):].strip()
                                if remainder_text:
                                    self._rolling_transcript_buffer = [{"text": remainder_text, "time": now_epoch}]
                                else:
                                    self._rolling_transcript_buffer = []

                    elif parse_res and parse_res.low_confidence_reasons:
                        parser_output_str = f"no match: {', '.join(parse_res.low_confidence_reasons)}"
                    else:
                        parser_output_str = "no match: no citation recognized"

                # Build ChunkDebugInfo and log
                debug_info = ChunkDebugInfo(
                    timestamp=now_str,
                    rms=rms,
                    chunk_length_sec=chunk_len_sec,
                    sample_rate=sr,
                    channels=ch,
                    raw_whisper_text=raw_chunk_text,
                    detected_language=detected_lang,
                    text_passed_to_parser=combined_buffer_text,
                    parser_output=parser_output_str,
                    audio_saved_path=saved_wav_path,
                )

                self._log_debug_info(debug_info)

                if self.on_debug_chunk:
                    self.on_debug_chunk(debug_info)

                # 3. If valid verse citation detected, lookup Bible text
                if parse_res and parse_res.is_valid and parse_res.reference:
                    ref = parse_res.reference
                    v_code = ref.version or self.default_version

                    text_primary = self.bible_provider.get_verse_text(
                        ref.book_id, ref.chapter, ref.verse_start, ref.verse_end, version=v_code
                    )

                    text_hindi = self.bible_provider.get_verse_text(
                        ref.book_id, ref.chapter, ref.verse_start, ref.verse_end, version="HIN_IRV"
                    )

                    if text_primary and self.on_verse_detected:
                        self.on_verse_detected(parse_res, text_primary, text_hindi)

            except Exception as e:
                logger.exception("Error in pipeline process loop")
                if self.on_error:
                    self.on_error(str(e))
