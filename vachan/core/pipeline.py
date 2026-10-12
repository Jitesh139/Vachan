"""Speech -> reference -> verse pipeline on a worker thread. Pure domain code: talks to
audio, STT and Bible lookup only through the ports, and to the UI through callbacks.

Each step (1.5 s) the newest 7 s of audio is transcribed, so a spoken reference is
always complete inside some window. The parser sees the previous + current window
text (the rolling text buffer) and returns the newest reference; repeats of the same
reference from overlapping windows are suppressed by a cooldown.
"""

from __future__ import annotations

import difflib
import logging
import re
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Callable, Optional

import numpy as np

from vachan.core.models import BibleReference, ChunkDebugInfo, ParseResult, Suggestion
from vachan.core.paths import DEBUG_AUDIO_DIR, LOGS_DIR
from vachan.core.ports import AudioSource, BibleProvider, ReferenceResolver, Transcriber
from vachan.core.settings import Settings

log = logging.getLogger(__name__)
STALL_SECONDS = 3.0  # no audio blocks for this long = device lost
# Whole-transcript hallucinations Whisper produces on noise/music
HALLUCINATIONS = {"thank you", "thanks for watching", "धन्यवाद", "subscribe", "you"}


def _noop(*_args) -> None:
    pass


@dataclass
class PipelineEvents:
    """Callbacks to the UI; invoked on the worker thread."""

    status: Callable[[str], None] = _noop
    transcript: Callable[[str, str], None] = _noop  # new words, language
    suggestion: Callable[[Suggestion], None] = _noop
    error: Callable[[str], None] = _noop
    debug: Callable[[ChunkDebugInfo], None] = _noop


def new_words(previous: str, current: str) -> str:
    """Words of `current` not already shown from the overlapping `previous` window."""
    prev, cur = previous.split(), current.split()
    if not prev:
        return current

    def norm(words: list[str]) -> list[str]:
        return [re.sub(r"[^\wऀ-ॿ]", "", w.lower()) for w in words]

    matcher = difflib.SequenceMatcher(None, norm(prev), norm(cur), autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 2]
    if not blocks:
        return current
    last = max(blocks, key=lambda b: b.b + b.size)
    return " ".join(cur[last.b + last.size :])


def _pipeline_logger() -> logging.Logger:
    logger = logging.getLogger("vachan.pipeline_trace")
    if not logger.handlers:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            LOGS_DIR / "pipeline.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@dataclass
class _State:
    context: Optional[BibleReference] = None
    texts: deque = field(default_factory=lambda: deque(maxlen=2))
    recent: dict = field(default_factory=dict)  # ref key -> time suggested
    last_follow_up: float = 0.0
    pending_chapter: Optional[tuple] = None  # chapter-only match awaiting a verse
    shown_text: str = ""


class PipelineWorker:
    """Owns the worker thread. start()/stop() are non-blocking and safe to repeat."""

    def __init__(
        self,
        audio: AudioSource,
        transcriber: Transcriber,
        parser: ReferenceResolver,
        bible: BibleProvider,
        settings: Settings,
        events: Optional[PipelineEvents] = None,
    ) -> None:
        self.audio, self.transcriber, self.parser, self.bible = audio, transcriber, parser, bible
        self.settings = settings
        self.events = events or PipelineEvents()
        self._state = _State()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._trace = _pipeline_logger()

    @property
    def running(self) -> bool:
        """True while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Load the model (if needed), open audio and begin processing."""
        if self.running and not self._stop.is_set():
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._stop,), daemon=True)
        self._thread.start()

    def stop(self, wait: float = 0.0) -> None:
        """Signal the worker to finish; optionally wait for it."""
        self._stop.set()
        self.audio.stop()
        if wait and self._thread:
            self._thread.join(wait)

    def set_context(self, ref: Optional[BibleReference]) -> None:
        """Reference last shown on screen; 'next verse' etc. are relative to it."""
        with self._lock:
            self._state.context = ref

    # ------------------------------------------------------------------ worker
    def _run(self, stop: threading.Event) -> None:
        try:
            self.transcriber.load(self.events.status)
        except Exception as e:
            log.exception("Model load failed")
            self.events.error(str(e))
            return
        if not self._open_audio():
            return
        self.events.status(
            f"Listening on {self.audio.device_name} - Whisper {self.transcriber.model_name} "
            f"on {self.transcriber.device.upper()}"
        )
        errors = 0
        next_tick = time.monotonic()
        while not stop.is_set():
            next_tick += self.settings.step_sec
            stop.wait(max(0.0, next_tick - time.monotonic()))
            next_tick = max(next_tick, time.monotonic())  # never queue up stale work
            if stop.is_set():
                break
            if self.audio.seconds_since_audio() > STALL_SECONDS:
                self._recover_audio(stop)
                continue
            try:
                self._tick(stop)
                errors = 0
            except Exception as e:
                errors += 1
                log.exception("Pipeline step failed")
                self.events.error(f"Processing error: {e}")
                stop.wait(min(10.0, errors))  # back off, keep the service running
        if stop is self._stop:  # a quick Stop->Start may already own the audio stream
            self.audio.stop()

    def _open_audio(self) -> bool:
        try:
            self.audio.start()
            return True
        except Exception as e:
            log.exception("Audio open failed")
            self.events.error(str(e))
            return False

    def _recover_audio(self, stop: threading.Event) -> None:
        name = self.audio.device_name or "audio input"
        self.events.error(f"No audio from '{name}' - is it unplugged? Retrying...")
        while not stop.is_set():
            try:
                self.audio.start()
                stop.wait(1.0)
                if self.audio.seconds_since_audio() < STALL_SECONDS:
                    self.events.status(f"Audio restored: {self.audio.device_name}")
                    return
            except Exception as e:
                log.info("Audio retry failed: %s", e)
            stop.wait(3.0)

    def _tick(self, stop: Optional[threading.Event] = None) -> None:
        s = self.settings
        audio = self.audio.read_latest(s.window_sec)
        if len(audio) < 16000:
            return
        rms = float(np.sqrt(np.mean(audio * audio)))
        info = dict(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            rms=rms,
            chunk_length_sec=len(audio) / 16000,
            device=self.audio.device_name,
            sample_rate=self.audio.sample_rate,
        )
        if rms < s.silence_rms or not self.transcriber.has_speech(audio):
            self._emit_debug(info, "", "-", 0.0, "", "skipped: silence (no speech detected)")
            return
        saved = self._save_audio(audio) if s.save_debug_audio else None
        t = self.transcriber.transcribe(audio)
        text = t.text if t.text.lower().strip(" .!") not in HALLUCINATIONS else ""
        result_line, suggestion = self._handle_text(text)
        self._emit_debug(info, t.text, t.language, t.seconds, self._parse_text(), result_line, saved)
        if text:
            fresh = new_words(self._state.shown_text, text)
            self._state.shown_text = text
            if fresh:
                self.events.transcript(fresh, t.language)
        if suggestion and not (stop and stop.is_set()):
            self.events.suggestion(suggestion)

    def _parse_text(self) -> str:
        return " ".join(self._state.texts)

    def _handle_text(self, text: str) -> tuple[str, Optional[Suggestion]]:
        if not text:
            return "no match: empty transcript", None
        st = self._state
        st.texts.append(text)
        with self._lock:
            context = st.context
        result = self.parser.parse(self._parse_text(), context_reference=context)
        if result is None:
            return "no match: no book name followed by chapter/verse numbers", None
        if not result.is_valid:
            return f"no match: '{result.raw_match}' is not a real chapter/verse", None
        ref = result.reference
        now = time.monotonic()
        label = f"{ref.format_reference()} (conf {result.confidence:.2f})"
        if result.is_follow_up:
            # the same "next verse" phrase stays inside overlapping windows for a while
            if now - st.last_follow_up < self.settings.window_sec + self.settings.step_sec:
                return f"duplicate follow-up ignored: {label}", None
        elif now - st.recent.get(ref.key, -1e9) < self.settings.duplicate_cooldown_sec:
            return f"duplicate ignored: {label}", None
        if "chapter_only_default_verse_1" in result.low_confidence_reasons:
            if st.pending_chapter != ref.key:  # the verse number may still be coming
                st.pending_chapter = ref.key
                return f"waiting for a verse number: {label}", None
        st.pending_chapter = None
        suggestion = self.build_suggestion(result)
        if suggestion is None:
            return f"no text in Bible database for {label}", None
        st.recent = {k: v for k, v in st.recent.items() if now - v < 600}
        st.recent[ref.key] = now
        if result.is_follow_up:
            st.last_follow_up = now
        st.texts.clear()  # the matched words must not trigger again
        return f"MATCH {label} {list(result.low_confidence_reasons)}", suggestion

    def build_suggestion(self, result: ParseResult) -> Optional[Suggestion]:
        """Attach verse text in the primary (+ secondary) translation."""
        ref = result.reference
        s = self.settings
        primary = self.bible.get_passage(ref, ref.version or s.primary_version)
        if primary is None:
            return None
        secondary = None
        other = s.secondary_version if s.secondary_version != primary.version else s.primary_version
        if other and other != primary.version:
            secondary = self.bible.get_passage(ref, other)
        return Suggestion(result, primary, secondary)

    # ------------------------------------------------------------------ telemetry
    def _emit_debug(self, info, raw, lang, secs, parsed, result, saved=None) -> None:
        d = ChunkDebugInfo(
            raw_whisper_text=raw,
            detected_language=lang,
            stt_seconds=secs,
            text_passed_to_parser=parsed,
            parser_output=result,
            audio_saved_path=saved,
            **info,
        )
        self._trace.info(
            f"[{d.timestamp}] RMS={d.rms:.4f} len={d.chunk_length_sec:.1f}s "
            f"device='{d.device}' {d.sample_rate}Hz lang={d.detected_language} "
            f"stt={d.stt_seconds:.2f}s\n  whisper: {d.raw_whisper_text!r}\n"
            f"  parser input: {d.text_passed_to_parser!r}\n  result: {d.parser_output}"
            + (f"\n  audio: {saved}" if saved else "")
        )
        self.events.debug(d)

    def _save_audio(self, audio: np.ndarray) -> Optional[str]:
        try:
            DEBUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            path = DEBUG_AUDIO_DIR / f"window_{datetime.now():%Y%m%d_%H%M%S_%f}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
            return str(path)
        except OSError as e:
            log.warning("Could not save debug audio: %s", e)
            return None
