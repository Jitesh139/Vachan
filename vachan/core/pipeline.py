"""Speech -> reference -> verse pipeline on a worker thread. Pure domain code: talks to
audio, STT and Bible lookup only through the ports, and to the UI through callbacks.

VAD endpointing: each tick looks at the audio since the open speech segment began.
When the speaker pauses (endpoint_sec of silence) the segment is transcribed ONCE and
appended to the rolling text buffer the parser reads; run-on speech is cut at
max_segment_sec at the longest short pause. While a segment is open it is re-transcribed
every partial_sec for the live transcript only. A reference is committed from a partial
only when two partials in a row agree on it and more words follow it (so it cannot still
be growing, e.g. "verse 2" -> "verse 21"); otherwise it is committed at the endpoint.
Language is detected once per stretch of speech and reused until lang_reset_sec of silence.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field, replace
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
SR = 16000
TICK_SEC = 0.25
LOOKBACK_SEC = 1.0  # idle: audio searched for the start of speech
PAD_SEC = 0.25  # audio kept around detected speech
PARTIAL_BUDGET_SEC = 1.0  # skip a partial expected to take longer (delays the endpoint)
FOLLOW_UP_GUARD_SEC = 5.0  # one "next verse" phrase seen by a partial and again by the final
# Whole-transcript hallucinations Whisper produces on noise/music
HALLUCINATIONS = {"thank you", "thanks for watching", "धन्यवाद", "subscribe", "you"}


def _noop(*_args) -> None:
    pass


@dataclass
class PipelineEvents:
    """Callbacks to the UI; invoked on the worker thread."""

    status: Callable[[str], None] = _noop
    transcript: Callable[[str, str], None] = _noop  # committed segment text, language
    partial: Callable[[str, str], None] = _noop  # unfinished segment text (replaces previous)
    latency: Callable[[float], None] = _noop  # end of phrase -> suggestion, seconds
    suggestion: Callable[[Suggestion], None] = _noop
    error: Callable[[str], None] = _noop
    debug: Callable[[ChunkDebugInfo], None] = _noop


def _ends_with(text: str, raw_match: str) -> bool:
    """True if the matched reference is the last thing said (its number may still grow)."""
    words = re.findall(r"[\w\u0900-\u097f]+", text.lower())
    return bool(words) and words[-1] == raw_match.split()[-1]


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
    context: Optional[BibleReference] = None  # reference on screen ("next verse" base)
    texts: deque = field(default_factory=lambda: deque(maxlen=3))  # (time, text) for the parser
    recent: dict = field(default_factory=dict)  # ref key (incl. verse) -> time suggested
    last_follow_up: float = 0.0
    pending: Optional[ParseResult] = None  # chapter-only match awaiting its verse
    pending_since: float = 0.0
    guessed: dict = field(default_factory=dict)  # (book, chapter, name) -> time a guess went out

    def outstanding(self, now: float, keep: float) -> Optional[BibleReference]:
        """Chapter still waiting for (or guessed without) a spoken verse."""
        if self.pending:
            return self.pending.reference
        live = [(t, k) for k, t in self.guessed.items() if now - t < keep]
        if not live:
            return None
        book, chapter, name = max(live)[1]
        return BibleReference(book, name, chapter, None)


@dataclass
class _Segment:
    """Endpointing state, in stream seconds (audio clock)."""

    start: Optional[float] = None  # open segment began here; None = waiting for speech
    last_end: float = 0.0  # previous segment ended here (never re-read)
    seen: float = 0.0  # stream time at the previous tick
    partial_len: float = 0.0  # segment length at the last partial
    partial_key: Optional[tuple] = None  # complete reference seen by the last partial
    committed: bool = False  # a stable partial already produced this segment's suggestion
    language: Optional[str] = None  # reused until a long silence; None = detect
    last_speech: float = -1e9  # speech last ended here


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
        self._clock = time.monotonic  # injectable for tests
        self._seg = _Segment()
        self._rtf = 0.5  # running STT seconds per audio second (partial budget)

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
        self._seg = _Segment()  # a new stream restarts the audio clock
        self.events.status(
            f"Listening on {self.audio.device_name} - Whisper {self.transcriber.model_name} "
            f"on {self.transcriber.device.upper()}"
        )
        errors = 0
        while not stop.is_set():
            stop.wait(TICK_SEC)  # STT time is not added on top: audio waits in the ring
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
                    self._seg = _Segment()
                    self.events.status(f"Audio restored: {self.audio.device_name}")
                    return
            except Exception as e:
                log.info("Audio retry failed: %s", e)
            stop.wait(3.0)

    def _tick(self, stop: Optional[threading.Event] = None) -> None:
        """One endpointing step: open, extend, partially transcribe or close a segment."""
        s, seg = self.settings, self._seg
        if seg.start is None:
            audio, now = self.audio.read_since(max(seg.last_end, seg.seen - LOOKBACK_SEC))
            seg.seen = now
            spans = self._spans(audio)
            if not spans:
                self._emit(self._check_pending(), stop)
                return
            seg.start = max(seg.last_end, now - len(audio) / SR + spans[0][0] - PAD_SEC)
            seg.partial_len, seg.partial_key, seg.committed = 0.0, None, False
            if seg.start - seg.last_speech > s.lang_reset_sec:
                seg.language = None  # long silence: the next speaker may use the other language
        audio, now = self.audio.read_since(seg.start)
        seg.seen = now
        begin, dur = now - len(audio) / SR, len(audio) / SR
        spans = self._spans(audio)
        if not spans:
            if dur > 1.0:
                seg.start = None  # noise that only looked like speech
            return
        speech_end = spans[-1][1]
        if dur - speech_end >= s.endpoint_sec:  # the speaker paused: commit the segment
            cut = min(dur, speech_end + PAD_SEC)
            seg.start, seg.last_end, seg.last_speech = None, begin + cut, begin + speech_end
            self._final(audio[: int(cut * SR)], begin + speech_end, stop)
        elif dur >= s.max_segment_sec:  # run-on speech: commit up to the best short pause
            cut = self._cut_point(audio, dur)
            seg.start = seg.last_end = seg.last_speech = begin + cut
            self._final(audio[: int(cut * SR)], begin + cut, stop)
            seg.partial_len, seg.partial_key, seg.committed = 0.0, None, False
        elif s.partial_sec and dur - seg.partial_len >= s.partial_sec:
            seg.partial_len = dur
            if self._rtf * dur > PARTIAL_BUDGET_SEC:
                return  # this machine is too slow for a partial of this length
            self._partial(audio, stop)

    def _spans(self, audio: np.ndarray, min_silence: Optional[float] = None) -> list:
        if len(audio) < SR // 4 or float(np.sqrt(np.mean(audio * audio))) < self.settings.silence_rms:
            return []
        return self.transcriber.speech_spans(audio, min_silence or self.settings.endpoint_sec)

    def _cut_point(self, audio: np.ndarray, dur: float) -> float:
        """Middle of the longest short pause in the second half, else the end."""
        spans = self._spans(audio, 0.1)
        gaps = [(b[0] - a[1], (a[1] + b[0]) / 2) for a, b in zip(spans, spans[1:])]
        gaps = [g for g in gaps if g[1] >= dur / 2]
        return max(gaps)[1] if gaps else dur

    def _stream_now(self) -> float:
        return self.audio.read_since(1e9)[1]  # a read from the future is empty: just the clock

    def _transcribe(self, audio: np.ndarray, kind: str):
        """STT with language reuse, hallucination filter, debug telemetry."""
        s, seg = self.settings, self._seg
        saved = self._save_audio(audio) if s.save_debug_audio and kind == "final" else None
        t = self.transcriber.transcribe(audio, seg.language)
        seg.language = t.language
        dur = len(audio) / SR
        self._rtf = 0.7 * self._rtf + 0.3 * (t.seconds / max(dur, 0.5))
        text = t.text if t.text.lower().strip(" .!") not in HALLUCINATIONS else ""
        info = dict(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            rms=float(np.sqrt(np.mean(audio * audio))) if len(audio) else 0.0,
            chunk_length_sec=dur,
            device=self.audio.device_name,
            sample_rate=self.audio.sample_rate,
        )
        return t, text, info, saved

    def _final(self, audio: np.ndarray, speech_end: float, stop) -> None:
        t, text, info, saved = self._transcribe(audio, "final")
        if text:
            self.events.transcript(text, t.language)
        parse_input = (self._parse_text() + " " + text).strip()
        line, suggestion = self._handle_text(text)
        if suggestion is None:
            suggestion = self._check_pending()
        self._emit_debug(info, t.text, t.language, t.seconds, parse_input, f"[final] {line}", saved)
        if suggestion and suggestion.action != "guess":
            self.events.latency(max(0.0, self._stream_now() - speech_end))
        self._emit(suggestion, stop)

    def _partial(self, audio: np.ndarray, stop) -> None:
        seg = self._seg
        t, text, info, _ = self._transcribe(audio, "partial")
        self.events.partial(text, t.language)
        line, suggestion = "partial: no complete reference", None
        result = self._parse((self._parse_text() + " " + text).strip(), self._clock()) if text else None
        complete = result and result.is_valid and result.reference.verse_start is not None
        key = result.reference.key if complete and not _ends_with(text, result.raw_match) else None
        if key and key == seg.partial_key and not seg.committed:
            line, suggestion = self._handle_text(text)  # stable: commit before the speaker pauses
            seg.committed = suggestion is not None
            line = f"stable partial: {line}"
        seg.partial_key = key
        self._emit_debug(info, t.text, t.language, t.seconds, text, f"[partial] {line}")
        if suggestion:
            self.events.latency(t.seconds)
        self._emit(suggestion, stop)

    def _emit(self, suggestion: Optional[Suggestion], stop: Optional[threading.Event]) -> None:
        if suggestion and not (stop and stop.is_set()):
            self.events.suggestion(suggestion)

    def _parse_text(self) -> str:
        return " ".join(text for _, text in self._state.texts)

    def _handle_text(self, text: str) -> tuple[str, Optional[Suggestion]]:
        """Parse the rolling text buffer after new committed text; decide and log."""
        if not text:
            return "no match: empty transcript", None
        st, s = self._state, self.settings
        now = self._clock()
        st.texts.append((now, text))
        while st.texts and now - st.texts[0][0] > s.context_sec:
            st.texts.popleft()  # old words must not join with new numbers
        result = self._parse(self._parse_text(), now)
        if result is None:
            return "no match: no book name followed by chapter/verse numbers", None
        if not result.is_valid:
            return f"no match: '{result.raw_match}' is not a real chapter/verse", None
        ref = result.reference
        chapter = (ref.book_id, ref.chapter, ref.book_name)
        if ref.verse_start is None:  # the verse number may still be coming: wait, never default
            if chapter in st.guessed:
                return self._decision("duplicate", result, "none"), None
            if st.pending is None or st.pending.reference.key != ref.key:
                st.pending, st.pending_since = result, now
            return self._decision("waiting", result, "none"), None
        if result.is_follow_up and now - st.last_follow_up < FOLLOW_UP_GUARD_SEC:
            # the same "next verse" phrase is seen by a stable partial and again by the final
            return self._decision("duplicate", result, "spoken"), None
        if not result.is_follow_up and now - st.recent.get(ref.key, -1e9) < s.duplicate_cooldown_sec:
            return self._decision("duplicate", result, "spoken"), None
        action = "upgrade" if chapter in st.guessed else "new"
        suggestion = self.build_suggestion(result, action)
        if suggestion is None:
            return f"no text in Bible database for {ref.format_reference()}", None
        st.pending = None  # a complete reference resolves (or supersedes) any waiting chapter
        st.guessed.pop(chapter, None)
        st.recent = {k: v for k, v in st.recent.items() if now - v < 600}
        st.recent[ref.key] = now
        if result.is_follow_up:
            st.last_follow_up = now
        st.texts.clear()  # the matched words must not trigger again
        return self._decision(action, result, "spoken"), suggestion

    def _parse(self, text: str, now: float) -> Optional[ParseResult]:
        """'verse 2' completes a chapter still waiting for its verse before it means
        'verse 2 of what is on screen'; 'next verse' is always relative to the screen."""
        st = self._state
        outstanding = st.outstanding(now, self.settings.duplicate_cooldown_sec)
        if outstanding is not None:
            result = self.parser.parse(text, context_reference=outstanding)
            if result and result.is_valid and result.reference.verse_start is not None:
                return result
        with self._lock:
            context = st.context
        return self.parser.parse(text, context_reference=context)

    def _check_pending(self) -> Optional[Suggestion]:
        """No verse followed the chapter in time: suggest verse 1, marked as a guess."""
        st = self._state
        now = self._clock()
        if st.pending is None or now - st.pending_since < self.settings.verse_wait_sec:
            return None
        result, st.pending = st.pending, None
        ref = result.reference
        suggestion = self.build_suggestion(result, "guess")
        st.guessed[(ref.book_id, ref.chapter, ref.book_name)] = now
        st.recent[ref.key] = now
        # buffer kept: a late "verse 2" still joins "Psalms 91" and upgrades the guess
        self._trace.info("  " + self._decision("guess", result, "default"))
        return suggestion

    def _decision(self, action: str, result: ParseResult, verse_source: str) -> str:
        """One log line per parse decision (also shown in the debug panel)."""
        ref = result.reference
        shown = ref.format_reference() + (":1 (GUESS)" if action == "guess" else "")
        match = "MATCH " if action in ("new", "upgrade", "guess") else ""
        return (
            f"{match}action={action} ref={shown} verse_source={verse_source} "
            f"conf={result.confidence:.2f} match={result.raw_match!r} "
            f"reasons={list(result.low_confidence_reasons)}"
        )

    def build_suggestion(self, result: ParseResult, action: str = "new") -> Optional[Suggestion]:
        """Attach verse text in the primary (+ secondary) translation.

        A chapter-only result becomes an explicit verse-1 guess, flagged "verse_guessed"
        so the card says so and auto-show never fires.
        """
        if result.reference.verse_start is None:
            result = replace(
                result,
                reference=replace(result.reference, verse_start=1),
                low_confidence_reasons=result.low_confidence_reasons + ("verse_guessed",),
            )
        ref = result.reference
        s = self.settings
        primary = self.bible.get_passage(ref, ref.version or s.primary_version)
        if primary is None:
            return None
        secondary = None
        other = s.secondary_version if s.secondary_version != primary.version else s.primary_version
        if other and other != primary.version:
            secondary = self.bible.get_passage(ref, other)
        return Suggestion(result, primary, secondary, action)

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
