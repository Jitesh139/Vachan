"""Pipeline tests with fake audio/STT adapters (no microphone or model needed)."""

import time

import numpy as np
import pytest

from vachan.bible.provider import DEFAULT_DB, LocalSqliteBibleProvider
from vachan.core.models import BibleReference, Transcript
from vachan.core.pipeline import TICK_SEC, PipelineEvents, PipelineWorker
from vachan.core.settings import Settings
from vachan.parser import ReferenceParser

pytestmark = pytest.mark.skipif(not DEFAULT_DB.exists(), reason="Bible DB not built")


class FakeAudio:
    """Speech/silence timeline on a manual (or real-time) stream clock: `on` s loud, `off` s quiet."""

    device_name, sample_rate = "fake", 16000

    def __init__(self, loud=True, fail=False, on=1.0, off=1.0, realtime=False):
        self.loud, self.fail, self.on, self.off, self.realtime = loud, fail, on, off, realtime
        self.t, self._t0 = 0.0, time.monotonic()

    def start(self):
        if self.fail:
            raise RuntimeError("Could not open 'fake': device unplugged")
        self._t0 = time.monotonic()

    def stop(self):
        pass

    def get_level(self):
        return 0.5

    def now(self):
        return time.monotonic() - self._t0 if self.realtime else self.t

    def read_since(self, t):
        now = self.now()
        t = max(0.0, min(t, now))
        ts = t + np.arange(int(round((now - t) * 16000))) / 16000
        speech = self.loud & ((ts % (self.on + self.off)) < self.on)
        return np.where(speech, 0.1, 0.0).astype(np.float32), now

    def seconds_since_audio(self):
        return 0.0


def fake_spans(audio, min_silence=0.5):
    """Silero semantics: loud runs, gaps < min_silence merged, last span open until quiet."""
    loud = np.abs(audio[: len(audio) // 160 * 160]).reshape(-1, 160).max(axis=1) > 0.05
    spans, start = [], None
    for i, on in enumerate(np.append(loud, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append([start / 100, i / 100])
            start = None
    merged = []
    for sp in spans:
        if merged and sp[0] - merged[-1][1] < min_silence:
            merged[-1][1] = sp[1]
        else:
            merged.append(sp)
    if merged and len(loud) / 100 - merged[-1][1] < min_silence:
        merged[-1][1] = len(loud) / 100  # still inside the pause window: span not closed
    return [tuple(sp) for sp in merged if sp[1] - sp[0] >= 0.25]  # min_speech_duration_ms=250


class FakeSTT:
    """Pops one text per transcription; `texts` may be a function of the audio length."""

    device, model_name = "cpu", "fake"

    def __init__(self, texts, fail_load=False, language="en"):
        self.texts = texts if callable(texts) else list(texts)
        self.fail_load, self.language = fail_load, language
        self.calls = []  # (seconds of audio, language hint)

    def load(self, progress=None):
        if self.fail_load:
            raise RuntimeError("Whisper model 'small' is not downloaded")

    def speech_spans(self, audio, min_silence=0.5):
        return fake_spans(audio, min_silence)

    def transcribe(self, audio, language=None):
        self.calls.append((len(audio) / 16000, language))
        if callable(self.texts):
            text = self.texts(len(audio) / 16000)
        else:
            text = self.texts.pop(0) if self.texts else ""
        return Transcript(text, language or self.language, 0.01)


@pytest.fixture(scope="module")
def bible():
    return LocalSqliteBibleProvider()


def make(texts, bible, **kw):
    got = {"suggestions": [], "errors": [], "debug": [], "transcript": [], "partial": [], "latency": []}
    events = PipelineEvents(
        suggestion=got["suggestions"].append,
        error=got["errors"].append,
        debug=got["debug"].append,
        transcript=lambda t, lang: got["transcript"].append(t),
        partial=lambda t, lang: got["partial"].append(t),
        latency=got["latency"].append,
    )
    audio = kw.pop("audio", None) or FakeAudio()
    stt = kw.pop("stt", None) or FakeSTT(texts)
    kw.setdefault("partial_sec", 0.0)  # one transcription per phrase unless a test wants partials
    worker = PipelineWorker(audio, stt, ReferenceParser(), bible, Settings(**kw), events)
    return worker, got


def advance(worker, seconds):
    """Let `seconds` of audio arrive, ticking like the worker thread does."""
    for _ in range(int(round(seconds / TICK_SEC))):
        worker.audio.t += TICK_SEC
        worker._tick()


def utter(worker, n=1):
    """n phrases: each one speech burst + pause (one FakeAudio on/off cycle)."""
    advance(worker, n * (worker.audio.on + worker.audio.off))


def test_transcript_to_suggestion_with_both_languages(bible):
    worker, got = make([" Turn to John chapter 3 verse 16."], bible)
    utter(worker)
    (s,) = got["suggestions"]
    assert s.reference.format_reference() == "John 3:16"
    assert s.primary.version == "KJV" and "For God so loved" in s.primary.text
    assert s.secondary.version == "HIN_IRV" and "परमेश्वर" in s.secondary.text
    assert "action=new ref=John 3:16 verse_source=spoken" in got["debug"][0].parser_output


def test_reference_split_across_phrases(bible):
    worker, got = make(["Turn with me to Romans", "chapter 8 verse 28 please"], bible)
    utter(worker, 2)
    assert [s.reference.format_reference() for s in got["suggestions"]] == ["Romans 8:28"]


def test_each_phrase_is_transcribed_once(bible):
    texts = ["Turn to John chapter 3 verse 16", "John chapter 3 verse 16 for God so loved"]
    worker, got = make(texts, bible)
    utter(worker, 2)
    assert len(worker.transcriber.calls) == 2
    assert got["transcript"] == texts
    assert len(got["suggestions"]) == 1  # the repeat is a duplicate
    assert "action=duplicate" in got["debug"][1].parser_output


def test_follow_up_uses_approved_context(bible):
    worker, got = make(["Now look at the next verse."], bible)
    worker.set_context(BibleReference("JHN", "John", 3, 16))
    utter(worker)
    assert got["suggestions"][0].reference.format_reference() == "John 3:17"


def test_chapter_then_verse_in_next_phrase(bible):
    worker, got = make(["Let us turn to Romans 8", "verse 28"], bible)
    utter(worker)
    assert got["suggestions"] == []
    utter(worker)
    assert [s.reference.format_reference() for s in got["suggestions"]] == ["Romans 8:28"]


def test_unavailable_version_falls_back_with_notice(bible):
    worker, got = make(["Turn to Ephesians 2 8 in the NIV"], bible)
    utter(worker)
    s = got["suggestions"][0]
    assert s.primary.version == "KJV" and "NIV is not available offline" in s.primary.notice


def test_silence_skips_stt(bible):
    worker, got = make(["should not be used"], bible, audio=FakeAudio(loud=False))
    utter(worker, 3)
    assert worker.transcriber.calls == [] and not got["suggestions"]


def test_load_and_audio_errors_are_reported_not_raised(bible):
    events_errors = []
    w = PipelineWorker(
        FakeAudio(),
        FakeSTT([], fail_load=True),
        ReferenceParser(),
        bible,
        Settings(),
        PipelineEvents(error=events_errors.append),
    )
    w.start()
    w.stop(wait=2)
    w2 = PipelineWorker(
        FakeAudio(fail=True),
        FakeSTT([]),
        ReferenceParser(),
        bible,
        Settings(),
        PipelineEvents(error=events_errors.append),
    )
    w2.start()
    w2.stop(wait=2)
    assert any("not downloaded" in e for e in events_errors)
    assert any("unplugged" in e for e in events_errors)


def test_threaded_start_stop(bible):
    worker, got = make(["Turn to John 3 16"] * 50, bible, audio=FakeAudio(on=0.5, off=0.6, realtime=True))
    worker.start()
    deadline = time.time() + 5
    while not got["suggestions"] and time.time() < deadline:
        time.sleep(0.05)
    worker.stop(wait=2)
    assert not worker.running and got["suggestions"]
