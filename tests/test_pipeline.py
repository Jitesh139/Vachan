"""Pipeline tests with fake audio/STT adapters (no microphone or model needed)."""

import time

import numpy as np
import pytest

from vachan.bible.provider import DEFAULT_DB, LocalSqliteBibleProvider
from vachan.core.models import BibleReference, Transcript
from vachan.core.pipeline import PipelineEvents, PipelineWorker, new_words
from vachan.core.settings import Settings
from vachan.parser import ReferenceParser

pytestmark = pytest.mark.skipif(not DEFAULT_DB.exists(), reason="Bible DB not built")


class FakeAudio:
    device_name, sample_rate = "fake", 16000

    def __init__(self, loud=True, fail=False):
        self.loud, self.fail = loud, fail

    def start(self):
        if self.fail:
            raise RuntimeError("Could not open 'fake': device unplugged")

    def stop(self):
        pass

    def get_level(self):
        return 0.5

    def read_latest(self, seconds):
        return np.full(int(seconds * 16000), 0.1 if self.loud else 0.0, np.float32)

    def seconds_since_audio(self):
        return 0.0


class FakeSTT:
    device, model_name = "cpu", "fake"

    def __init__(self, texts, fail_load=False):
        self.texts, self.fail_load = list(texts), fail_load

    def load(self, progress=None):
        if self.fail_load:
            raise RuntimeError("Whisper model 'small' is not downloaded")

    def has_speech(self, audio):
        return True

    def transcribe(self, audio):
        return Transcript(self.texts.pop(0) if self.texts else "", "en", 0.01)


@pytest.fixture(scope="module")
def bible():
    return LocalSqliteBibleProvider()


def make(texts, bible, **kw):
    got = {"suggestions": [], "errors": [], "debug": [], "transcript": []}
    events = PipelineEvents(
        suggestion=got["suggestions"].append,
        error=got["errors"].append,
        debug=got["debug"].append,
        transcript=lambda t, lang: got["transcript"].append(t),
    )
    audio = kw.pop("audio", None) or FakeAudio()
    worker = PipelineWorker(audio, FakeSTT(texts), ReferenceParser(), bible, Settings(**kw), events)
    return worker, got


def test_transcript_to_suggestion_with_both_languages(bible):
    worker, got = make([" Turn to John chapter 3 verse 16."], bible)
    worker._tick()
    (s,) = got["suggestions"]
    assert s.reference.format_reference() == "John 3:16"
    assert s.primary.version == "KJV" and "For God so loved" in s.primary.text
    assert s.secondary.version == "HIN_IRV" and "परमेश्वर" in s.secondary.text
    assert "MATCH John 3:16" in got["debug"][0].parser_output


def test_reference_split_across_windows(bible):
    worker, got = make(["Turn with me to Romans", "chapter 8 verse 28 please"], bible)
    worker._tick()
    worker._tick()
    assert [s.reference.format_reference() for s in got["suggestions"]] == ["Romans 8:28"]


def test_overlapping_windows_do_not_duplicate(bible):
    texts = ["Turn to John chapter 3 verse 16", "John chapter 3 verse 16 for God so loved"]
    worker, got = make(texts, bible)
    worker._tick()
    worker._tick()
    assert len(got["suggestions"]) == 1
    assert "duplicate" in got["debug"][1].parser_output
    assert got["transcript"] == ["Turn to John chapter 3 verse 16", "for God so loved"]


def test_follow_up_uses_approved_context(bible):
    worker, got = make(["Now look at the next verse."], bible)
    worker.set_context(BibleReference("JHN", "John", 3, 16))
    worker._tick()
    assert got["suggestions"][0].reference.format_reference() == "John 3:17"


def test_chapter_only_waits_one_step_for_verse(bible):
    worker, got = make(["Let us turn to Romans 8", "Let us turn to Romans 8 verse 28"], bible)
    worker._tick()
    assert got["suggestions"] == []
    worker._tick()
    assert got["suggestions"][0].reference.format_reference() == "Romans 8:28"


def test_unavailable_version_falls_back_with_notice(bible):
    worker, got = make(["Turn to Ephesians 2 8 in the NIV"], bible)
    worker._tick()
    s = got["suggestions"][0]
    assert s.primary.version == "KJV" and "NIV is not available offline" in s.primary.notice


def test_silence_skips_stt(bible):
    worker, got = make(["should not be used"], bible, audio=FakeAudio(loud=False))
    worker._tick()
    assert "silence" in got["debug"][0].parser_output and not got["suggestions"]


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
    worker, got = make(["Turn to John 3 16"] * 50, bible, step_sec=0.05)
    worker.start()
    deadline = time.time() + 5
    while not got["suggestions"] and time.time() < deadline:
        time.sleep(0.05)
    worker.stop(wait=2)
    assert not worker.running and got["suggestions"]


def test_new_words_merges_overlap():
    assert new_words("", "hello there") == "hello there"
    assert new_words("a b c d e", "c d e f g") == "f g"
    assert new_words("completely different", "new words here") == "new words here"
