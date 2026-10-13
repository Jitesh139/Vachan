"""VAD endpointing: segments, partials, stable commits, language reuse, latency."""

from tests.test_pipeline import FakeAudio, FakeSTT, advance, bible, make, utter  # noqa: F401


def refs(got):
    return [s.reference.format_reference() for s in got["suggestions"]]


def test_phrase_committed_after_pause_with_latency(bible):  # noqa: F811
    worker, got = make(["Turn to Psalms 91 verse 2"], bible)
    advance(worker, 1.0)  # still speaking
    assert worker.transcriber.calls == []
    advance(worker, 1.0)  # paused
    assert refs(got) == ["Psalms 91:2"]
    (secs, _), = worker.transcriber.calls
    assert 1.0 <= secs <= 1.6  # the speech plus padding, not a fixed 7 s window
    (delay,) = got["latency"]
    assert 0.5 <= delay <= 0.8  # endpoint pause + one tick (fake STT is instant)


def test_partials_shown_but_reference_waits_for_stable(bible):  # noqa: F811
    def growing(secs):  # what Whisper would say about the first `secs` of a long sentence
        words = "Turn to Psalms 91 verse 2 he that dwelleth in the secret place".split()
        return " ".join(words[: max(1, int(secs * 2))])

    stt = FakeSTT(growing)
    worker, got = make([], bible, stt=stt, partial_sec=1.0, audio=FakeAudio(on=8.0, off=1.0))
    advance(worker, 3.0)  # partials so far end in "...verse 2" or "...he": not stable yet
    assert got["partial"] and refs(got) == []
    advance(worker, 2.0)  # two partials agree on 91:2 with words after it
    assert refs(got) == ["Psalms 91:2"]
    advance(worker, 4.0)  # endpoint: final transcript repeats it -> duplicate, not a new card
    assert refs(got) == ["Psalms 91:2"]
    assert got["transcript"] == [growing(8.25)]


def test_reference_at_end_of_partial_is_not_committed(bible):  # noqa: F811
    # "verse 1" could still become "verse 16" while the speaker is talking
    stt = FakeSTT(lambda secs: "Turn to John 3 verse 1" if secs < 3 else "Turn to John 3 verse 16")
    worker, got = make([], bible, stt=stt, partial_sec=1.0, audio=FakeAudio(on=3.5, off=1.0))
    advance(worker, 4.5)
    assert refs(got) == ["John 3:16"]


def test_language_detected_once_then_reused_until_long_silence(bible):  # noqa: F811
    stt = FakeSTT(["one", "two", "three"], language="hi")
    worker, _ = make([], bible, stt=stt, audio=FakeAudio(on=1.0, off=1.0), lang_reset_sec=3.0)
    utter(worker, 2)  # pause between phrases (1 s) is short: language reused
    worker.audio.loud = False
    advance(worker, 4.0)  # long silence
    worker.audio.loud = True
    utter(worker)
    assert [lang for _, lang in stt.calls] == [None, "hi", None]


def test_run_on_speech_cut_into_pieces(bible):  # noqa: F811
    stt = FakeSTT(lambda secs: "and so on")
    worker, got = make([], bible, stt=stt, audio=FakeAudio(on=25.0, off=1.0), max_segment_sec=10.0)
    advance(worker, 26.0)
    lengths = [secs for secs, _ in stt.calls]
    assert len(lengths) == 3 and all(secs <= 10.5 for secs in lengths)
    assert abs(sum(lengths) - 25.0) < 1.0  # nothing lost, nothing transcribed twice


def test_split_reference_across_cut_still_found(bible):  # noqa: F811
    stt = FakeSTT(["blah blah Turn to Psalms 91", "verse 2 he that dwelleth"])
    worker, got = make([], bible, stt=stt, audio=FakeAudio(on=14.0, off=1.0), max_segment_sec=10.0)
    advance(worker, 15.0)
    assert refs(got) == ["Psalms 91:2"]


def test_slow_machine_skips_partials(bible):  # noqa: F811
    stt = FakeSTT(lambda secs: "words")
    worker, got = make([], bible, stt=stt, partial_sec=1.0, audio=FakeAudio(on=6.0, off=1.0))
    worker._rtf = 2.0  # measured: STT twice as slow as real time
    advance(worker, 7.0)
    assert got["partial"] == [] and len(stt.calls) == 1


def test_short_noise_is_not_a_segment(bible):  # noqa: F811
    worker, got = make(["x"], bible, audio=FakeAudio(on=0.1, off=2.0))
    utter(worker, 2)
    assert worker.transcriber.calls == []
