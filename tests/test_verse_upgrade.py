"""Bug 1: "Psalms 91 verse 2" showed Psalm 91:1 (chapter-only default leaking out)."""

import logging

import pytest

from tests.test_pipeline import bible, make  # noqa: F401  (fixture)
from vachan.parser import ReferenceParser


def refs(got):
    return [s.reference.format_reference() for s in got["suggestions"]]


def test_regression_psalms_91_verse_2_split_across_windows(bible):  # noqa: F811
    # exact transcripts from the bug: the verse number arrived after the book name had
    # scrolled out of the old 7 s window; the guess 91:1 went out and cleared the buffer
    worker, got = make([], bible)
    for text in ["Psalms 91", "Psalms 91 verse", "verse 2."]:
        _, s = worker._handle_text(text)
        got["suggestions"] += [s] if s else []
    assert refs(got) == ["Psalms 91:2"]


def test_regression_psalms_91_verse_2_as_spoken_phrases(bible):  # noqa: F811
    from tests.test_pipeline import utter

    worker, got = make(["Psalms 91", "verse 2."], bible)  # speaker paused after the chapter
    utter(worker, 2)
    assert refs(got) == ["Psalms 91:2"]


# ------------------------------------------------------------------ parser: one window
@pytest.fixture(scope="module")
def parser():
    return ReferenceParser()


def fmt(res):
    return res.reference.format_reference() if res and res.is_valid else None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Psalms 91 verse 2", "Psalms 91:2"),
        ("Psalm 91, 2", "Psalms 91:2"),  # Whisper punctuation
        ("Psalm 91, 2.", "Psalms 91:2"),
        ("Psalm 91. 2.", "Psalms 91:2"),
        ("Psalms 91, verse 2.", "Psalms 91:2"),
        ("Psalms 91. Verse 2.", "Psalms 91:2"),
        ("Psalms 91 verse to", "Psalms 91:2"),  # homophones after a verse cue
        ("Psalms 91 verse too", "Psalms 91:2"),
        ("Psalms 91 verse tu", "Psalms 91:2"),
        ("Psalms 91 verse two", "Psalms 91:2"),
        ("Psalms 91 verse for", "Psalms 91:4"),
        ("Psalm ninety one verse two", "Psalms 91:2"),
        ("Psalms 91 and verse 2", "Psalms 91:2"),
        ("Psalms chapter 91 verse 2", "Psalms 91:2"),
        ("Psalm 91 verse number 2", "Psalms 91:2"),
        ("Psalms 91:2", "Psalms 91:2"),
        ("Psalms 91:12", "Psalms 91:12"),
        ("Psalms 119:105", "Psalms 119:105"),
        ("Psalm 119 verse 105", "Psalms 119:105"),
        ("Psalms 91 verses 2 to 4", "Psalms 91:2-4"),  # 'to' after a number is a range
        ("Psalms 91 verses 2 through 4", "Psalms 91:2-4"),
        ("Psalms 91 verse two to four", "Psalms 91:2-4"),
        ("Psalms 91, verse 2, he that dwelleth", "Psalms 91:2"),
        ("भजन संहिता 91 पद 2", "Psalms 91:2"),
        ("भजन संहिता 91 पद दो", "Psalms 91:2"),
        ("भजन संहिता इक्यानवे पद दो", "Psalms 91:2"),
        ("भजन संहिता 91 वचन 2", "Psalms 91:2"),
        ("भजन संहिता के 91 अध्याय का 2 पद", "Psalms 91:2"),
        ("भजन संहिता 91 अध्याय और 2 पद", "Psalms 91:2"),
        ("भजन संहिता इक्यानवे अध्याय दूसरा पद", "Psalms 91:2"),
        ("भजन संहिता 91 पद 2 से 4", "Psalms 91:2-4"),
        ("Psalm 91 वर्स 2", "Psalms 91:2"),  # Hinglish: English word in Devanagari
        ("Bhajan Sanhita 91 pad 2", "Psalms 91:2"),
        ("Bhajan Sanhita 91 pad do", "Psalms 91:2"),
        ("Bhajan Sanhita 91 adhyay 2 pad", "Psalms 91:2"),
        ("zabur 91 pad 2", "Psalms 91:2"),
        # spellings seen in real Whisper Hindi output
        ("सामस 91 वरस 2", "Psalms 91:2"),
        ("भजन संहिता 91 वर्स्ट 2", "Psalms 91:2"),
        ("Psalms 91 पद डो", "Psalms 91:2"),
        ("भजन संहिता 91 पद टू", "Psalms 91:2"),
        ("आज हम सांस 91, verse 2 पढ़ेंगे.", "Psalms 91:2"),  # large-v3-turbo on the iGPU
        ("रोमियों, आठ, अध्याय, अठ्थाईस, पद.", "Romans 8:28"),
    ],
)
def test_psalms_91_verse_2_forms(parser, text, expected):
    assert fmt(parser.parse(text)) == expected


@pytest.mark.parametrize("text", ["Psalms 91", "Turn to Psalms 91", "भजन संहिता 91", "Psalms 91 verse",
                                  "भजन संहिता 91 पद", "Psalms chapter 91"])  # fmt: skip
def test_chapter_only_has_no_verse(parser, text):
    res = parser.parse(text)
    assert res.reference.verse_start is None
    assert res.reference.format_reference() == "Psalms 91"
    assert "chapter_only_no_verse" in res.low_confidence_reasons


def test_run_together_912_is_flagged_ambiguous(parser):
    res = parser.parse("Psalm 912")
    assert fmt(res) == "Psalms 9:12"
    assert "number_split: 912 -> 9:12 (or 91:2)" in res.low_confidence_reasons


def test_homophone_not_read_without_cue(parser):
    assert parser.parse("Psalms 91 to").reference.verse_start is None


# ------------------------------------------------------------------ pipeline: split windows
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def run(bible, steps):  # noqa: F811
    """steps: transcript strings (1 s apart) or float seconds of silence."""
    worker, got = make([], bible)
    got["lines"] = []
    worker._clock = clock = Clock()
    for step in steps:
        if isinstance(step, float):
            clock.t += step
        else:
            clock.t += 1.0
            line, s = worker._handle_text(step)
            got["lines"].append(line)
            if s:
                got["suggestions"].append(s)
        s = worker._check_pending()
        if s:
            got["suggestions"].append(s)
    return worker, got


@pytest.mark.parametrize(
    "steps",
    [
        ["Psalms 91", "verse 2"],
        ["Psalms 91", "verse 2."],
        ["Psalms 91 verse", "2"],
        ["Psalms 91 verse", "to"],
        ["Psalms 91,", "verse two."],
        ["Turn to Psalms 91", "and verse 2 says"],
        ["भजन संहिता 91", "पद 2"],
        ["भजन संहिता 91", "पद दो"],
        ["भजन संहिता 91 अध्याय", "और 2 पद"],
        ["Bhajan Sanhita 91", "pad do"],
        ["Psalms 91", "verse 2", "verse 2"],  # overlapping windows repeat the verse
    ],
)
def test_partial_then_complete_gives_one_correct_suggestion(bible, steps):  # noqa: F811
    _, got = run(bible, steps)
    assert refs(got) == ["Psalms 91:2"]
    assert got["suggestions"][0].action == "new"
    assert not got["suggestions"][0].is_guess


def test_chapter_only_waits_before_guessing(bible):  # noqa: F811
    _, got = run(bible, ["Psalms 91", 1.9])  # 2.9 s after the chapter was heard
    assert refs(got) == []


def test_chapter_only_guess_is_marked(bible):  # noqa: F811
    _, got = run(bible, ["Psalms 91", 3.5])
    (s,) = got["suggestions"]
    assert s.reference.format_reference() == "Psalms 91:1"
    assert s.is_guess and s.action == "guess"
    assert "verse_guessed" in s.warnings


def test_late_verse_upgrades_guess(bible):  # noqa: F811
    _, got = run(bible, ["Psalms 91", 3.5, "verse 2"])
    assert refs(got) == ["Psalms 91:1", "Psalms 91:2"]
    assert [s.action for s in got["suggestions"]] == ["guess", "upgrade"]
    assert not got["suggestions"][1].is_guess


def test_late_verse_after_other_words_upgrades_guess(bible):  # noqa: F811
    steps = ["Turn to Psalms 91", 3.5, "He that dwelleth in the secret place", "verse 2 says"]
    _, got = run(bible, steps)
    assert refs(got) == ["Psalms 91:1", "Psalms 91:2"]


def test_late_verse_range_upgrades_guess(bible):  # noqa: F811
    _, got = run(bible, ["Psalms 91", 3.5, "verses 2 to 4"])
    assert refs(got)[-1] == "Psalms 91:2-4"


def test_hindi_late_verse_upgrades_guess(bible):  # noqa: F811
    _, got = run(bible, ["भजन संहिता 91", 3.5, "पद 2"])
    assert refs(got) == ["Psalms 91:1", "Psalms 91:2"]


def test_guess_not_repeated_while_chapter_stays_in_buffer(bible):  # noqa: F811
    _, got = run(bible, ["Psalms 91", 3.5, "Psalms 91", 3.5, "Psalms 91", 3.5])
    assert refs(got) == ["Psalms 91:1"]


def test_new_chapter_supersedes_waiting_one(bible):  # noqa: F811
    _, got = run(bible, ["Psalms 91", "John 3:16", 5.0])
    assert refs(got) == ["John 3:16"]


def test_dedup_key_includes_verse(bible):  # noqa: F811
    _, got = run(bible, ["Psalms 91 verse 2", "Psalms 91 verse 4", "Psalms 91 verse 4"])
    assert refs(got) == ["Psalms 91:2", "Psalms 91:4"]


def test_spoken_verse_1_after_guess_is_not_a_duplicate(bible):  # noqa: F811
    _, got = run(bible, ["Psalms 91", 3.5, "Psalms 91 verse 1"])
    assert refs(got) == ["Psalms 91:1", "Psalms 91:1"]
    assert got["suggestions"][1].action == "upgrade" and not got["suggestions"][1].is_guess


def test_next_verse_uses_screen_not_guess(bible):  # noqa: F811
    from vachan.core.models import BibleReference

    worker, _ = run(bible, ["Psalms 91", 3.5])  # guess 91:1 went out, operator did NOT show it
    worker.set_context(BibleReference("JHN", "John", 3, 16))  # what is actually on screen
    _, s = worker._handle_text("the next verse")
    assert s.reference.format_reference() == "John 3:17"


def test_every_decision_is_logged(bible):  # noqa: F811
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    trace = logging.getLogger("vachan.pipeline_trace")
    trace.addHandler(handler)
    try:
        worker, got = run(bible, ["Psalms 91", 3.5, "verse 2"])
    finally:
        trace.removeHandler(handler)
    assert any("action=guess ref=Psalms 91:1 (GUESS) verse_source=default" in r for r in records)
    assert "action=waiting ref=Psalms 91 verse_source=none" in got["lines"][0]
    assert "action=upgrade ref=Psalms 91:2 verse_source=spoken" in got["lines"][1]
    line, _ = worker._handle_text("Psalms 91 verse 2")
    assert "action=duplicate ref=Psalms 91:2" in line
