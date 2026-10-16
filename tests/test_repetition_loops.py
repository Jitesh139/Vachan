"""Whisper repetition loops on a live laptop mic (logs/pipeline.log, 2026-10-16 17:30-17:40).

Whisper sometimes repeats prompt words for the rest of a window. Loops are cut where they
start, so the repeated words cannot join a book name in the parser buffer.
"""

import numpy as np
import pytest

from tests.test_pipeline import FakeSTT, bible, make  # noqa: F401  (fixture)
from tests.test_verse_upgrade import run
from vachan.core.pipeline import cut_repetition

LOOP = "अध्याय, वचन, पहला, पहला, पहला, पहला, पहला, पहला"

# verbatim, truncated
REAL_LOOPS = [
    ("अध्याय, पत्री, अध्याय, पत्री, अध्याय, पत्री, अध्याय, पत्री, अध्याय, अध्याय, अध्याय, अध्य", ""),
    ("अध्याय, वचन, पहला, पहला, पहला, पहला, पहला, पहला, पहला, पहला, पहला, पहला, पहला", "अध्याय, वचन,"),
    ("अध्याय, मत्ती, वचन, वचन, वचन, वचन, वचन, वचन, वचन, वचन, वचन", "अध्याय, मत्ती,"),
    ("नॉल, वचन, वचन, वचन, वचन, वचन, वचन, वचन, वचन, वचन, वच", "नॉल,"),
    ("अध्याय, वचन, पत्री, अध्याय, वचन, पत्री, अध्याय, वचन, पत्री, वचन, पत्री, वचन", ""),
    (
        "अध्याय, अध्याय, दूसरा, चौथा, यूसको, हम कह सेते हैं, यूसको, हम कह सेते हैं, यूसको, हम कह सेते हैं",
        "अध्याय, अध्याय, दूसरा, चौथा,",
    ),
]


@pytest.mark.parametrize("text,kept", REAL_LOOPS)
def test_real_loops_are_cut(text, kept):
    assert cut_repetition(text).strip() == kept


@pytest.mark.parametrize(
    "text",
    [
        "यूहन्ना, रचित सुसमाचार, पहला, ध्याय, दूसरा, वचन.",
        "Turn to John chapter 3 verse 16.",
        "Psalm 119 verse 105, one hundred and five",
        "पहले अध्याय का पहला वचन",
        "नीतिवचन तीसरे अध्याय पाँचवें और छठे पद",
        "holy holy holy is the Lord",  # a word three times is not yet a loop
    ],
)
def test_normal_speech_untouched(text):
    assert cut_repetition(text) == text


def test_reference_before_a_loop_survives():
    assert cut_repetition("यूहन्ना 3 अध्याय 16 पद, पद, पद, पद, पद") == "यूहन्ना 3 अध्याय 16 "


def test_uncut_loop_would_invent_a_verse(bible):  # noqa: F811
    _, got = run(bible, ["आज हम मत्ती", LOOP, 5.0])  # the danger the cut removes
    assert [s.reference.format_reference() for s in got["suggestions"]] == ["Matthew 1:1"]


def test_cut_loop_after_book_gives_no_reference(bible):  # noqa: F811
    _, got = run(bible, ["आज हम मत्ती", cut_repetition(LOOP), 5.0])
    assert got["suggestions"] == []


def test_transcribe_applies_the_cut(bible):  # noqa: F811
    worker, _ = make([LOOP], bible, stt=FakeSTT([LOOP]))
    _, text, _, _ = worker._transcribe(np.zeros(16000, np.float32), "final")
    assert text.strip() == "अध्याय, वचन,"
