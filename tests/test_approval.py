"""Auto-show must never put an uncertain verse on air."""

import pytest

from tests.test_pipeline import bible  # noqa: F401  (fixture)
from vachan.core.approval import should_auto_show
from vachan.core.pipeline import PipelineWorker
from vachan.core.settings import Settings
from vachan.parser import ReferenceParser


def test_manual_approve_is_the_default():
    assert Settings().auto_show is False


@pytest.fixture
def make(bible):  # noqa: F811
    parser = ReferenceParser()
    worker = PipelineWorker(None, None, parser, bible, Settings())

    def suggestion(text, action="new"):
        return worker.build_suggestion(parser.parse(text), action)

    return suggestion


AUTO = Settings(auto_show=True)


def test_clean_reference_auto_shows(make):
    assert should_auto_show(make("यूहन्ना रचित सुसमाचार पहला अध्याय दूसरा वचन"), AUTO)


def test_nothing_auto_shows_in_manual_mode(make):
    assert not should_auto_show(make("यूहन्ना रचित सुसमाचार पहला अध्याय दूसरा वचन"), Settings())


@pytest.mark.parametrize(
    "text,why",
    [
        ("Turn to Psalms 91", "chapter only / guessed verse"),
        ("turn to romens 8 28", "fuzzy book name"),
        ("यूहन्ना रचित सुसमाचार पहला दूसरा", "no chapter/verse words"),
        ("यूहन्ना रचित सुसमाचार पहला ध्याय हम देखते हैं दूसरा वचन", "words between"),
        ("Acts 2 38", "no cue words, below threshold"),
        ("यूहन्ना की पत्री चौथा अध्याय आठवाँ पद", "letter number not spoken"),
    ],
)
def test_uncertain_never_auto_shows(make, text, why):
    s = make(text, "guess" if "Psalms 91" in text else "new")
    assert not should_auto_show(s, AUTO), why
