"""Bug 1: "Psalms 91 verse 2" showed Psalm 91:1 (chapter-only default leaking out)."""

import pytest

from tests.test_pipeline import bible, make  # noqa: F401  (fixture)


def refs(got):
    return [s.reference.format_reference() for s in got["suggestions"]]


def test_regression_psalms_91_verse_2_split_across_windows(bible):  # noqa: F811
    # exact sequence: verse number arrives after the book name scrolled out of the window
    worker, got = make(["Psalms 91", "Psalms 91 verse", "verse 2."], bible)
    for _ in range(3):
        worker._tick()
    assert refs(got)[-1] == "Psalms 91:2"
    assert "Psalms 91:1" not in refs(got)
