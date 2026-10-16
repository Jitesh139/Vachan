"""Missed-verse corpus: saved from the control window, replayed by scripts/eval_corpus.py."""

import pytest

from scripts.eval_corpus import replay
from vachan.core.corpus import load_missed, save_missed
from vachan.parser import ReferenceParser


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "missed.jsonl"
    save_missed([{"t": -4.0, "text": "यूहन्ना रचित सुसमाचार", "lang": "hi"}], "John 2:1", None, [], path)
    save_missed([{"t": -1.0, "text": "agla vachan", "lang": "hi"}], "John 2:2", "John 2:1", [], path)
    entries = load_missed(path)
    assert [e["expected"] for e in entries] == ["John 2:1", "John 2:2"]
    assert entries[0]["segments"][0]["text"] == "यूहन्ना रचित सुसमाचार"


def test_load_missing_file_is_empty(tmp_path):
    assert load_missed(tmp_path / "none.jsonl") == []


@pytest.fixture(scope="module")
def parser():
    return ReferenceParser()


def test_replay_finds_reference_split_across_segments(parser):
    entry = {
        "expected": "John 2:1",
        "context": None,
        "segments": [
            {"t": -6, "text": "आइए हम खोलें यूहन्ना रचित सुसमाचार।"},
            {"t": -3, "text": "दूसरा अध्याय, पहला वचन।"},
        ],
    }
    ok, found = replay(entry, parser)
    assert ok and "John 2:1" in found


def test_replay_uses_context_for_follow_ups(parser):
    entry = {"expected": "John 2:2", "context": "John 2:1", "segments": [{"t": -1, "text": "अगला वचन"}]}
    assert replay(entry, parser)[0]


def test_replay_reports_miss(parser):
    entry = {"expected": "John 2:1", "context": None, "segments": [{"t": -1, "text": "परमेश्वर का वचन"}]}
    assert replay(entry, parser) == (False, [])
