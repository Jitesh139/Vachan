"""Missed-verse corpus: what was said when Vachan missed a reference.

One JSON object per line in data/corpus/missed.jsonl (local only, git-ignored):
{"time", "expected": "John 2:1", "context": "John 1:51" | null,
 "segments": [{"t": seconds before the button press, "text", "lang"}], "audio": [paths]}
scripts/eval_corpus.py replays it through the parser.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from vachan.core.paths import CORPUS_DIR

MISSED_FILE = CORPUS_DIR / "missed.jsonl"
WINDOW_SEC = 30.0  # transcript kept for a "Missed verse" entry


def save_missed(
    segments: list[dict],
    expected: str,
    context: Optional[str],
    audio: list[str],
    path: Path = MISSED_FILE,
) -> dict:
    """Append one entry and return it."""
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "expected": expected,
        "context": context,
        "segments": segments,
        "audio": audio,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load_missed(path: Path = MISSED_FILE) -> list[dict]:
    """All entries; a missing file is an empty corpus."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
