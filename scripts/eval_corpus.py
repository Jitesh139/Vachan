"""Replay the missed-verse corpus (data/corpus/missed.jsonl) through the parser.

    python scripts/eval_corpus.py [--file other.jsonl]

Each entry's transcript segments are fed in order, the way the pipeline does: the parser
reads the last few segments said within context_sec, with the reference that was on screen
as context. An entry passes if the expected reference comes out at any point.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vachan.core.corpus import MISSED_FILE, load_missed  # noqa: E402
from vachan.core.settings import Settings  # noqa: E402
from vachan.parser import ReferenceParser  # noqa: E402

BUFFER = 3  # segments in the pipeline's rolling parser buffer (_State.texts)


def replay(entry: dict, parser: ReferenceParser) -> tuple[bool, list[str]]:
    """(expected found?, every reference the parser produced along the way)."""
    ctx = parser.parse(entry["context"], typed=True) if entry.get("context") else None
    context = ctx.reference if ctx and ctx.is_valid else None
    window = Settings().context_sec
    texts: deque = deque(maxlen=BUFFER)
    found: list[str] = []
    for seg in entry["segments"]:
        texts.append((seg["t"], seg["text"]))
        while texts and seg["t"] - texts[0][0] > window:
            texts.popleft()
        res = parser.parse(" ".join(t for _, t in texts), context_reference=context)
        if res and res.is_valid:
            ref = res.reference.format_reference()
            if ref not in found:
                found.append(ref)
            texts.clear()  # matched words must not trigger again (as in the pipeline)
    return entry["expected"] in found, found


def main() -> int:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path, default=MISSED_FILE)
    args = ap.parse_args()
    entries = load_missed(args.file)
    if not entries:
        print(f"No entries in {args.file}. Press 'Missed verse' (F7) during a service to add some.")
        return 0
    parser = ReferenceParser()
    failures = []
    for i, entry in enumerate(entries, 1):
        ok, found = replay(entry, parser)
        if not ok:
            failures.append((i, entry, found))
    passed = len(entries) - len(failures)
    print(f"{passed}/{len(entries)} found ({100 * passed / len(entries):.0f}%)")
    for i, entry, found in failures:
        print(f"\n#{i} {entry.get('time', '')}  expected {entry['expected']}  got {found or 'nothing'}")
        if entry.get("context"):
            print(f"   on screen: {entry['context']}")
        for seg in entry["segments"]:
            print(f"   {seg['t']:>6}s [{seg.get('lang', '?')}] {seg['text']}")
        for path in entry.get("audio", []):
            print(f"   audio: {path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
