"""Download public-domain / openly licensed Bibles and build the offline SQLite DB.

Source: eBible.org verse-per-line (VPL) files.

| Code    | Translation                    | License                                   |
|---------|--------------------------------|-------------------------------------------|
| KJV     | King James Version (1769)      | Public Domain                             |
| ASV     | American Standard Version 1901 | Public Domain                             |
| WEB     | World English Bible            | Public Domain                             |
| HIN_IRV | Hindi Indian Revised Version   | CC BY-SA 4.0, (c) Bridge Connectivity Sol.|

Usage:  python scripts/download_bibles.py [--force]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sqlite3
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "bibles" / "vachan_bibles.db"

VERSIONS = {
    "KJV": ("eng-kjv2006", "King James Version", "en", "Public Domain"),
    "ASV": ("eng-asv", "American Standard Version", "en", "Public Domain"),
    "WEB": ("eng-web", "World English Bible", "en", "Public Domain"),
    "HIN_IRV": (
        "hin2017",
        "Hindi Indian Revised Version",
        "hi",
        "CC BY-SA 4.0 - (c) 2017-2019 Bridge Connectivity Solutions",
    ),
}
URL = "https://ebible.org/Scriptures/{}_vpl.zip"

# eBible VPL book codes that differ from the USFM codes used in data/books.json
VPL_TO_USFM = {
    "JOH": "JHN", "MAR": "MRK", "SOL": "SNG", "EZE": "EZK", "JOE": "JOL",
    "NAH": "NAM", "JAM": "JAS", "PHI": "PHP", "1JO": "1JN", "2JO": "2JN",
    "3JO": "3JN",
}  # fmt: skip

LINE_RE = re.compile(r"^(\w{3}) (\d+):(\d+) (.*)$")
# Hindi IRV appends cross references like "(इब्रा. 1:10, इब्रा. 11:3)"
CROSS_REF_RE = re.compile(r"\s*\([^()]*\d+:\d+[^()]*\)")


def clean_text(text: str) -> str:
    """Strip pilcrows, KJV italic brackets and inline cross references."""
    text = text.replace("¶", "")
    text = CROSS_REF_RE.sub("", text)
    text = re.sub(r"\[([^\]]*)\]", r"\1", text)
    # quotations spanning several verses leave a dangling mark on a single verse
    if text.count("“") != text.count("”"):
        text = text.replace("“", "").replace("”", "")
    return re.sub(r"\s+", " ", text).strip()


def parse_vpl(raw: str, valid_books: set[str]) -> list[tuple[str, int, int, str]]:
    """Parse a VPL file into (book, chapter, verse, text) rows for the 66-book canon."""
    rows = []
    for line in raw.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        book = VPL_TO_USFM.get(m.group(1), m.group(1))
        text = clean_text(m.group(4))
        if book in valid_books and text:
            rows.append((book, int(m.group(2)), int(m.group(3)), text))
    return rows


def download(ebible_id: str) -> str:
    """Download and unzip one VPL text file."""
    req = urllib.request.Request(URL.format(ebible_id), headers={"User-Agent": "Vachan"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return zf.read(f"{ebible_id}_vpl.txt").decode("utf-8-sig")


def build(db_path: Path = DB_PATH) -> None:
    """Build the database from scratch into a temp file, then swap it in."""
    verse_counts: dict[str, dict[str, int]] = json.loads(
        (DATA_DIR / "verse_counts.json").read_text(encoding="utf-8")
    )
    expected = sum(sum(c.values()) for c in verse_counts.values())
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    conn.executescript("""
        CREATE TABLE versions (id TEXT PRIMARY KEY, name TEXT NOT NULL,
            language TEXT NOT NULL, license TEXT NOT NULL, source TEXT NOT NULL);
        CREATE TABLE verses (version_id TEXT NOT NULL, book_id TEXT NOT NULL,
            chapter INTEGER NOT NULL, verse INTEGER NOT NULL, text TEXT NOT NULL,
            PRIMARY KEY (version_id, book_id, chapter, verse)) WITHOUT ROWID;
    """)
    for code, (ebible_id, name, lang, lic) in VERSIONS.items():
        print(f"Downloading {code} ({name}) ...", flush=True)
        rows = parse_vpl(download(ebible_id), set(verse_counts))
        conn.execute(
            "INSERT INTO versions VALUES (?, ?, ?, ?, ?)",
            (code, name, lang, lic, URL.format(ebible_id)),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO verses VALUES (?, ?, ?, ?, ?)",
            [(code, *r) for r in rows],
        )
        present = {(b, c, v) for b, c, v, _ in rows}
        missing = [
            f"{b} {c}:{v}"
            for b, chapters in verse_counts.items()
            for c, n in chapters.items()
            for v in range(1, n + 1)
            if (b, int(c), v) not in present
        ]
        print(f"  {len(rows)} verses, {expected - len(missing)}/{expected} canon verses")
        if missing:
            print(f"  missing {len(missing)}: {', '.join(missing[:8])}")
    conn.commit()
    conn.close()
    tmp.replace(db_path)
    print(f"Database ready: {db_path}")


def main() -> int:
    """CLI entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="rebuild even if DB exists")
    args = ap.parse_args()
    if DB_PATH.exists() and not args.force:
        print(f"Database already exists: {DB_PATH} (use --force to rebuild)")
        return 0
    try:
        build()
    except OSError as e:
        print(f"ERROR: could not download Bibles ({e}). Check internet and retry.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
