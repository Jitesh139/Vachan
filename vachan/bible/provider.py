"""Offline SQLite Bible provider implementing the BibleProvider port."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from vachan.core.models import BibleReference, Passage
from vachan.core.paths import DATA_DIR

DEFAULT_DB = DATA_DIR / "bibles" / "vachan_bibles.db"
FALLBACK_VERSION = "KJV"
MAX_RANGE = 12  # longer ranges do not fit on a broadcast card

# Spoken/abbreviated names that map onto an installed translation
VERSION_ALIASES = {"HINDI": "HIN_IRV", "HIN": "HIN_IRV", "IRV": "HIN_IRV"}


class BibleDataMissing(RuntimeError):
    """Raised when the Bible database has not been built yet."""


class LocalSqliteBibleProvider:
    """Read-only lookups against data/bibles/vachan_bibles.db."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB)
        if not self.db_path.exists():
            raise BibleDataMissing(
                f"Bible database not found at {self.db_path}. "
                "Run run.bat or: python scripts\\download_bibles.py"
            )
        self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            rows = self._conn.execute("SELECT id, name, license FROM versions").fetchall()
        self.version_info = {r[0]: (r[1], r[2]) for r in rows}

    def versions(self) -> list[str]:
        """Installed translation codes."""
        return list(self.version_info)

    def is_version_supported(self, version: str) -> bool:
        """True if the translation is installed offline."""
        v = version.upper()
        return VERSION_ALIASES.get(v, v) in self.version_info

    def _fetch(self, version: str, ref: BibleReference, end: int) -> list[tuple[int, str]]:
        with self._lock:
            return self._conn.execute(
                "SELECT verse, text FROM verses WHERE version_id=? AND book_id=? "
                "AND chapter=? AND verse BETWEEN ? AND ? ORDER BY verse",
                (version, ref.book_id, ref.chapter, ref.verse_start, end),
            ).fetchall()

    def get_passage(self, ref: BibleReference, version: str) -> Optional[Passage]:
        """Look up a reference, falling back to KJV with a notice when needed."""
        requested = VERSION_ALIASES.get(version.upper(), version.upper())
        notices = []
        end = ref.verse_end or ref.verse_start
        if end - ref.verse_start + 1 > MAX_RANGE:
            end = ref.verse_start + MAX_RANGE - 1
            notices.append(f"Range shortened to {MAX_RANGE} verses")
        used = requested
        if used not in self.version_info:
            used = FALLBACK_VERSION
            notices.append(f"{requested} is not available offline - showing {used}")
        rows = self._fetch(used, ref, end)
        if not rows and used != FALLBACK_VERSION:
            notices.append(f"Verse not in {used} - showing {FALLBACK_VERSION}")
            used = FALLBACK_VERSION
            rows = self._fetch(used, ref, end)
        if not rows:
            return None
        return Passage(verses=tuple(rows), version=used, notice="; ".join(notices))

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
