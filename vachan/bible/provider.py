"""Local SQLite Bible provider implementing the BibleProvider protocol.

Provides fast offline lookup across local SQLite databases.
Supports KJV, WEB, ASV, and Hindi (HIN_IRV).
"""

import sqlite3
from pathlib import Path
from typing import Optional


class LocalSqliteBibleProvider:
    """Offline Bible lookup provider powered by SQLite."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent.parent / "data" / "bibles" / "vachan_bibles.db"

        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Verify the database file exists."""
        if not self.db_path.exists():
            # If not yet created, run download_bibles initializer
            from scripts.download_bibles import main as init_bibles
            init_bibles()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def is_version_supported(self, version: str) -> bool:
        """Check if the given version code exists in SQLite."""
        norm = version.upper()
        if norm in ("HINDI", "HIN", "IRV"):
            norm = "HIN_IRV"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM versions WHERE id = ?", (norm,))
            row = cursor.fetchone()
            return row is not None
        finally:
            conn.close()

    def get_verse_text(
        self,
        book_id: str,
        chapter: int,
        verse_start: int,
        verse_end: Optional[int] = None,
        version: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch verse text from local SQLite database."""
        v_code = (version or "KJV").upper()
        if v_code in ("HINDI", "HIN", "IRV"):
            v_code = "HIN_IRV"

        # Check if version exists; if not, fallback gracefully
        if not self.is_version_supported(v_code):
            v_code = "KJV"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if verse_end and verse_end > verse_start:
                cursor.execute("""
                    SELECT verse, text FROM verses
                    WHERE version_id = ? AND book_id = ? AND chapter = ? AND verse >= ? AND verse <= ?
                    ORDER BY verse ASC
                """, (v_code, book_id.upper(), chapter, verse_start, verse_end))
                rows = cursor.fetchall()
                if not rows:
                    return None
                return " ".join(f"[{v}] {t}" for v, t in rows)
            else:
                cursor.execute("""
                    SELECT text FROM verses
                    WHERE version_id = ? AND book_id = ? AND chapter = ? AND verse = ?
                """, (v_code, book_id.upper(), chapter, verse_start))
                row = cursor.fetchone()
                return row[0] if row else None
        finally:
            conn.close()
