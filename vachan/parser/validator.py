"""Bible canon validation and verse count boundary checker.

Ensures extracted chapter and verse numbers are within canonical limits
(e.g., rejects John 99:1, Genesis 51:1, or Jude 2:1).
"""

import json
from pathlib import Path
from typing import Dict, Optional


class CanonValidator:
    """Validates book chapters and verse counts against Protestant 66-book canon."""

    def __init__(self, data_path: Optional[Path] = None) -> None:
        if data_path is None:
            data_path = Path(__file__).resolve().parent.parent.parent / "data" / "verse_counts.json"

        self.verse_counts: Dict[str, Dict[str, int]] = {}
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                self.verse_counts = json.load(f)

    def is_valid_chapter(self, book_id: str, chapter: int) -> bool:
        """Check if book has the specified chapter."""
        book_counts = self.verse_counts.get(book_id.upper())
        if not book_counts:
            return False
        return str(chapter) in book_counts

    def get_max_verse(self, book_id: str, chapter: int) -> Optional[int]:
        """Return the maximum verse count for the book and chapter, or None if invalid."""
        book_counts = self.verse_counts.get(book_id.upper())
        if not book_counts:
            return None
        return book_counts.get(str(chapter))

    def is_valid_reference(
        self,
        book_id: str,
        chapter: int,
        verse_start: int,
        verse_end: Optional[int] = None,
    ) -> bool:
        """Validate chapter and verse numbers."""
        max_verse = self.get_max_verse(book_id, chapter)
        if max_verse is None:
            return False

        if verse_start < 1 or verse_start > max_verse:
            return False

        if verse_end is not None:
            if verse_end < verse_start or verse_end > max_verse:
                return False

        return True
