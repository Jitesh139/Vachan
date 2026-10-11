"""Domain models for Bible references, parsing results, and validation.

This module is part of the pure Python domain core (vachan/core).
No external GUI, audio, or network dependencies are permitted here.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BibleReference:
    """Represents a canonical Bible scripture citation."""

    book_id: str             # e.g., "JHN", "ROM"
    book_name: str           # Canonical English name, e.g., "John"
    chapter: int             # e.g., 3
    verse_start: int         # e.g., 16
    verse_end: Optional[int] = None  # e.g., 18 for John 3:16-18
    version: Optional[str] = None    # e.g., "ESV", "KJV", "NIV"

    def format_reference(self, default_version: Optional[str] = None) -> str:
        """Format as human-readable string: e.g. 'John 3:16' or 'Romans 8:28 (ESV)'."""
        v = self.version or default_version
        if self.verse_end and self.verse_end != self.verse_start:
            verse_part = f"{self.verse_start}-{self.verse_end}"
        else:
            verse_part = f"{self.verse_start}"

        ref_str = f"{self.book_name} {self.chapter}:{verse_part}"
        if v:
            return f"{ref_str} ({v.upper()})"
        return ref_str


@dataclass(frozen=True)
class ParseResult:
    """Result of scripture reference extraction from a speech or text segment."""

    reference: Optional[BibleReference]
    raw_match: str                   # Exact matched substring
    confidence: float                # 0.0 to 1.0 confidence score
    is_follow_up: bool = False       # True if resolved via relative reference ("next verse")
    low_confidence_reasons: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """True if a reference was detected with positive confidence."""
        return self.reference is not None and self.confidence > 0.0
