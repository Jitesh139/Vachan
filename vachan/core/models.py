"""Domain models. Pure Python: no GUI, audio or ML imports allowed here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BibleReference:
    """A canonical scripture citation."""

    book_id: str  # USFM code, e.g. "JHN"
    book_name: str  # English name, e.g. "John"
    chapter: int
    verse_start: int
    verse_end: Optional[int] = None
    version: Optional[str] = None  # translation spoken by the preacher, e.g. "ESV"

    def format_reference(self) -> str:
        """'John 3:16' or 'John 3:16-18'."""
        verses = str(self.verse_start)
        if self.verse_end and self.verse_end != self.verse_start:
            verses += f"-{self.verse_end}"
        return f"{self.book_name} {self.chapter}:{verses}"

    @property
    def key(self) -> tuple:
        """Identity used for duplicate suppression (ignores version)."""
        return (self.book_id, self.chapter, self.verse_start, self.verse_end)


@dataclass(frozen=True)
class ParseResult:
    """Outcome of reference extraction from a text snippet."""

    reference: Optional[BibleReference]
    raw_match: str
    confidence: float  # 0.0 - 1.0
    is_follow_up: bool = False
    low_confidence_reasons: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """True if a usable reference was found."""
        return self.reference is not None and self.confidence > 0.0


@dataclass(frozen=True)
class Passage:
    """Verse text returned by a BibleProvider."""

    verses: tuple[tuple[int, str], ...]  # (verse number, text)
    version: str  # translation actually used
    notice: str = ""  # e.g. "ESV is not available offline - showing KJV"

    @property
    def text(self) -> str:
        """Plain text; verse numbers are prefixed only for ranges."""
        if len(self.verses) == 1:
            return self.verses[0][1]
        return " ".join(f"{n} {t}" for n, t in self.verses)


@dataclass(frozen=True)
class Suggestion:
    """A detected reference with its text, waiting for operator approval."""

    result: ParseResult
    primary: Passage
    secondary: Optional[Passage] = None  # Hindi alongside English (or vice versa)

    @property
    def reference(self) -> BibleReference:
        """The detected reference (always set for a suggestion)."""
        assert self.result.reference is not None
        return self.result.reference

    @property
    def warnings(self) -> tuple[str, ...]:
        """Human-readable reasons to double-check before showing."""
        notes = [p.notice for p in (self.primary, self.secondary) if p and p.notice]
        return tuple(self.result.low_confidence_reasons) + tuple(notes)


@dataclass(frozen=True)
class Transcript:
    """Output of a Transcriber for one audio window."""

    text: str
    language: str
    seconds: float  # inference time


@dataclass(frozen=True)
class ChunkDebugInfo:
    """Telemetry for one audio window processed by the pipeline."""

    timestamp: str
    rms: float
    chunk_length_sec: float
    device: str
    sample_rate: int
    raw_whisper_text: str
    detected_language: str
    stt_seconds: float
    text_passed_to_parser: str
    parser_output: str
    audio_saved_path: Optional[str] = None
