"""Architectural ports (protocols) for the Vachan application.

Per VACHAN OVERRIDES in skills/hexagonal-architecture.md:
- AudioSource: Audio capture / stream input
- Transcriber: Speech-to-text transcription
- BibleProvider: Scripture text retrieval
- ReferenceResolver: High-level reference detection / parsing
- OutputSink: Display / broadcasting output destination

Pure Python typing protocols only. No external framework imports.
"""

from typing import Any, Callable, Optional, Protocol
from vachan.core.models import BibleReference, ParseResult


class AudioSource(Protocol):
    """Port for audio input streams (microphone, line-in, or file source)."""

    def start(self, callback: Callable[[Any], None]) -> None:
        """Start streaming audio chunks to callback."""
        ...

    def stop(self) -> None:
        """Stop audio streaming."""
        ...

    def get_level(self) -> float:
        """Get current audio volume level (0.0 to 1.0) for meter."""
        ...


class Transcriber(Protocol):
    """Port for speech-to-text models (e.g. faster-whisper)."""

    def transcribe(self, audio_data: Any) -> str:
        """Transcribe audio chunk and return transcribed text."""
        ...


class BibleProvider(Protocol):
    """Port for Bible scripture text lookups (SQLite, ESV API, API.Bible)."""

    def get_verse_text(
        self,
        book_id: str,
        chapter: int,
        verse_start: int,
        verse_end: Optional[int] = None,
        version: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch verse text for given reference. Return None if not found."""
        ...

    def is_version_supported(self, version: str) -> bool:
        """Check if provider supports a given translation code."""
        ...


class ReferenceResolver(Protocol):
    """Port for parsing and resolving Bible references from text snippets."""

    def resolve(
        self,
        text: str,
        context_reference: Optional[BibleReference] = None,
    ) -> Optional[ParseResult]:
        """Parse text and extract Bible reference, using context if relative."""
        ...


class OutputSink(Protocol):
    """Port for broadcast output sinks (PySide6 window, NDI, Browser Source)."""

    def show_verse(
        self,
        reference: BibleReference,
        text_primary: str,
        text_secondary: Optional[str] = None,
    ) -> None:
        """Display approved verse on the output."""
        ...

    def clear(self) -> None:
        """Clear the current broadcast output."""
        ...
