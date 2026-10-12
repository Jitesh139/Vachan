"""Ports (protocols) between the domain core and the adapters.

Domain code depends only on these; adapters (sounddevice, faster-whisper, SQLite,
PySide6) implement them. No framework imports here.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

import numpy as np

from vachan.core.models import BibleReference, ParseResult, Passage, Suggestion, Transcript


class AudioSource(Protocol):
    """Live (or file) audio feeding a ring buffer."""

    device_name: str
    sample_rate: int  # native stream rate

    def start(self) -> None:
        """Open the stream; raises on failure."""

    def stop(self) -> None:
        """Close the stream."""

    def get_level(self) -> float:
        """Meter level 0..1."""

    def read_latest(self, seconds: float) -> np.ndarray:
        """Most recent audio as 16 kHz mono float32."""

    def seconds_since_audio(self) -> float:
        """Time since the last audio block arrived (stall / unplug detection)."""


class Transcriber(Protocol):
    """Speech-to-text engine."""

    device: str
    model_name: str

    def load(self, progress: Optional[Callable[[str], None]] = None) -> None:
        """Load (and download if needed) the model; idempotent."""

    def has_speech(self, audio: np.ndarray) -> bool:
        """Cheap voice-activity check."""

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Transcribe 16 kHz mono float32 audio."""


class BibleProvider(Protocol):
    """Scripture text lookup."""

    def get_passage(self, ref: BibleReference, version: str) -> Optional[Passage]:
        """Text for a reference, with fallback notice; None if not found."""

    def versions(self) -> list[str]:
        """Installed translation codes."""


class ReferenceResolver(Protocol):
    """Finds Bible references in text."""

    def parse(
        self,
        text: str,
        context_reference: Optional[BibleReference] = None,
        typed: bool = False,
    ) -> Optional[ParseResult]:
        """Return the newest reference in text (relative ones need context)."""


class OutputSink(Protocol):
    """Where approved verses are shown (the broadcast window)."""

    def show_suggestion(self, suggestion: Suggestion) -> None:
        """Display an approved verse."""

    def clear(self) -> None:
        """Blank the output."""
