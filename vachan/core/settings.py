"""User settings, persisted as JSON next to the app (settings.json, git-ignored)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from vachan.core.paths import SETTINGS_FILE

log = logging.getLogger(__name__)

# Vocabulary only - never put chapter/verse numbers here: Whisper sometimes echoes the
# prompt, and a prompt containing "Romans 8:28" produced that reference out of thin air.
DEFAULT_PROMPT = (
    "Sermon with Bible readings. यूहन्ना, रोमियों, मत्ती, भजन संहिता, उत्पत्ति, "
    "अध्याय, पद. John, Romans, Matthew, Psalms, Genesis, chapter, verse."
)


@dataclass
class Settings:
    """All user-tunable options with safe defaults."""

    input_device: str = ""  # device name; "" = Windows default input
    input_gain: float = 1.0
    endpoint_sec: float = 0.5  # pause that ends a phrase (then it is transcribed once)
    max_segment_sec: float = 10.0  # run-on speech is committed in pieces this long
    partial_sec: float = 1.0  # live transcript refresh while speaking; 0 = off
    lang_reset_sec: float = 3.0  # silence after which the language is detected again
    silence_rms: float = 0.002  # below this audio is not even checked for speech
    profile: str = "auto"  # auto | accurate | balanced | fast | custom (= device/model below)
    device: str = "auto"  # auto | cuda | cpu
    model_gpu: str = "large-v3-turbo"
    model_cpu: str = "small"
    language: str = "auto"  # auto (Hindi/English per window) | hi | en
    beam_size: int = 5  # GPU only; on CPU decoding is always greedy (beam 1)
    cpu_threads: int = 0  # 0 = physical core count
    initial_prompt: str = DEFAULT_PROMPT
    primary_version: str = "KJV"
    secondary_version: str = "HIN_IRV"  # "" = show only one language
    display_mode: str = "both"  # both | primary | secondary
    theme: str = "Midnight"
    auto_show: bool = False  # Manual Approve is the safe default for live streams
    auto_show_threshold: float = 0.9
    duplicate_cooldown_sec: float = 45.0
    verse_wait_sec: float = 3.0  # chapter heard: wait this long for a verse before guessing 1
    context_sec: float = 15.0  # transcript kept for the parser (joins "Psalms 91" + "verse 2")
    save_debug_audio: bool = False

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> Settings:
        """Read settings; unknown keys are ignored, a broken file falls back to defaults."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            defaults = cls()
            kwargs = {}
            for f in fields(cls):
                value, default = data.get(f.name), getattr(defaults, f.name)
                if isinstance(default, float) and type(value) in (int, float):
                    kwargs[f.name] = float(value)
                elif type(value) is type(default):
                    kwargs[f.name] = value
            return cls(**kwargs)
        except (OSError, ValueError, TypeError) as e:
            log.warning("Ignoring unreadable settings file %s: %s", path, e)
            return cls()

    def save(self, path: Path = SETTINGS_FILE) -> None:
        """Write settings atomically."""
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
