"""Approval policy: when may a suggestion go on air without the operator?

A wrong verse on a live stream is worse than none, so auto-show needs high confidence
AND no warnings (fuzzy book name, missing cue words, version fallback, ...).
"""

from vachan.core.models import Suggestion
from vachan.core.settings import Settings

REASON_TEXT = {
    "no_preacher_cue_words": "No cue words (turn to / chapter / अध्याय) - could be casual speech",
    "chapter_only_default_verse_1": "Only a chapter was heard - verse 1 assumed",
    "book_name_is_common_word": "Book name is also an everyday word",
    "invalid_chapter_or_verse_count": "Chapter/verse does not exist",
}


def should_auto_show(suggestion: Suggestion, settings: Settings) -> bool:
    """True only in Auto mode for confident, warning-free suggestions."""
    return (
        settings.auto_show
        and suggestion.result.confidence >= settings.auto_show_threshold
        and not suggestion.warnings
    )


def describe_warning(reason: str) -> str:
    """Operator-friendly text for a parser reason code or provider notice."""
    if reason.startswith("fuzzy_book_match: "):
        return "Book name guessed: " + reason.split(": ", 1)[1]
    if reason.startswith("number_split: "):
        return "Numbers ran together: " + reason.split(": ", 1)[1]
    if reason.startswith("relative_to: "):
        return "Relative to " + reason.split(": ", 1)[1]
    return REASON_TEXT.get(reason, reason)
