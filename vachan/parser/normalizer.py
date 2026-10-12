"""Text normalization and number parsing for Whisper output (English, Hindi, Hinglish).

Whisper output differs from typed text in predictable ways (measured, see docs/AUDIT.md):
commas between every word in Hindi ("यूहन्ना, तीन, अध्याय, सोलह, पद."), English loanwords
written in Devanagari ("चैप्टर", "रोमेंस"), misspelled अध्याय ("अद्याय", "अध्याइ"),
"13-4" for "thirteen four", Devanagari digits, nukta/chandrabindu variants.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

from vachan.core.paths import DATA_DIR

ENGLISH_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12,
}  # fmt: skip
ENGLISH_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
HINDI_ORDINALS = {
    "पहला": 1, "पहली": 1, "पहले": 1, "दूसरा": 2, "दूसरी": 2, "दूसरे": 2,
    "तीसरा": 3, "तीसरी": 3, "तीसरे": 3, "चौथा": 4, "चौथी": 4, "चौथे": 4,
    "छठा": 6, "छठी": 6, "छठे": 6,
}  # fmt: skip
# Romanized Hindi number words that are also everyday English/Hinglish words
AMBIGUOUS_NUMBER_WORDS = {"do", "no", "to", "so"}

DEV = "\u0900-\u097f"
_B, _E = f"(?<![{DEV}])", f"(?![{DEV}])"  # Devanagari word edges (\b fails on matras)
# Whisper spellings -> canonical token
SPELLING_FIXES = [
    (_B + "(?:अ[दध]्?या[इईये]?य?|अद्ध्याय)" + _E, "अध्याय"),
    (_B + "च[ैे]?प्टर्?स?" + _E, "chapter"),
    (_B + "(?:वर्सेस|वर्सेज|वर्सिस|वर्स)" + _E, "verse"),
    (_B + "नेक्स्ट" + _E, "next"),
    (_B + "(?:प्रीवियस|पिछला|पिछली|पिछले)" + _E, "previous"),
]
ORDINAL_DIGITS = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
HINDI_ORDINAL_SUFFIX = re.compile(_B + f"([{DEV}]+?)(वां|वें|वीं|वा|वे)" + _E)
TOKEN_RE = re.compile(r"\d+|[a-z]+|[\u0900-\u0963\u0966-\u097f]+|[:\-]")


def fold(text: str) -> str:
    """Unicode-fold: NFC, drop nukta, chandrabindu -> anusvara, lowercase."""
    text = unicodedata.normalize("NFD", text).replace("\u093c", "")
    text = unicodedata.normalize("NFC", text).replace("ँ", "ं")
    return text.lower()


class NumberNormalizer:
    """Normalizes transcripts and parses spoken/written numbers from tokens."""

    def __init__(self, data_path: Optional[Path] = None) -> None:
        path = data_path or DATA_DIR / "hindi_numbers.json"
        raw: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
        self.hindi = {fold(k): v for k, v in raw.items() if k not in AMBIGUOUS_NUMBER_WORDS}
        self.hindi.update(HINDI_ORDINALS)
        self._fixes = [(re.compile(p), r) for p, r in SPELLING_FIXES]

    def normalize(self, text: str) -> str:
        """Canonicalize a Whisper transcript or typed query for parsing."""
        text = fold(text).replace("’", "'").replace("‘", "'")
        text = re.sub(r"'s\b", "", text).replace("'", "")
        text = text.translate({0x0966 + i: str(i) for i in range(10)})  # ० - ९
        text = ORDINAL_DIGITS.sub(r"\1", text)
        text = re.sub(r"\b(" + "|".join(ENGLISH_TENS) + r")-", r"\1 ", text)
        for pattern, repl in self._fixes:
            text = pattern.sub(repl, text)
        text = HINDI_ORDINAL_SUFFIX.sub(self._strip_ordinal_suffix, text)
        text = re.sub(r"(\d)(?:\.|\s*:\s*)(\d)", r"\1:\2", text)
        text = re.sub(r"(\d)\s*[-–—]\s*(\d)", r"\1-\2", text)
        text = re.sub(r"(?<!\d)[-–—]|[-–—](?!\d)", " ", text)
        text = re.sub(r"[^\w\u0900-\u0963\u0966-\u097f:\- ]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _strip_ordinal_suffix(self, m: re.Match) -> str:
        # "सोलहवां" -> "सोलह", but leave real words that merely end in "वा"
        return m.group(1) if m.group(1) in self.hindi else m.group(0)

    def tokenize(self, normalized: str) -> list[str]:
        """Split normalized text into word, digit and separator tokens."""
        return TOKEN_RE.findall(normalized)

    def parse_number(self, tokens: list[str], i: int) -> Optional[tuple[int, int]]:
        """Parse a number starting at tokens[i]; return (value, next_index)."""
        if i >= len(tokens):
            return None
        tok = tokens[i]
        if tok.isdigit():
            return int(tok), i + 1
        if tok in ENGLISH_TENS:
            value = ENGLISH_TENS[tok]
            if i + 1 < len(tokens) and ENGLISH_NUMBERS.get(tokens[i + 1], 99) < 10:
                return value + ENGLISH_NUMBERS[tokens[i + 1]], i + 2
            return value, i + 1
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok in ("hundred", "सौ") or (tok in ("one", "a", "एक") and nxt in ("hundred", "सौ")):
            j = i + 1 if tok in ("hundred", "सौ") else i + 2
            if j < len(tokens) and tokens[j] in ("and", "और"):
                j += 1
            rest = self.parse_number(tokens, j)
            if rest and rest[0] < 100:
                return 100 + rest[0], rest[1]
            return 100, j
        if tok in ENGLISH_NUMBERS:
            return ENGLISH_NUMBERS[tok], i + 1
        if tok in self.hindi:
            return self.hindi[tok], i + 1
        return None

    def parse_number_token(self, token: str) -> Optional[int]:
        """Parse a single word/phrase as a number, or None."""
        tokens = self.tokenize(self.normalize(token))
        parsed = self.parse_number(tokens, 0) if tokens else None
        return parsed[0] if parsed and parsed[1] == len(tokens) else None
