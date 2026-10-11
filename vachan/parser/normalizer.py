"""Number normalization for Bible reference parsing.

Handles Arabic digits ('1', '16'), Devanagari numerals ('१', '१६'),
English words ('three', 'sixteen', 'first'), and Hindi words / Romanized Hinglish
('तीन', 'सोलह', 'teen', 'solah', 'pehla').
"""

import json
from pathlib import Path
from typing import Dict, Optional

# Standard English number words to integers
ENGLISH_NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "first": 1, "two": 2, "second": 2,
    "three": 3, "third": 3, "four": 4, "fourth": 4, "five": 5, "fifth": 5,
    "six": 6, "sixth": 6, "seven": 7, "seventh": 7, "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9, "ten": 10, "tenth": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty-one": 21, "twenty-two": 22, "twenty-three": 23, "twenty-four": 24, "twenty-five": 25,
    "twenty-six": 26, "twenty-seven": 27, "twenty-eight": 28, "twenty-nine": 29, "thirty": 30,
    "thirty-one": 31, "thirty-two": 32, "thirty-three": 33, "thirty-four": 34, "thirty-five": 35,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "one hundred": 100,
}

DEVANAGARI_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}


class NumberNormalizer:
    """Normalizes numbers from digits, English words, and Hindi words."""

    def __init__(self, data_path: Optional[Path] = None) -> None:
        if data_path is None:
            data_path = Path(__file__).resolve().parent.parent.parent / "data" / "hindi_numbers.json"

        self.hindi_dict: Dict[str, int] = {}
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                self.hindi_dict = json.load(f)

    def convert_devanagari_digits_to_ascii(self, text: str) -> str:
        """Convert any Devanagari numerals in the text to ASCII digits (e.g. '१६' -> '16')."""
        out = []
        for ch in text:
            if ch in DEVANAGARI_DIGITS:
                out.append(DEVANAGARI_DIGITS[ch])
            else:
                out.append(ch)
        return "".join(out)

    def parse_number_token(self, token: str) -> Optional[int]:
        """Convert a single token or clean phrase into an integer, if recognizable."""
        token = token.strip().lower()
        if not token:
            return None

        # Check pure ASCII digits
        if token.isdigit():
            try:
                return int(token)
            except ValueError:
                return None

        # Check Devanagari digits
        ascii_candidate = self.convert_devanagari_digits_to_ascii(token)
        if ascii_candidate.isdigit():
            try:
                return int(ascii_candidate)
            except ValueError:
                return None

        # Check English number words
        if token in ENGLISH_NUMBER_WORDS:
            return ENGLISH_NUMBER_WORDS[token]

        # Check compound English numbers like "twenty five"
        parts = token.replace("-", " ").split()
        if len(parts) == 2 and parts[0] in ENGLISH_NUMBER_WORDS and parts[1] in ENGLISH_NUMBER_WORDS:
            tens = ENGLISH_NUMBER_WORDS[parts[0]]
            ones = ENGLISH_NUMBER_WORDS[parts[1]]
            if tens in (20, 30, 40, 50, 60, 70, 80, 90) and 1 <= ones <= 9:
                return tens + ones

        # Check Hindi number dictionary (Devanagari and Romanized)
        if token in self.hindi_dict:
            return int(self.hindi_dict[token])

        return None
