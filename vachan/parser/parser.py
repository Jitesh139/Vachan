"""Bible reference parser implementing fast rule-based extraction.

Supports:
- English: 'John 3:16', 'John chapter three verse sixteen', 'turn to Romans 8', 'verses 5 to 9'
- Hindi: 'यूहन्ना 3:16', 'यूहन्ना तीन अध्याय सोलह पद', 'रोमियों 8 अध्याय 28 पद'
- Hinglish: 'Romans ke aath adhyay mein', 'Yuhanna teen solah'
- Follow-ups: 'next verse', 'previous verse', 'verse 17' relative to context
- Version detection: ESV, NIV, KJV, NLT, NKJV, NASB, ASV, WEB, BSI, IRV
- Confidence scoring and reason breakdown
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from vachan.core.models import BibleReference, ParseResult
from vachan.parser.normalizer import NumberNormalizer
from vachan.parser.validator import CanonValidator


KNOWN_VERSIONS = {
    "esv": "ESV",
    "niv": "NIV",
    "kjv": "KJV",
    "nlt": "NLT",
    "nkjv": "NKJV",
    "nasb": "NASB",
    "asv": "ASV",
    "web": "WEB",
    "bsi": "BSI",
    "irv": "IRV",
    "hindi": "IRV",
}

# Cue phrases indicating scripture citations
CUE_PATTERNS = [
    r"\bturn to\b",
    r"\bopen to\b",
    r"\bread from\b",
    r"\bbook of\b",
    r"\bchapter\b",
    r"\bverse\b",
    r"\bverses\b",
    r"अध्याय",
    r"पद",
    r"आयत",
    r"पढ़िए",
    r"पढ़ें",
    r"निकालिए",
    r"वचन",
    r"\badhyay\b",
    r"\bke\s+adhyay\b",
    r"\bpad\b",
    r"\baayat\b",
    r"\bmein\b",
]


@dataclass
class BookAlias:
    book_id: str
    canonical_name: str
    pattern: str
    is_multi_word: bool


class ReferenceParser:
    """Fast regex and phonetic/alias rule-based Bible citation parser."""

    def __init__(
        self,
        books_path: Optional[Path] = None,
        verse_counts_path: Optional[Path] = None,
        hindi_numbers_path: Optional[Path] = None,
    ) -> None:
        base_data = Path(__file__).resolve().parent.parent.parent / "data"
        if books_path is None:
            books_path = base_data / "books.json"

        self.normalizer = NumberNormalizer(hindi_numbers_path)
        self.validator = CanonValidator(verse_counts_path)
        self.aliases: List[BookAlias] = []
        self._fuzzy_alias_dict: Dict[str, BookAlias] = {}
        self._fuzzy_alias_keys: List[str] = []
        self._load_books(books_path)

    def _load_books(self, path: Path) -> None:
        """Load books dictionary and build sorted regex alias matchers."""
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            books_data = json.load(f)

        raw_aliases: List[Tuple[str, str, str]] = []  # (alias, book_id, canonical_name)

        for b in books_data:
            book_id = b["id"]
            name = b["name"]

            # Canonical English name
            raw_aliases.append((name.lower(), book_id, name))

            # Numbered book ordinal variations
            if name.startswith("1 "):
                base = name[2:].lower()
                raw_aliases.append((f"first {base}", book_id, name))
                raw_aliases.append((f"1st {base}", book_id, name))
                raw_aliases.append((f"pehla {base}", book_id, name))
            elif name.startswith("2 "):
                base = name[2:].lower()
                raw_aliases.append((f"second {base}", book_id, name))
                raw_aliases.append((f"2nd {base}", book_id, name))
                raw_aliases.append((f"doosra {base}", book_id, name))
            elif name.startswith("3 "):
                base = name[2:].lower()
                raw_aliases.append((f"third {base}", book_id, name))
                raw_aliases.append((f"3rd {base}", book_id, name))
                raw_aliases.append((f"teesra {base}", book_id, name))

            # Common misspellings and church variations
            if book_id == "PSA":
                for p_alias in ["zabur", "zaboor", "ज़बूर", "psalms", "psalm", "bhajan samhita", "bhajan"]:
                    raw_aliases.append((p_alias, book_id, name))
            elif book_id == "ROM":
                for r_alias in ["romans", "romiyo", "romiyon", "romi"]:
                    raw_aliases.append((r_alias, book_id, name))
            elif book_id == "JHN":
                for j_alias in ["john", "yuhanna", "yohanna", "yuhana", "yohana"]:
                    raw_aliases.append((j_alias, book_id, name))
            elif book_id == "MAT":
                for m_alias in ["matthew", "mathew", "mati", "matti"]:
                    raw_aliases.append((m_alias, book_id, name))
            elif book_id == "REV":
                for rev_alias in ["revelation", "revelations", "prakashitvakya", "prakashit vakya"]:
                    raw_aliases.append((rev_alias, book_id, name))

            # Abbreviations
            for abbr in b.get("abbreviations", []):
                raw_aliases.append((abbr.lower(), book_id, name))

            # Hindi name (Devanagari)
            hindi_name = b.get("hindi_name", "")
            if hindi_name:
                raw_aliases.append((hindi_name.lower(), book_id, name))

            # Romanized aliases
            for r in b.get("roman_aliases", []):
                raw_aliases.append((r.lower(), book_id, name))

        # Deduplicate while preserving order
        seen_aliases = set()
        deduped_raw: List[Tuple[str, str, str]] = []
        for a_text, b_id, c_name in raw_aliases:
            a_clean = a_text.strip().lower()
            if a_clean and a_clean not in seen_aliases:
                seen_aliases.add(a_clean)
                deduped_raw.append((a_clean, b_id, c_name))

        # Sort by length descending so longer phrases match first (e.g. '1 samuel' before 'samuel')
        deduped_raw.sort(key=lambda x: len(x[0]), reverse=True)

        for alias_text, book_id, name in deduped_raw:
            escaped = re.escape(alias_text)
            is_multi = " " in alias_text
            alias_obj = BookAlias(
                book_id=book_id,
                canonical_name=name,
                pattern=escaped,
                is_multi_word=is_multi,
            )
            self.aliases.append(alias_obj)
            self._fuzzy_alias_dict[alias_text] = alias_obj

        # Filter candidate keys for rapidfuzz
        self._fuzzy_alias_keys = [k for k in self._fuzzy_alias_dict.keys() if len(k) >= 3]

    def detect_version(self, text: str) -> Optional[str]:
        """Detect any Bible version keyword mentioned in text."""
        lower = text.lower()
        for kw, version_code in KNOWN_VERSIONS.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                return version_code
        return None

    def _clean_text(self, text: str) -> str:
        """Pre-clean input text for punctuation, spoken numbers, and Devanagari numerals."""
        # 1. Normalize quotes and apostrophes
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        text = re.sub(r"\b([a-zA-Z]+)'s\b", r"\1s", text)
        text = re.sub(r"\b([a-zA-Z]+)'\b", r"\1", text)

        # 2. Convert Devanagari numerals
        text = self.normalizer.convert_devanagari_digits_to_ascii(text)

        # 3. Pre-normalize spoken numbers in text (e.g. "twenty eight" -> "28")
        text = self.normalizer.normalize_spoken_numbers_in_text(text)

        # 4. Standardize chapter.verse to colon (e.g. "3.16" -> "3:16")
        text = re.sub(r"\b(\d+)[\.](\d+)\b", r"\1:\2", text)

        return text

    def parse(
        self,
        text: str,
        context_reference: Optional[BibleReference] = None,
    ) -> Optional[ParseResult]:
        """Parse raw text snippet and return detected reference with confidence score."""
        if not text or not text.strip():
            return None

        cleaned = self._clean_text(text)
        version = self.detect_version(cleaned)

        # 1. Check for relative follow-up citations ("next verse", "previous verse", "verse 17")
        follow_up_result = self._check_follow_up(cleaned, context_reference, version)
        if follow_up_result:
            return follow_up_result

        # 2. Check for explicit book references
        return self._find_book_reference(cleaned, version)

    def _check_follow_up(
        self,
        text: str,
        context: Optional[BibleReference],
        version: Optional[str],
    ) -> Optional[ParseResult]:
        """Detect relative updates: 'next verse', 'previous verse', 'verse 17'."""
        if not context:
            return None

        lower = text.lower()

        # Check 'next verse' / 'agla pad' / 'agli aayat'
        next_patterns = [
            r"\bnext verse\b",
            r"\bagla pad\b",
            r"\bagli aayat\b",
            r"अगला पद",
            r"अगली आयत",
        ]
        for p in next_patterns:
            m = re.search(p, lower)
            if m:
                target_verse = (context.verse_end or context.verse_start) + 1
                if self.validator.is_valid_reference(context.book_id, context.chapter, target_verse):
                    ref = BibleReference(
                        book_id=context.book_id,
                        book_name=context.book_name,
                        chapter=context.chapter,
                        verse_start=target_verse,
                        version=version or context.version,
                    )
                    return ParseResult(
                        reference=ref,
                        raw_match=m.group(0),
                        confidence=0.95,
                        is_follow_up=True,
                    )

        # Check 'previous verse' / 'pichhla pad'
        prev_patterns = [
            r"\bprevious verse\b",
            r"\bpichla pad\b",
            r"\bpichhla pad\b",
            r"पिछला पद",
            r"पिछली आयत",
        ]
        for p in prev_patterns:
            m = re.search(p, lower)
            if m:
                target_verse = context.verse_start - 1
                if target_verse >= 1 and self.validator.is_valid_reference(
                    context.book_id, context.chapter, target_verse
                ):
                    ref = BibleReference(
                        book_id=context.book_id,
                        book_name=context.book_name,
                        chapter=context.chapter,
                        verse_start=target_verse,
                        version=version or context.version,
                    )
                    return ParseResult(
                        reference=ref,
                        raw_match=m.group(0),
                        confidence=0.95,
                        is_follow_up=True,
                    )

        # Check single verse update: 'verse 17', 'pad 17', 'verses 5 to 9'
        verse_update_match = re.search(
            r"(?:verses?|pad|aayat|पद|आयत)\s+([a-zA-Z0-9\u0900-\u097F]+)(?:\s*(?:to|se|se lekar|तक|-)\s*([a-zA-Z0-9\u0900-\u097F]+))?",
            lower,
        )
        if verse_update_match:
            v_start_tok = verse_update_match.group(1)
            v_end_tok = verse_update_match.group(2)

            v_start = self.normalizer.parse_number_token(v_start_tok)
            v_end = self.normalizer.parse_number_token(v_end_tok) if v_end_tok else None

            if v_start and self.validator.is_valid_reference(context.book_id, context.chapter, v_start, v_end):
                ref = BibleReference(
                    book_id=context.book_id,
                    book_name=context.book_name,
                    chapter=context.chapter,
                    verse_start=v_start,
                    verse_end=v_end,
                    version=version or context.version,
                )
                return ParseResult(
                    reference=ref,
                    raw_match=verse_update_match.group(0),
                    confidence=0.90,
                    is_follow_up=True,
                )

        return None

    def _find_book_reference(
        self,
        text: str,
        version: Optional[str],
    ) -> Optional[ParseResult]:
        """Scan text for a book mention and extract chapter / verse."""
        lower = text.lower()

        for alias in self.aliases:
            # Word boundary regex: \b for latin, or boundary check for devanagari
            boundary_pattern = r"(?<!\w)" + alias.pattern + r"(?!\w)"
            match = re.search(boundary_pattern, lower)
            if not match:
                continue

            book_end_pos = match.end()
            book_start_pos = match.start()
            remainder = text[book_end_pos:].strip()

            # Attempt extraction of chapter and verse from the text right after the book
            chap_verse = self._extract_numbers(remainder)
            if chap_verse:
                chapter, verse_start, verse_end, matched_tokens_str = chap_verse

                # Validate against canon verse counts
                is_valid = self.validator.is_valid_reference(
                    alias.book_id, chapter, verse_start, verse_end
                )

                if not is_valid:
                    # Invalid reference (e.g. John 99:1)
                    return ParseResult(
                        reference=None,
                        raw_match=f"{match.group(0)} {matched_tokens_str}",
                        confidence=0.0,
                        low_confidence_reasons=("invalid_chapter_or_verse_count",),
                    )

                ref = BibleReference(
                    book_id=alias.book_id,
                    book_name=alias.canonical_name,
                    chapter=chapter,
                    verse_start=verse_start,
                    verse_end=verse_end,
                    version=version,
                )

                # Confidence calculation
                confidence, reasons = self._calculate_confidence(
                    text, book_start_pos, book_end_pos, alias, version, matched_tokens_str
                )

                return ParseResult(
                    reference=ref,
                    raw_match=text[book_start_pos : book_end_pos + len(matched_tokens_str) + 1].strip(),
                    confidence=confidence,
                    low_confidence_reasons=tuple(reasons),
                )

            # Check if book + chapter only was mentioned (e.g., 'turn to Romans 8')
            chap_only = self._extract_chapter_only(remainder)
            if chap_only:
                chapter, matched_tokens_str = chap_only
                if self.validator.is_valid_chapter(alias.book_id, chapter):
                    # Chapter-only reference: default to verse 1
                    ref = BibleReference(
                        book_id=alias.book_id,
                        book_name=alias.canonical_name,
                        chapter=chapter,
                        verse_start=1,
                        version=version,
                    )
                    confidence, reasons = self._calculate_confidence(
                        text, book_start_pos, book_end_pos, alias, version, matched_tokens_str
                    )
                    # Deduct slightly because verse was omitted
                    confidence = max(0.1, confidence - 0.1)
                    reasons.append("chapter_only_default_verse_1")
                    return ParseResult(
                        reference=ref,
                        raw_match=text[book_start_pos : book_end_pos + len(matched_tokens_str) + 1].strip(),
                        confidence=confidence,
                        low_confidence_reasons=tuple(reasons),
                    )

        # 3. Fuzzy matching fallback for near-miss book names
        fuzzy_res = self._find_fuzzy_book_reference(text, lower, version)
        if fuzzy_res:
            return fuzzy_res

        return None

    def _find_fuzzy_book_reference(
        self,
        text: str,
        lower: str,
        version: Optional[str],
    ) -> Optional[ParseResult]:
        """Scan text using fuzzy similarity against book aliases when exact match fails."""
        words = lower.split()
        if not words:
            return None

        best_alias: Optional[BookAlias] = None
        best_score: float = 0.0
        best_span: Optional[Tuple[int, int]] = None
        best_ngram_text = ""

        # Check n-grams up to 3 words
        for n in (3, 2, 1):
            if best_score >= 88.0:
                break
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                if len(ngram) < 3 or ngram in ("and", "the", "for", "from", "with", "verse", "chapter", "turn", "read"):
                    continue

                candidate = process.extractOne(ngram, self._fuzzy_alias_keys, scorer=fuzz.ratio)
                if candidate and candidate[1] >= 80.0:
                    matched_key, score = candidate[0], candidate[1]
                    ngram_pos = lower.find(ngram)
                    if ngram_pos != -1:
                        remainder = text[ngram_pos + len(ngram):].strip()
                        if self._extract_numbers(remainder) or self._extract_chapter_only(remainder):
                            if score > best_score:
                                best_score = score
                                best_alias = self._fuzzy_alias_dict[matched_key]
                                best_span = (ngram_pos, ngram_pos + len(ngram))
                                best_ngram_text = ngram

        if best_alias and best_span and best_score >= 80.0:
            book_start_pos, book_end_pos = best_span
            remainder = text[book_end_pos:].strip()
            chap_verse = self._extract_numbers(remainder)
            if chap_verse:
                chapter, verse_start, verse_end, matched_tokens_str = chap_verse
                if self.validator.is_valid_reference(best_alias.book_id, chapter, verse_start, verse_end):
                    ref = BibleReference(
                        book_id=best_alias.book_id,
                        book_name=best_alias.canonical_name,
                        chapter=chapter,
                        verse_start=verse_start,
                        verse_end=verse_end,
                        version=version,
                    )
                    confidence, reasons = self._calculate_confidence(
                        text, book_start_pos, book_end_pos, best_alias, version, matched_tokens_str
                    )
                    confidence = max(0.30, confidence - 0.20)
                    reasons.append(f"fuzzy_book_match: {best_ngram_text}->{best_alias.canonical_name} ({best_score:.0f}%)")
                    return ParseResult(
                        reference=ref,
                        raw_match=text[book_start_pos : book_end_pos + len(matched_tokens_str) + 1].strip(),
                        confidence=confidence,
                        low_confidence_reasons=tuple(reasons),
                    )

            chap_only = self._extract_chapter_only(remainder)
            if chap_only:
                chapter, matched_tokens_str = chap_only
                if self.validator.is_valid_chapter(best_alias.book_id, chapter):
                    ref = BibleReference(
                        book_id=best_alias.book_id,
                        book_name=best_alias.canonical_name,
                        chapter=chapter,
                        verse_start=1,
                        version=version,
                    )
                    confidence, reasons = self._calculate_confidence(
                        text, book_start_pos, book_end_pos, best_alias, version, matched_tokens_str
                    )
                    confidence = max(0.20, confidence - 0.30)
                    reasons.append(f"fuzzy_book_match: {best_ngram_text}->{best_alias.canonical_name} ({best_score:.0f}%)")
                    reasons.append("chapter_only_default_verse_1")
                    return ParseResult(
                        reference=ref,
                        raw_match=text[book_start_pos : book_end_pos + len(matched_tokens_str) + 1].strip(),
                        confidence=confidence,
                        low_confidence_reasons=tuple(reasons),
                    )

        return None

    def _extract_numbers(
        self,
        text: str,
    ) -> Optional[Tuple[int, int, Optional[int], str]]:
        """Extract chapter, verse_start, and optional verse_end from text."""
        # 1. Standard format: "3:16", "3.16", "3-16", "3:16-18", "3.16-18", "3: 16"
        std_match = re.match(r"^[:\s]*(\d+)[:\.\-]\s*(\d+)(?:\s*(?:[-–—]|to|se|तक)\s*(\d+))?", text)
        if std_match:
            ch = int(std_match.group(1))
            v_start = int(std_match.group(2))
            v_end = int(std_match.group(3)) if std_match.group(3) else None
            return ch, v_start, v_end, std_match.group(0)

        # 2. Space separated: "3 16", "3 16-18", "3 16 to 18"
        space_match = re.match(r"^[:\s]*(\d+)\s+(\d+)(?:\s*(?:[-–—]|to|se|तक)\s*(\d+))?", text)
        if space_match:
            ch = int(space_match.group(1))
            v_start = int(space_match.group(2))
            v_end = int(space_match.group(3)) if space_match.group(3) else None
            return ch, v_start, v_end, space_match.group(0)

        # 3. Natural language with cue words:
        # e.g.:
        # "chapter 3 verse 16"
        # "तीन अध्याय सोलह पद"
        # "teen adhyay solah pad"
        # "ke aath adhyay mein 28 pad"
        # "teen solah" (two consecutive number words)
        words = re.split(r"[\s,;]+", text)
        clean_tokens = [w.strip(".,!?;:") for w in words[:12] if w.strip(".,!?;:")]

        parsed_nums: List[Tuple[int, int]] = []  # (value, token_index)
        for idx, tok in enumerate(clean_tokens):
            num = self.normalizer.parse_number_token(tok)
            if num is not None:
                parsed_nums.append((num, idx))

        if len(parsed_nums) >= 2:
            ch = parsed_nums[0][0]
            v_start = parsed_nums[1][0]
            v_end = None

            # Check if there's a 3rd number for range (e.g. "verses 5 to 9" or "16 se 18")
            if len(parsed_nums) >= 3:
                # If third number is close to second number
                if parsed_nums[2][1] <= parsed_nums[1][1] + 3:
                    v_end = parsed_nums[2][0]

            matched_span = " ".join(clean_tokens[: parsed_nums[-1][1] + 1])
            return ch, v_start, v_end, matched_span

        return None

    def _extract_chapter_only(self, text: str) -> Optional[Tuple[int, str]]:
        """Extract chapter when no verse is mentioned (e.g., 'Romans 8', 'Romans chapter 8')."""
        match = re.match(
            r"^(?:chapter|adhyay|अध्याय|\s|ke|\:)*\s*([a-zA-Z0-9\u0900-\u097F]+)",
            text.strip(),
            re.IGNORECASE,
        )
        if match:
            tok = match.group(1)
            num = self.normalizer.parse_number_token(tok)
            if num is not None and 1 <= num <= 150:
                return num, match.group(0)
        return None

    def _calculate_confidence(
        self,
        full_text: str,
        book_start: int,
        book_end: int,
        alias: BookAlias,
        version: Optional[str],
        matched_tokens: str,
    ) -> Tuple[float, List[str]]:
        """Compute 0.0-1.0 confidence score based on cues, abbreviation risk, and version."""
        score = 0.85
        reasons: List[str] = []

        # Context cues around the match
        lower_full = full_text.lower()
        has_cue = False
        for cue in CUE_PATTERNS:
            if re.search(cue, lower_full):
                has_cue = True
                break

        if has_cue:
            score += 0.10
        else:
            reasons.append("no_preacher_cue_words")

        # Explicit version adds confidence
        if version:
            score += 0.05

        # Very short abbreviations (e.g. 'Ge', 'Jn') are riskier in speech
        if len(alias.pattern) <= 2:
            score -= 0.15
            reasons.append("short_abbreviation_ambiguity")

        score = max(0.1, min(1.0, score))
        return round(score, 2), reasons
