"""Rule-based Bible reference parser for English, Hindi (Devanagari) and Hinglish.

Pipeline: normalize -> tokenize -> find book aliases (exact, then fuzzy) -> read a
structured chapter/verse sequence after each book -> add follow-ups ("next verse",
"verse 17") relative to the last shown reference -> return the LAST valid reference
in the text (the rolling transcript buffer ends with the newest speech).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

from vachan.core.models import BibleReference, ParseResult
from vachan.core.paths import DATA_DIR
from vachan.parser.normalizer import NumberNormalizer, phonetic
from vachan.parser.validator import CanonValidator

CHAPTER_CUES = {"chapter", "chapters", "adhyay", "adhyaya", "adyay", "अध्याय"}
VERSE_CUES = {"verse", "verses", "pad", "aayat", "ayat", "vachan", "bachan", "पद", "आयत", "वचन",
              "बचन", "पैड"}  # fmt: skip
GLUED_CUE = re.compile("(.+?)(पद|वचन|बचन)$")  # Whisper: "चौथापद", "नवापद", "28 वापद"
SUFFIX_ONLY = {"वा", "वां", "वे", "वें", "वी", "वीं"}  # ordinal ending left after a digit
LINK_WORDS = {"ke", "ka", "ki", "के", "का", "की"}
# Words that describe the book: "यूहन्ना रचित सुसमाचार", "रोमियों के नाम पौलुस की पत्री"
GOSPEL_WORDS = {"rachit", "susamachar", "रचित", "सुसमाचार", "लिखित", "द्वारा", "gospel"}
EPISTLE_WORDS = {"patri", "patra", "पत्री", "पत्र", "letter", "epistle"}
DESCRIPTORS = GOSPEL_WORDS | EPISTLE_WORDS | {
    "naam", "paulus", "pustak", "kitab", "नाम", "पौलुस", "पुस्तक", "किताब",
}  # fmt: skip
# Words allowed between a book name and its numbers ("Romans ke aath adhyay mein 28 pad")
FILLERS = CHAPTER_CUES | VERSE_CUES | {
    "of", "the", "in", "number", "ke", "ka", "ki", "mein", "me", "se",
    "के", "का", "की", "में", "मे", "को", ":",
}  # fmt: skip
# "वचन" also means "word" ("परमेश्वर का वचन"): a verse-only follow-up with it needs one of these
VERSE_ANCHORS = {"ab", "now", "अब", "में", "mein", "dekhiye", "padhiye", "देखिए", "पढ़िए", "से"}
# Whisper writes a spoken number as a homophone ("verse to", "pad do"); only trusted right
# after a chapter/verse cue word, where nothing but a number makes sense
CUED_HOMOPHONES = {"to": 2, "too": 2, "tu": 2, "do": 2, "for": 4, "fore": 4, "won": 1, "ate": 8,
                   "डो": 2, "टू": 2}  # fmt: skip
CONJUNCTIONS = {"and", "aur", "और"}  # "Psalms 91 and verse 2"
RANGE_SEPS = {"-", "to", "through", "thru", "till", "until", "and", "se", "से", "और", "aur"}
# Words that make it likely a preacher is citing scripture
CUE_WORDS = CHAPTER_CUES | VERSE_CUES | {
    "turn", "open", "read", "reading", "bible", "bibles", "scripture", "book", "says",
    "written", "dekhenge", "padhenge", "padhiye", "kholiye", "nikaliye", "nikalenge",
    "बाइबल", "बाइबिल", "पढ़ें", "पढ़िए", "पढ़ेंगे", "पढते", "देखें", "देखिए", "देखेंगे",
    "निकालिए", "निकालें", "निकालेंगे", "खोलिए", "खोलें", "लिखा", "पुस्तक", "वचन",
    "आइए", "aaiye", "kholen", "kholiye", "kholte", "dekhiye", "padhen", "padhte", "dekhen",
} | DESCRIPTORS  # fmt: skip
NEXT_WORDS = {"next", "following", "agla", "agli", "agle", "अगला", "अगली", "अगले"}
PREV_WORDS = {"previous", "pichla", "pichli", "pichle"}  # Devanagari -> "previous" in normalize
# Book names that are also everyday words / names - need a cue word to be trusted
COMMON_WORD_BOOKS = {"mark", "job", "acts", "numbers", "james", "ruth", "jude", "amos",
                     "john", "song", "जॉब", "मार्क", "जॉन", "नंबर्स", "जेम्स",
                     "सांस", "raja", "राजा", "राजाओं", "itihas", "इतिहास", "kings", "किंग्स",
                     "peter", "पीटर", "पतरस", "timothy", "टिमोथी", "samuel", "सैमुअल"}  # fmt: skip
NUMBER_PREFIXES = {
    1: ["1", "first", "pehla", "pahla", "pehli", "पहला", "पहली", "पहले", "प्रथम"],
    2: ["2", "second", "doosra", "dusra", "doosri", "दूसरा", "दूसरी", "दूसरे", "द्वितीय"],
    3: ["3", "third", "teesra", "tisra", "teesri", "तीसरा", "तीसरी", "तीसरे", "तृतीय"],
}
TYPED_ONLY_EXTRAS = {"PSA": ["bhajan"]}  # spoken "bhajan 23" is usually a hymn number
SPEECH_EXTRAS = {"MAT": ["mathew", "mathi"], "REV": ["revelations"], "JHN": ["yahunna", "जौन"],
                 "PSA": ["psalm", "zabur", "zaboor"]}  # fmt: skip
# Gospel John vs the three letters ("यूहन्ना की पहली पत्री")
EPISTLE_OF = {"JHN": "1JN"}
ORDINAL_SUFFIX = re.compile(r"v[ei]?n?$")  # phonetic key of -वाँ/-वें/-वीं, -va/-ve/-van
RAW_ORDINAL = re.compile(r"(?:[vw](?:aa|a|e|ee|i|ei)n?|व[ािीे]?ं?)$")  # "now" is not "nau-va"
KNOWN_VERSIONS = {"esv": "ESV", "niv": "NIV", "kjv": "KJV", "nlt": "NLT", "nkjv": "NKJV",
                  "nasb": "NASB", "asv": "ASV", "irv": "HIN_IRV"}  # fmt: skip
FUZZY_CUTOFF = 80.0


@dataclass(frozen=True)
class _Alias:
    book_id: str
    name: str
    typed_only: bool
    common_word: bool
    bare: bool = False  # "कुरिन्थियों" without its 1/2: the number must come from a descriptor


@dataclass(frozen=True)
class _BookHit:
    start: int
    end: int
    alias: _Alias
    fuzzy: str = ""  # description when matched fuzzily
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Numbers:
    chapter: int
    verse: Optional[int]
    verse_end: Optional[int]
    end: int  # token index after the numbers
    chapter_cue: bool


class ReferenceParser:
    """Implements the ReferenceResolver port."""

    def __init__(
        self,
        books_path: Optional[Path] = None,
        verse_counts_path: Optional[Path] = None,
        hindi_numbers_path: Optional[Path] = None,
    ) -> None:
        self.normalizer = NumberNormalizer(hindi_numbers_path)
        self.validator = CanonValidator(verse_counts_path)
        self._canon = self._build_canon()
        self._aliases: dict[tuple[str, ...], _Alias] = {}
        self._names: dict[str, str] = {}
        self._load_books(books_path or DATA_DIR / "books.json")
        self._max_alias_len = max(len(k) for k in self._aliases)
        self._fuzzy_keys = {
            " ".join(k): a for k, a in self._aliases.items() if not a.typed_only and len(" ".join(k)) >= 4
        }
        # same aliases under the spelling key; a key shared by two books is dropped
        self._phon_aliases: dict[tuple[str, ...], Optional[_Alias]] = {}
        for k, a in self._aliases.items():
            pk = tuple(phonetic(t) for t in k)
            old = self._phon_aliases.get(pk, a)
            self._phon_aliases[pk] = a if old and old.book_id == a.book_id else None

    # ------------------------------------------------------------------ setup
    def _build_canon(self) -> dict[str, Optional[str]]:
        """Spelling key -> known word: snaps "adhyaay", "racchit", "पहले" onto the vocabulary."""
        words = (CHAPTER_CUES | VERSE_CUES | FILLERS | CUE_WORDS | LINK_WORDS | NEXT_WORDS
                 | PREV_WORDS | CONJUNCTIONS | {"तक", "taq", "tak"})  # fmt: skip
        self._vocab = words
        canon: dict[str, Optional[str]] = {}
        meaning: dict[str, object] = {}
        entries = [(w, w) for w in sorted(words)]
        entries += [(w, self.normalizer.hindi[w]) for w in sorted(self.normalizer.hindi)]
        for word, sense in entries:
            key = phonetic(word)
            if key not in canon:
                canon[key], meaning[key] = word, sense
            elif isinstance(sense, int) or isinstance(meaning[key], int):
                if meaning[key] != sense:  # "sat" 7 vs "saath" 60 must not merge
                    canon[key] = None
        return canon

    def _canonical(self, token: str) -> str:
        """Known word for a variant spelling; ordinal endings stripped ("solahwa" -> "solah")."""
        if (
            token.isdigit()
            or token in self._vocab
            or token in self.normalizer.hindi
            or (token.isascii() and len(token) <= 2)
        ):
            return token
        key = phonetic(token)
        hit = self._canon.get(key)
        if hit is None and key not in self._canon and RAW_ORDINAL.search(token):
            for stem in (key[:-1] if key.endswith("n") else "", ORDINAL_SUFFIX.sub("", key)):
                found = self._canon.get(stem) if stem and stem != key else None
                if found in self.normalizer.hindi:
                    return found
        return hit or token

    def _tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for tok in self.normalizer.tokenize(self.normalizer.normalize(text)):
            glued = GLUED_CUE.match(tok)
            head = self._canonical(glued.group(1)) if glued else ""
            if head in SUFFIX_ONLY or head in self.normalizer.hindi:  # split "चौथापद" only
                out += ([] if head in SUFFIX_ONLY else [head]) + [glued.group(2)]
            else:
                out.append(self._canonical(tok))
        return out

    def _key(self, alias: str) -> tuple[str, ...]:
        return tuple(self._tokens(alias))

    def _add(self, alias: str, book_id: str, name: str, typed_only: bool, bare: bool = False) -> None:
        key = self._key(alias)
        existing = self._aliases.get(key)
        if not key or all(t.isdigit() for t in key):  # "1Th" normalizes to "1"
            return
        if existing and (typed_only or not existing.typed_only):
            return
        common = " ".join(key) in COMMON_WORD_BOOKS or alias in COMMON_WORD_BOOKS
        self._aliases[key] = _Alias(book_id, name, typed_only, common, bare)

    def _load_books(self, path: Path) -> None:
        books = json.loads(path.read_text(encoding="utf-8"))
        all_prefixes = {p for ps in NUMBER_PREFIXES.values() for p in ps}
        for b in books:
            bid, name = b["id"], b["name"]
            self._names[bid] = name
            spoken = [name, b.get("hindi_name", ""), *b.get("roman_aliases", [])]
            spoken += b.get("spoken_aliases", []) + SPEECH_EXTRAS.get(bid, [])
            typed = b.get("abbreviations", []) + TYPED_ONLY_EXTRAS.get(bid, [])
            if bid[0].isdigit():  # numbered book: rebuild every prefix x base-name combo
                bases = set()
                for alias in spoken:
                    words = alias.split()
                    bases.add(" ".join(words[1:] if words and words[0] in all_prefixes else words))
                spoken = [f"{p} {base}" for p in NUMBER_PREFIXES[int(bid[0])] for base in bases]
            for alias in spoken:
                if alias and alias.lower() not in TYPED_ONLY_EXTRAS.get(bid, []):
                    self._add(alias, bid, name, typed_only=False)
            for base in bases if bid[0] == "1" else ():  # bare "कुरिन्थियों" -> 1 Cor until told
                self._add(base, bid, name, typed_only=False, bare=True)
            for alias in typed:
                self._add(alias, bid, name, typed_only=True)

    # ------------------------------------------------------------------ public
    def parse(
        self,
        text: str,
        context_reference: Optional[BibleReference] = None,
        typed: bool = False,
    ) -> Optional[ParseResult]:
        """Return the last reference in `text`, an invalid-reference result, or None.

        `typed=True` (manual search box) also accepts abbreviations like "Jn 3:16"
        and chapter-only references without cue words.
        """
        if not text or not text.strip():
            return None
        tokens = self._tokens(text)
        candidates: list[tuple[int, ParseResult]] = []
        consumed: set[int] = set()

        hits = self._find_books(tokens, typed)
        free = 0  # numbers before a book may start here
        for k, hit in enumerate(hits):
            limit = hits[k + 1].start if k + 1 < len(hits) else len(tokens)
            hit = self._describe(tokens, hit, limit)
            nums = self._scan_numbers(tokens, hit.end, limit)
            if nums is None:  # "दूसरा अध्याय पहला वचन, यूहन्ना" / "chapter 3 verse 16 of John"
                before = self._numbers_before(tokens, free, hit.start)
                if before is None:
                    continue
                start, nums = before
                hit, nums = replace(hit, start=start), replace(nums, end=hit.end)
            result = self._book_result(tokens, hit, nums, typed)
            if result is not None:
                candidates.append((hit.start, result))
                consumed.update(range(hit.start, nums.end))
                free = nums.end

        if context_reference is not None:
            candidates += self._follow_ups(tokens, consumed, context_reference)

        valid = [c for c in candidates if c[1].is_valid]
        if valid:
            return max(valid, key=lambda c: c[0])[1]
        return candidates[-1][1] if candidates else None

    # ------------------------------------------------------------------ books
    def _find_books(self, tokens: list[str], typed: bool) -> list[_BookHit]:
        exact = self._scan(tokens, 0, len(tokens), lambda t, i: self._exact_at(t, i, typed))
        # fuzzy matching only inside the gaps between exact hits
        hits: list[_BookHit] = []
        gap_start = 0
        for nxt in exact + [None]:
            gap_end = nxt.start if nxt else len(tokens)
            hits += self._scan(tokens, gap_start, gap_end, lambda t, i: self._fuzzy_at(t, i, typed))
            if nxt:
                hits.append(nxt)
                gap_start = nxt.end
        return hits

    @staticmethod
    def _scan(tokens: list[str], start: int, end: int, match) -> list[_BookHit]:
        hits, i, window = [], start, tokens[:end]
        while i < end:
            hit = match(window, i)
            if hit:
                hits.append(hit)
            i = hit.end if hit else i + 1
        return hits

    def _exact_at(self, tokens: list[str], i: int, typed: bool) -> Optional[_BookHit]:
        for n in range(min(self._max_alias_len, len(tokens) - i), 0, -1):
            words = tuple(tokens[i : i + n])
            alias = self._aliases.get(words)
            if alias and (typed or not alias.typed_only):
                return _BookHit(i, i + n, alias)
            alias = self._phon_aliases.get(tuple(map(phonetic, words)))
            if alias and not alias.typed_only:  # same sound, other spelling: "romens", "mathi"
                return _BookHit(i, i + n, alias, f"{' '.join(words)}->{alias.name} (spelling)")
        return None

    def _fuzzy_at(self, tokens: list[str], i: int, typed: bool) -> Optional[_BookHit]:
        for n in (2, 1):
            words = tokens[i : i + n]
            if len(words) < n or any(w.isdigit() or w in FILLERS or w in CUE_WORDS for w in words):
                continue
            phrase = " ".join(words)
            if len(phrase) < 4:
                continue
            best = process.extractOne(
                phrase, self._fuzzy_keys.keys(), scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF
            )
            if not best:
                continue
            nums = self._scan_numbers(tokens, self._skip_descriptors(tokens, i + n), len(tokens))
            # a guessed book name must be followed by a full chapter AND verse
            if nums and (nums.verse is not None or typed):
                alias = self._fuzzy_keys[best[0]]
                return _BookHit(i, i + n, alias, f"{phrase}->{alias.name} ({best[1]:.0f}%)")
        return None

    # ------------------------------------------------------------------ descriptors
    def _ordinal_book_number(self, tokens: list[str], i: int, limit: int) -> Optional[int]:
        """1-3 in "की पहली पत्री" (an ordinal directly before a letter word)."""
        if i + 1 < limit and tokens[i + 1] in EPISTLE_WORDS:
            parsed = self.normalizer.parse_number(tokens, i)
            if parsed and parsed[1] == i + 1 and 1 <= parsed[0] <= 3:
                return parsed[0]
        return None

    def _skip_descriptors(self, tokens: list[str], i: int, limit: Optional[int] = None) -> int:
        limit = len(tokens) if limit is None else limit
        while i < limit:
            nxt = tokens[i + 1] if i + 1 < limit else ""
            if tokens[i] in DESCRIPTORS or self._ordinal_book_number(tokens, i, limit):
                i += 1
            elif tokens[i] in LINK_WORDS and (
                nxt in DESCRIPTORS or self._ordinal_book_number(tokens, i + 1, limit)
            ):
                i += 1
            else:
                break
        return i

    def _describe(self, tokens: list[str], hit: _BookHit, limit: int) -> _BookHit:
        """Extend a book hit over "रचित सुसमाचार" / "की पहली पत्री" and pick the right book."""
        end = self._skip_descriptors(tokens, hit.end, limit)
        span = tokens[hit.end : end]
        number = next(
            (n for j in range(hit.end, end) if (n := self._ordinal_book_number(tokens, j, end))), None
        )
        book = hit.alias.book_id
        letter = any(t in EPISTLE_WORDS for t in span)
        family = EPISTLE_OF.get(book, book) if letter or hit.alias.bare else book
        reasons = hit.reasons
        if family[0] == "1" and number:
            book = f"{number}{family[1:]}"
        elif family != book or hit.alias.bare:
            book = family
            reasons += ("book_number_not_spoken",)  # "यूहन्ना की पत्री": 1 John assumed
        if end == hit.end and book == hit.alias.book_id:
            return hit
        alias = replace(hit.alias, book_id=book, name=self._names[book], bare=False, common_word=False)
        return replace(hit, end=end, alias=alias, reasons=reasons)

    def _numbers_before(self, tokens: list[str], lo: int, hi: int) -> Optional[tuple[int, _Numbers]]:
        """A full 'N अध्याय M पद' said just before the book name (only fillers in between)."""
        for s in range(max(lo, hi - 12), hi):
            if tokens[s] not in CHAPTER_CUES and self._number(tokens, s, hi) is None:
                continue
            nums = self._scan_numbers(tokens, s, hi)
            if (
                nums
                and nums.verse is not None
                and nums.chapter_cue
                and hi - nums.end <= 2
                and all(t in FILLERS for t in tokens[nums.end : hi])
            ):
                return s, nums
        return None

    # ------------------------------------------------------------------ numbers
    def _skip_fillers(self, tokens: list[str], i: int, limit: int) -> tuple[int, bool, bool]:
        chapter_cue = verse_cue = False
        for _ in range(4):
            if i + 1 < limit and tokens[i] in CONJUNCTIONS and (
                tokens[i + 1] in FILLERS or tokens[i - 1] in CHAPTER_CUES  # "91 अध्याय और 2 पद"
            ):
                i += 1
            if i < limit and tokens[i] in FILLERS:
                chapter_cue |= tokens[i] in CHAPTER_CUES
                verse_cue |= tokens[i] in VERSE_CUES
                i += 1
        return i, chapter_cue, verse_cue

    def _number(
        self, tokens: list[str], i: int, limit: int, cued: bool = False
    ) -> Optional[tuple[int, int]]:
        parsed = self.normalizer.parse_number(tokens[:limit], i)
        if parsed is None and cued and i < limit and tokens[i] in CUED_HOMOPHONES:
            return CUED_HOMOPHONES[tokens[i]], i + 1
        return parsed if parsed and parsed[1] <= limit else None

    def _scan_numbers(self, tokens: list[str], i: int, limit: int) -> Optional[_Numbers]:
        """Read 'chapter [verse [- end]]' with only filler words in between."""
        i, chapter_cue, _ = self._skip_fillers(tokens, i, limit)
        first = self._number(tokens, i, limit)
        if not first:
            return None
        chapter, i = first
        verse_cue = False
        if i < limit and tokens[i] in (":", "-"):
            j = i + 1
        else:
            j, cue, verse_cue = self._skip_fillers(tokens, i, limit)
            chapter_cue |= cue
        second = self._number(tokens, j, limit, cued=verse_cue)
        if not second:
            return _Numbers(chapter, None, None, i, chapter_cue)
        verse, i = second
        verse_end = None
        if i < limit and tokens[i] in RANGE_SEPS:
            third = self._number(tokens, i + 1, limit)
            if third and third[0] > verse:
                verse_end, i = third
        while i < limit and tokens[i] in VERSE_CUES | {"तक"}:
            i += 1
        return _Numbers(chapter, verse, verse_end, i, chapter_cue)

    # ------------------------------------------------------------------ results
    def _book_result(
        self, tokens: list[str], hit: _BookHit, nums: _Numbers, typed: bool
    ) -> Optional[ParseResult]:
        book = hit.alias
        reasons: list[str] = list(hit.reasons)
        chapter, verse, verse_end = nums.chapter, nums.verse, nums.verse_end
        raw = " ".join(tokens[hit.start : nums.end])

        if verse is None:
            splits = self._split_run_together(book.book_id, chapter)
            if splits:
                chapter, verse = splits[0]
                others = "".join(f" (or {c}:{v})" for c, v in splits[1:])  # "912": 9:12 or 91:2
                reasons.append(f"number_split: {nums.chapter} -> {chapter}:{verse}{others}")
        has_cue = self._has_cue(tokens, hit.start, nums.end)

        if verse is None:
            cued = typed or has_cue or nums.chapter_cue
            if not cued and (book.common_word or not tokens[nums.end - 1].isdigit()):
                return None  # "John 3" / "लूका एक वैद्य था" in passing is not worth a suggestion
            if not self.validator.is_valid_chapter(book.book_id, chapter):
                return self._invalid(raw)
            reasons.append("chapter_only_no_verse")  # never default to verse 1 here
        elif not self.validator.is_valid_reference(book.book_id, chapter, verse, verse_end):
            return self._invalid(raw)

        score = 0.55
        if has_cue:
            score += 0.25
        elif not typed:
            reasons.append("no_preacher_cue_words")
            if book.common_word:
                score -= 0.10
                reasons.append("book_name_is_common_word")
        if verse is not None:
            score += 0.20
        if hit.fuzzy:
            score -= 0.25
            reasons.append(f"fuzzy_book_match: {hit.fuzzy}")
        if any(r.startswith("number_split") for r in reasons):
            score -= 0.10
        if "book_number_not_spoken" in reasons:
            score -= 0.15

        ref = BibleReference(
            book_id=book.book_id,
            book_name=book.name,
            chapter=chapter,
            verse_start=verse,
            verse_end=verse_end,
            version=self._version_near(tokens, hit.start, nums.end),
        )
        return ParseResult(ref, raw, round(max(0.05, min(1.0, score)), 2), False, tuple(reasons))

    def _split_run_together(self, book_id: str, number: int) -> list[tuple[int, int]]:
        """Whisper often writes 'John 316' for 'John 3 16'; every valid split, shortest chapter first."""
        digits = str(number)
        if len(digits) < 3 or self.validator.is_valid_chapter(book_id, number):
            return []
        splits = [(int(digits[:k]), int(digits[k:])) for k in range(1, len(digits)) if digits[k] != "0"]
        return [(c, v) for c, v in splits if self.validator.is_valid_reference(book_id, c, v)]

    def _has_cue(self, tokens: list[str], start: int, end: int) -> bool:
        return any(t in CUE_WORDS for t in tokens[max(0, start - 6) : end])

    def _version_near(self, tokens: list[str], start: int, end: int) -> Optional[str]:
        window = tokens[max(0, start - 3) : end + 8]
        for a, b in zip(window, window[1:] + [""]):
            if (a, b) == ("king", "james"):
                return "KJV"
            if a in KNOWN_VERSIONS:
                return KNOWN_VERSIONS[a]
        return None

    @staticmethod
    def _invalid(raw: str) -> ParseResult:
        return ParseResult(None, raw, 0.0, False, ("invalid_chapter_or_verse_count",))

    # ------------------------------------------------------------------ follow-ups
    def _follow_ups(
        self, tokens: list[str], consumed: set[int], ctx: BibleReference
    ) -> list[tuple[int, ParseResult]]:
        out: list[tuple[int, ParseResult]] = []
        relative = (f"relative_to: {ctx.format_reference()}",)
        for i, tok in enumerate(tokens):
            if i in consumed:
                continue
            pair = (tok, tokens[i + 1] if i + 1 < len(tokens) else "")
            step = tok in NEXT_WORDS or tok in PREV_WORDS
            if step and pair[1] in VERSE_CUES and ctx.verse_start is not None:
                nxt = tok in NEXT_WORDS
                verse = (ctx.verse_end or ctx.verse_start) + 1 if nxt else ctx.verse_start - 1
                ref = self._ctx_ref(ctx, ctx.chapter, verse, None)
                if ref:
                    out.append((i, ParseResult(ref, " ".join(pair), 0.9, True)))
            elif tok in VERSE_CUES and (before := self._verse_before(tokens, i, consumed)):
                start, verse, end = before  # "अब सातवें वचन में", "पाँचवें और छठे पद"
                ref = self._ctx_ref(ctx, ctx.chapter, verse, end)
                if ref:
                    raw = " ".join(tokens[start : i + 1])
                    why = relative + ("verse_only_follow_up",)
                    out.append((start, ParseResult(ref, raw, 0.7, True, why)))
            elif tok in VERSE_CUES:
                nums = self._number(tokens, i + 1, len(tokens), cued=True)
                if nums and nums[1] - 1 not in consumed:
                    end = None
                    if nums[1] < len(tokens) and tokens[nums[1]] in RANGE_SEPS:
                        third = self._number(tokens, nums[1] + 1, len(tokens))
                        end = third[0] if third and third[0] > nums[0] else None
                    ref = self._ctx_ref(ctx, ctx.chapter, nums[0], end)
                    if ref:
                        raw = " ".join(tokens[i : nums[1]])
                        out.append((i, ParseResult(ref, raw, 0.8, True, relative)))
            elif tok in CHAPTER_CUES:
                nums = self._scan_numbers(tokens, i, len(tokens))
                if nums and not consumed.intersection(range(i, nums.end)):
                    ref = self._ctx_ref(ctx, nums.chapter, nums.verse, nums.verse_end)
                    if ref:
                        raw = " ".join(tokens[i : nums.end])
                        why = relative + (() if nums.verse else ("chapter_only_no_verse",))
                        out.append((i, ParseResult(ref, raw, 0.7, True, why)))
                        consumed.update(range(i, nums.end))
        return out

    def _verse_before(
        self, tokens: list[str], cue: int, consumed: set[int]
    ) -> Optional[tuple[int, int, Optional[int]]]:
        """'N [और M] पद' ending at the cue word: (start, verse, verse_end)."""
        if tokens[cue] in ("वचन", "vachan") and not (
            (cue >= 1 and tokens[cue - 1] in VERSE_ANCHORS)
            or any(t in VERSE_ANCHORS for t in tokens[max(0, cue - 4) : cue - 1] + tokens[cue + 1 : cue + 2])
        ):
            return None  # "पहला वचन परमेश्वर ने दिया"
        for s in range(max(0, cue - 6), cue):
            if consumed.intersection(range(s, cue)):
                continue
            first = self._number(tokens, s, cue)
            if not first:
                continue
            if first[1] == cue:
                return s, first[0], None
            if tokens[first[1]] in RANGE_SEPS:
                second = self._number(tokens, first[1] + 1, cue)
                if second and second[1] == cue and second[0] > first[0]:
                    return s, first[0], second[0]
        return None

    def _ctx_ref(
        self, ctx: BibleReference, chapter: int, verse: Optional[int], verse_end: Optional[int]
    ) -> Optional[BibleReference]:
        if verse is None:
            valid = self.validator.is_valid_chapter(ctx.book_id, chapter)
        else:
            valid = self.validator.is_valid_reference(ctx.book_id, chapter, verse, verse_end)
        if not valid:
            return None
        return BibleReference(ctx.book_id, ctx.book_name, chapter, verse, verse_end, ctx.version)
