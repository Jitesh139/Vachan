"""Comprehensive unit tests for the Vachan Bible reference parser.

Covers 40+ sentences across:
- Standard English citations
- English spoken number words ("chapter three verse sixteen")
- Devanagari Hindi citations ("यूहन्ना 3:16", "रोमियों 8 अध्याय 28 पद")
- Hinglish code-switching ("Romans ke aath adhyay mein", "Yuhanna teen solah")
- Spoken cues ("turn to", "open to", "padhiye")
- Relative follow-ups ("next verse", "previous verse", "verse 17")
- Version detection (ESV, NIV, KJV)
- Verse ranges ("John 3:16-18", "verses 5 to 9")
- Invalid reference rejection (e.g. John 99:1, Jude 2:1)
"""

import pytest

from vachan.core.models import BibleReference
from vachan.parser import ReferenceParser


@pytest.fixture
def parser() -> ReferenceParser:
    return ReferenceParser()


# ==========================================
# 1. Standard English & Number Words Tests
# ==========================================


def test_english_standard_colon(parser: ReferenceParser) -> None:
    res = parser.parse("Please open to John 3:16 today.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_english_spoken_words(parser: ReferenceParser) -> None:
    res = parser.parse("Turn to Matthew chapter five verse seven.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Matthew"
    assert res.reference.chapter == 5
    assert res.reference.verse_start == 7


def test_english_with_version(parser: ReferenceParser) -> None:
    res = parser.parse("Let us read from Romans 8:28 in the ESV.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 28
    assert res.reference.version == "ESV"


def test_english_numbered_book_first_corinthians(parser: ReferenceParser) -> None:
    res = parser.parse("Turn with me to 1 Corinthians 13:4.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "1 Corinthians"
    assert res.reference.chapter == 13
    assert res.reference.verse_start == 4


def test_english_numbered_book_second_timothy(parser: ReferenceParser) -> None:
    res = parser.parse("Open to 2 Timothy chapter 1 verse 7.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "2 Timothy"
    assert res.reference.chapter == 1
    assert res.reference.verse_start == 7


def test_english_verse_range(parser: ReferenceParser) -> None:
    res = parser.parse("Reading Ephesians 2:8-10.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Ephesians"
    assert res.reference.chapter == 2
    assert res.reference.verse_start == 8
    assert res.reference.verse_end == 10


def test_english_psalms(parser: ReferenceParser) -> None:
    res = parser.parse("Psalm 23:1 is our meditation.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Psalms"
    assert res.reference.chapter == 23
    assert res.reference.verse_start == 1


def test_english_genesis(parser: ReferenceParser) -> None:
    res = parser.parse("In the beginning, Genesis 1:1.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Genesis"
    assert res.reference.chapter == 1
    assert res.reference.verse_start == 1


def test_english_turn_to_chapter_only(parser: ReferenceParser) -> None:
    res = parser.parse("Let's turn to Romans 8.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start is None  # chapter only: never defaulted to 1


# ==========================================
# 2. Devanagari Hindi Tests
# ==========================================


def test_hindi_devanagari_standard(parser: ReferenceParser) -> None:
    res = parser.parse("आइए हम यूहन्ना 3:16 पढ़ें।")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_hindi_devanagari_words_with_cues(parser: ReferenceParser) -> None:
    res = parser.parse("यूहन्ना तीन अध्याय सोलह पद निकालिए।")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_hindi_devanagari_romiyon(parser: ReferenceParser) -> None:
    res = parser.parse("रोमियों 8 अध्याय 28 पद पढ़िए।")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 28


def test_hindi_devanagari_numerals(parser: ReferenceParser) -> None:
    res = parser.parse("मत्ती ५:७ पद")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Matthew"
    assert res.reference.chapter == 5
    assert res.reference.verse_start == 7


def test_hindi_bhajan_samhita(parser: ReferenceParser) -> None:
    res = parser.parse("भजन संहिता तेईस अध्याय एक पद")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Psalms"
    assert res.reference.chapter == 23
    assert res.reference.verse_start == 1


def test_hindi_preriton_ke_kaam(parser: ReferenceParser) -> None:
    res = parser.parse("प्रेरितों के काम 1 अध्याय 8 पद")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Acts"
    assert res.reference.chapter == 1
    assert res.reference.verse_start == 8


def test_hindi_utpatti(parser: ReferenceParser) -> None:
    res = parser.parse("उत्पत्ति 12 अध्याय 1 से 3 पद")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Genesis"
    assert res.reference.chapter == 12
    assert res.reference.verse_start == 1
    assert res.reference.verse_end == 3


def test_hindi_prakashitvakya(parser: ReferenceParser) -> None:
    res = parser.parse("प्रकाशितवाक्य 3:20")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Revelation"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 20


def test_hindi_pehla_kurinthiyon(parser: ReferenceParser) -> None:
    res = parser.parse("1 कुरिन्थियों 13 अध्याय 4 पद")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "1 Corinthians"
    assert res.reference.chapter == 13
    assert res.reference.verse_start == 4


# ==========================================
# 3. Hinglish & Code-Switching Tests
# ==========================================


def test_hinglish_romans_ke_aath_adhyay(parser: ReferenceParser) -> None:
    res = parser.parse("Romans ke aath adhyay mein 28 pad dekhenge.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 28


def test_hinglish_yuhanna_teen_solah(parser: ReferenceParser) -> None:
    res = parser.parse("Ab hum Yuhanna teen solah padhenge.")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_hinglish_pehla_samuel(parser: ReferenceParser) -> None:
    res = parser.parse("Pehla Samuel chapter 16 verse 7")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "1 Samuel"
    assert res.reference.chapter == 16
    assert res.reference.verse_start == 7


def test_hinglish_ifisiyo(parser: ReferenceParser) -> None:
    res = parser.parse("Efisi chapter 6 verse 10 se 12")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Ephesians"
    assert res.reference.chapter == 6
    assert res.reference.verse_start == 10
    assert res.reference.verse_end == 12


def test_hinglish_ginti(parser: ReferenceParser) -> None:
    res = parser.parse("Ginti 6:24-26 padhiye")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Numbers"
    assert res.reference.chapter == 6
    assert res.reference.verse_start == 24
    assert res.reference.verse_end == 26


def test_hinglish_filippiyo(parser: ReferenceParser) -> None:
    res = parser.parse("Philipiyo 4:13 mein likha hai")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Philippians"
    assert res.reference.chapter == 4
    assert res.reference.verse_start == 13


def test_hinglish_ibraniyon(parser: ReferenceParser) -> None:
    res = parser.parse("Ibrani 11:1 faith is the substance")
    assert res is not None
    assert res.is_valid
    assert res.reference.book_name == "Hebrews"
    assert res.reference.chapter == 11
    assert res.reference.verse_start == 1


# ==========================================
# 4. Version Detection Tests
# ==========================================


def test_version_kjv(parser: ReferenceParser) -> None:
    res = parser.parse("Read Psalm 91:1 in the KJV version.")
    assert res is not None
    assert res.reference.version == "KJV"


def test_version_niv(parser: ReferenceParser) -> None:
    res = parser.parse("Proverbs 3:5-6 according to NIV.")
    assert res is not None
    assert res.reference.version == "NIV"


def test_version_nlt(parser: ReferenceParser) -> None:
    res = parser.parse("Jeremiah 29:11 NLT translation")
    assert res is not None
    assert res.reference.version == "NLT"


# ==========================================
# 5. Follow-ups & Relative Reference Tests
# ==========================================


def test_follow_up_next_verse(parser: ReferenceParser) -> None:
    ctx = BibleReference(book_id="JHN", book_name="John", chapter=3, verse_start=16)
    res = parser.parse("Now let's look at the next verse.", context_reference=ctx)
    assert res is not None
    assert res.is_valid
    assert res.is_follow_up
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 17


def test_follow_up_hindi_agla_pad(parser: ReferenceParser) -> None:
    ctx = BibleReference(book_id="ROM", book_name="Romans", chapter=8, verse_start=28)
    res = parser.parse("अगला पद देखिए", context_reference=ctx)
    assert res is not None
    assert res.is_valid
    assert res.is_follow_up
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 29


def test_follow_up_previous_verse(parser: ReferenceParser) -> None:
    ctx = BibleReference(book_id="JHN", book_name="John", chapter=3, verse_start=17)
    res = parser.parse("Go back to the previous verse.", context_reference=ctx)
    assert res is not None
    assert res.is_valid
    assert res.is_follow_up
    assert res.reference.verse_start == 16


def test_follow_up_specific_verse_number(parser: ReferenceParser) -> None:
    ctx = BibleReference(book_id="ROM", book_name="Romans", chapter=8, verse_start=28)
    res = parser.parse("Now check verse 31.", context_reference=ctx)
    assert res is not None
    assert res.is_valid
    assert res.is_follow_up
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 31


def test_follow_up_verses_range(parser: ReferenceParser) -> None:
    ctx = BibleReference(book_id="PSA", book_name="Psalms", chapter=23, verse_start=1)
    res = parser.parse("Read verses 4 to 6.", context_reference=ctx)
    assert res is not None
    assert res.is_valid
    assert res.is_follow_up
    assert res.reference.chapter == 23
    assert res.reference.verse_start == 4
    assert res.reference.verse_end == 6


# ==========================================
# 6. Invalid & Out-of-Bounds Rejection Tests
# ==========================================


def test_reject_john_99_1(parser: ReferenceParser) -> None:
    res = parser.parse("Please turn to John 99:1.")
    assert res is not None
    # Confidence is 0.0, reference is None
    assert not res.is_valid
    assert "invalid_chapter_or_verse_count" in res.low_confidence_reasons


def test_reject_jude_2_1(parser: ReferenceParser) -> None:
    # Jude only has 1 chapter
    res = parser.parse("Look at Jude 2:5.")
    assert res is not None
    assert not res.is_valid
    assert "invalid_chapter_or_verse_count" in res.low_confidence_reasons


def test_reject_genesis_verse_out_of_bounds(parser: ReferenceParser) -> None:
    # Genesis 1 only has 31 verses
    res = parser.parse("Genesis 1:99")
    assert res is not None
    assert not res.is_valid
    assert "invalid_chapter_or_verse_count" in res.low_confidence_reasons


def test_ignore_casual_text_without_scripture(parser: ReferenceParser) -> None:
    res = parser.parse("Good morning brothers and sisters, welcome to church today.")
    assert res is None


def test_ignore_random_numbers(parser: ReferenceParser) -> None:
    res = parser.parse("We had 25 people attend at 10:30 am.")
    assert res is None


# ==========================================
# 7. Additional Canonical Book Coverage Tests
# ==========================================


def test_joshua_english(parser: ReferenceParser) -> None:
    res = parser.parse("Joshua 1:9 be strong and courageous")
    assert res is not None
    assert res.reference.book_name == "Joshua"
    assert res.reference.chapter == 1
    assert res.reference.verse_start == 9


def test_isaiah_hindi_yashayah(parser: ReferenceParser) -> None:
    res = parser.parse("यशायाह 40:31")
    assert res is not None
    assert res.reference.book_name == "Isaiah"
    assert res.reference.chapter == 40
    assert res.reference.verse_start == 31


def test_daniel_daniyyel(parser: ReferenceParser) -> None:
    res = parser.parse("दानिय्येल 6:10")
    assert res is not None
    assert res.reference.book_name == "Daniel"
    assert res.reference.chapter == 6
    assert res.reference.verse_start == 10


def test_habakkuk_spoken(parser: ReferenceParser) -> None:
    res = parser.parse("Habakkuk chapter two verse four")
    assert res is not None
    assert res.reference.book_name == "Habakkuk"
    assert res.reference.chapter == 2
    assert res.reference.verse_start == 4


def test_malachi_spoken(parser: ReferenceParser) -> None:
    res = parser.parse("Malachi 3:10 bring the whole tithe")
    assert res is not None
    assert res.reference.book_name == "Malachi"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 10


# =========================================================================
# 8. Real Whisper-Style Output Tests (Lowercase, Punctuation, Digits, Split)
# =========================================================================


def test_whisper_lowercase_dot_separator(parser: ReferenceParser) -> None:
    res = parser.parse("john 3.16")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_space_separated_digits(parser: ReferenceParser) -> None:
    res = parser.parse("john 3 16")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_hyphen_separated_digits(parser: ReferenceParser) -> None:
    res = parser.parse("john 3-16")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_spaced_colon(parser: ReferenceParser) -> None:
    res = parser.parse("john 3 : 16")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_three_sixteen(parser: ReferenceParser) -> None:
    res = parser.parse("john three sixteen")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_devanagari_numerals_colon(parser: ReferenceParser) -> None:
    res = parser.parse("यूहन्ना ३:१६")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_devanagari_numerals_dot(parser: ReferenceParser) -> None:
    res = parser.parse("यूहन्ना ३.१६")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_devanagari_numerals_spaced(parser: ReferenceParser) -> None:
    res = parser.parse("यूहन्ना ३ १६")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_yohanna_alias(parser: ReferenceParser) -> None:
    res = parser.parse("yohanna 3 16")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_yuhanna_alias_lowercase(parser: ReferenceParser) -> None:
    res = parser.parse("yuhanna 3:16")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_yuhanna_spoken_words(parser: ReferenceParser) -> None:
    res = parser.parse("yuhanna teen solah")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_john_possessive_apostrophe(parser: ReferenceParser) -> None:
    res = parser.parse("john's 3:16")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_romans_apostrophe(parser: ReferenceParser) -> None:
    res = parser.parse("roman's 8:28")
    assert res is not None
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 28


def test_whisper_romans_space_digits(parser: ReferenceParser) -> None:
    res = parser.parse("romans 8 28")
    assert res is not None
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 28


def test_whisper_romans_spoken_twenty_eight(parser: ReferenceParser) -> None:
    res = parser.parse("turn to romans chapter eight verse twenty eight")
    assert res is not None
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 28


def test_whisper_romiyo_hindi_romanized(parser: ReferenceParser) -> None:
    res = parser.parse("romiyo 8 28")
    assert res is not None
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 28


def test_whisper_romiyo_spoken_hindi_numbers(parser: ReferenceParser) -> None:
    res = parser.parse("romiyo aath adhyay atthais pad")
    assert res is not None
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 28


def test_whisper_psalm_single_chapter(parser: ReferenceParser) -> None:
    res = parser.parse("psalm 23")
    assert res is not None
    assert res.reference.book_name == "Psalms"
    assert res.reference.chapter == 23
    assert res.reference.verse_start is None  # chapter only: never defaulted to 1


def test_whisper_psalms_plural(parser: ReferenceParser) -> None:
    res = parser.parse("psalms 23 1")
    assert res is not None
    assert res.reference.book_name == "Psalms"
    assert res.reference.chapter == 23
    assert res.reference.verse_start == 1


def test_whisper_zabur_alias(parser: ReferenceParser) -> None:
    res = parser.parse("zabur 23")
    assert res is not None
    assert res.reference.book_name == "Psalms"
    assert res.reference.chapter == 23
    assert res.reference.verse_start is None  # chapter only: never defaulted to 1


def test_whisper_zaboor_alternate(parser: ReferenceParser) -> None:
    res = parser.parse("zaboor 23 1")
    assert res is not None
    assert res.reference.book_name == "Psalms"
    assert res.reference.chapter == 23
    assert res.reference.verse_start == 1


def test_whisper_bhajan_samhita_devanagari(parser: ReferenceParser) -> None:
    res = parser.parse("भजन संहिता २३")
    assert res is not None
    assert res.reference.book_name == "Psalms"
    assert res.reference.chapter == 23
    assert res.reference.verse_start is None  # chapter only: never defaulted to 1


def test_whisper_first_corinthians_words(parser: ReferenceParser) -> None:
    res = parser.parse("first corinthians 13 4")
    assert res is not None
    assert res.reference.book_name == "1 Corinthians"
    assert res.reference.chapter == 13
    assert res.reference.verse_start == 4


def test_whisper_first_corinthians_spoken_numbers(parser: ReferenceParser) -> None:
    res = parser.parse("first corinthians chapter thirteen verse four")
    assert res is not None
    assert res.reference.book_name == "1 Corinthians"
    assert res.reference.chapter == 13
    assert res.reference.verse_start == 4


def test_whisper_second_timothy_ordinal(parser: ReferenceParser) -> None:
    res = parser.parse("second timothy 1 7")
    assert res is not None
    assert res.reference.book_name == "2 Timothy"
    assert res.reference.chapter == 1
    assert res.reference.verse_start == 7


def test_whisper_matthew_misspelling_mathew(parser: ReferenceParser) -> None:
    res = parser.parse("mathew 5 7")
    assert res is not None
    assert res.reference.book_name == "Matthew"
    assert res.reference.chapter == 5
    assert res.reference.verse_start == 7


def test_whisper_fuzzy_johan(parser: ReferenceParser) -> None:
    res = parser.parse("johan 3:16")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_fuzzy_romens(parser: ReferenceParser) -> None:
    res = parser.parse("romens 8:28")
    assert res is not None
    assert res.reference.book_name == "Romans"
    assert res.reference.chapter == 8
    assert res.reference.verse_start == 28


def test_whisper_fuzzy_philipians(parser: ReferenceParser) -> None:
    res = parser.parse("philipians 4 13")
    assert res is not None
    assert res.reference.book_name == "Philippians"
    assert res.reference.chapter == 4
    assert res.reference.verse_start == 13


def test_whisper_revelations_plural(parser: ReferenceParser) -> None:
    res = parser.parse("revelations 21 4")
    assert res is not None
    assert res.reference.book_name == "Revelation"
    assert res.reference.chapter == 21
    assert res.reference.verse_start == 4


def test_whisper_split_conversational_filler(parser: ReferenceParser) -> None:
    res = parser.parse("and now church please turn with me to john chapter 3 verse 16 for today's sermon")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16


def test_whisper_hindi_conversational_filler(parser: ReferenceParser) -> None:
    res = parser.parse("आइए हम सब मिलकर निकालेंगे यूहन्ना 3 अध्याय 16 पद")
    assert res is not None
    assert res.reference.book_name == "John"
    assert res.reference.chapter == 3
    assert res.reference.verse_start == 16
