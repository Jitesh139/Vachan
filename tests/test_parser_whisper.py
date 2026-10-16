"""Parser cases built from real faster-whisper output (see docs/AUDIT.md).

Strings marked `# real` were produced verbatim by faster-whisper (small / large-v3-turbo)
on synthesized sermon phrases; the rest follow the same observed patterns: commas
between Hindi words, English names in Devanagari, misspelled अध्याय, "13-4", "316".
"""

import pytest

from vachan.core.models import BibleReference
from vachan.parser import ReferenceParser

JOHN_3_16 = BibleReference("JHN", "John", 3, 16)


@pytest.fixture(scope="module")
def parser() -> ReferenceParser:
    return ReferenceParser()


def fmt(res) -> str | None:
    return res.reference.format_reference() if res and res.is_valid else None


POSITIVE = [
    # English
    ("Turn to John chapter 3 verse 16.", "John 3:16"),  # real
    ("Open your Bibles to 1 Corinthians 13-4.", "1 Corinthians 13:4"),  # real
    ("Open your Bibles to 1st Corinthians 13-4.", "1 Corinthians 13:4"),  # real
    ("Open your Bibles to 1 Corinthians 13 4.", "1 Corinthians 13:4"),  # real
    ("Let us read Psalm 23 verse 1.", "Psalms 23:1"),  # real
    ("Turn with me to Romans chapter 8, verse 28.", "Romans 8:28"),
    ("Romans 8.28 says that all things work together", "Romans 8:28"),
    ("John 316 is the most famous verse", "John 3:16"),
    ("Let's go to Romans 828.", "Romans 8:28"),
    ("Ephesians chapter 2 verses 8 and 9", "Ephesians 2:8-9"),
    ("John 3:16-18", "John 3:16-18"),
    ("Turn to John 3, 16 through 18.", "John 3:16-18"),
    ("Psalm 119, verse 105.", "Psalms 119:105"),
    ("Psalm one hundred and nineteen verse one hundred and five", "Psalms 119:105"),
    ("Second Timothy chapter one verse seven", "2 Timothy 1:7"),
    ("Turn to Genesis chapter one verse one.", "Genesis 1:1"),
    ("Hebrews 11 verse 1 says, now faith is", "Hebrews 11:1"),
    ("In Philippians 4:13 Paul says", "Philippians 4:13"),
    ("Jeremiah 29 11", "Jeremiah 29:11"),
    ("Proverbs chapter 3, verses 5 and 6.", "Proverbs 3:5-6"),
    ("First John 1 9", "1 John 1:9"),
    ("Isaiah 40 verse 31", "Isaiah 40:31"),
    ("Matthew 28, 19 and 20", "Matthew 28:19-20"),
    ("Read from the book of Revelation chapter 21 verse 4.", "Revelation 21:4"),
    ("Turn to Romans 8.", "Romans 8"),
    ("Turn to John chapter 3. Turn to John chapter 3 verse 16.", "John 3:16"),
    ("John 3:16 says God loved the world. Now Romans 5 8.", "Romans 5:8"),
    ("Galatians 5, 22 and 23", "Galatians 5:22-23"),
    ("James chapter 1 verse 5", "James 1:5"),
    ("Turn to Matthew chapter twenty-eight verse nineteen", "Matthew 28:19"),
    # Hindi (Devanagari)
    ("यूहन्ना, तीन, अध्याय, सोलह, पद.", "John 3:16"),  # real
    ("यूहन्ना 3 अद्याय 16", "John 3:16"),  # real
    ("रूमेंस के आठ अध्याय में 28 पद देखेंगे.", "Romans 8:28"),  # real
    ("रोमेंस के आठ अध्याय में 28 पद देखेंगे.", "Romans 8:28"),  # real
    ("रूमेन्स के 8 अद्याय में 28 पद देखेंगे", "Romans 8:28"),  # real
    ("रोमेंस के 8 अद्याई में 28 पद देखेंगे", "Romans 8:28"),  # real
    ("रोमियो आठ अध्याय, अट्ठाईस पद.", "Romans 8:28"),
    ("यूहन्ना के तीसरे अध्याय का सोलहवां पद", "John 3:16"),
    ("भजन संहिता 23, 1", "Psalms 23:1"),
    ("भजन संहिता तेईस अध्याय एक पद", "Psalms 23:1"),
    ("मत्ती 5:7", "Matthew 5:7"),
    ("आइए हम यूहन्ना ३:१६ पढ़ें।", "John 3:16"),
    ("प्रेरितों के काम 2 अध्याय 38 पद", "Acts 2:38"),
    ("1 कुरिन्थियों 13 अध्याय 4 पद", "1 Corinthians 13:4"),
    ("पहला कुरिन्थियों तेरह अध्याय चार पद", "1 Corinthians 13:4"),
    ("इब्रानियों 11 अध्याय 1 पद", "Hebrews 11:1"),
    ("जॉन चैप्टर 3 वर्स 16", "John 3:16"),
    ("रोमन्स चैप्टर 12 वर्स 1 और 2", "Romans 12:1-2"),
    ("फिलिप्पियों 4 अध्याय 13 पद", "Philippians 4:13"),
    ("उत्पत्ति 1 अध्याय 1 पद", "Genesis 1:1"),
    ("प्रकाशितवाक्य 21 अध्याय 4 पद", "Revelation 21:4"),
    ("यशायाह 53 अध्याय 5 पद", "Isaiah 53:5"),
    ("यूहन्ना 3 अध्याय 16 से 18 पद तक", "John 3:16-18"),
    ("यूहन्ना 3 अध्याय 16 पद", "John 3:16"),
    # Hinglish (romanized)
    ("Romans ke aath adhyay mein 28 pad dekhenge", "Romans 8:28"),
    ("Yuhanna teen adhyay solah pad", "John 3:16"),
    ("Matti paanch adhyay saat pad", "Matthew 5:7"),
    ("Bhajan Samhita 23 adhyay 1 pad", "Psalms 23:1"),
]


@pytest.mark.parametrize("text,expected", POSITIVE)
def test_detects_reference(parser, text, expected):
    assert fmt(parser.parse(text)) == expected


FOLLOW_UPS = [
    ("Now look at the next verse.", "John 3:17"),  # real
    ("अगला पद देखिए", "John 3:17"),
    ("नेक्स्ट वर्स", "John 3:17"),
    ("previous verse", "John 3:15"),
    ("पिछला पद", "John 3:15"),
    ("Now read verse 18", "John 3:18"),
    ("अब पद 18 देखिए", "John 3:18"),
    ("Now chapter 4 verse 1", "John 4:1"),
    ("Turn to Romans 8 28", "Romans 8:28"),  # explicit reference beats relative
]


@pytest.mark.parametrize("text,expected", FOLLOW_UPS)
def test_follow_ups(parser, text, expected):
    assert fmt(parser.parse(text, context_reference=JOHN_3_16)) == expected


NEGATIVE = [
    "He gave me one of those looks. Mark my words, in 2 weeks",  # was Hebrews 1:2
    "he will be there in 3 days, so do it",
    "I sat down with John yesterday and we talked for 2 hours",
    "We sang bhajan 23 this morning",
    "Thank you.",
    "यूहन्ना, रोमियों, भजन संहिता, अध्याय, पद, chapter, verse.",  # prompt echo
    "My job 3 years ago was hard",
    "Mark 2 people as present",
    "Now look at the next verse.",  # no context -> nothing to follow
    "Good morning church, welcome. We have 25 visitors at 10:30.",
]


@pytest.mark.parametrize("text", NEGATIVE)
def test_no_false_positive(parser, text):
    assert fmt(parser.parse(text)) is None


@pytest.mark.parametrize("text", ["John 99:1", "Jude 2:5", "यूहन्ना 30 अध्याय 1 पद"])
def test_invalid_reference_reported(parser, text):
    res = parser.parse(text)
    assert res is not None and not res.is_valid
    assert "invalid_chapter_or_verse_count" in res.low_confidence_reasons


def test_strong_cue_is_high_confidence(parser):
    res = parser.parse("Turn to Romans 8 28")
    assert res.confidence >= 0.9 and not res.low_confidence_reasons


def test_no_cue_common_word_is_flagged(parser):
    res = parser.parse("Acts 2 38")
    assert fmt(res) == "Acts 2:38"
    assert res.confidence < 0.9 and "no_preacher_cue_words" in res.low_confidence_reasons


def test_fuzzy_match_is_flagged(parser):
    res = parser.parse("turn to romens 8 28")
    assert fmt(res) == "Romans 8:28"
    assert any(r.startswith("fuzzy_book_match") for r in res.low_confidence_reasons)


def test_chapter_only_is_flagged(parser):
    res = parser.parse("Turn to Romans 8.")
    assert "chapter_only_no_verse" in res.low_confidence_reasons
    assert res.reference.verse_start is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Jn 3:16", "John 3:16"),
        ("1 Cor 13:4", "1 Corinthians 13:4"),
        ("Ps 23", "Psalms 23"),
        ("Rom 2:1", "Romans 2:1"),
        ("Jn 2:1", "John 2:1"),
    ],  # "1Th" must not make "1" a book
)
def test_abbreviations_only_when_typed(parser, text, expected):
    assert fmt(parser.parse(text, typed=True)) == expected
    assert fmt(parser.parse(text)) is None


def test_version_detected(parser):
    assert parser.parse("Ephesians 6:10 in the ESV").reference.version == "ESV"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("यूहन्ना, तीन अध्याय, सुलह पद.", "John 3:16"),  # real (small, CPU)
        ("रोमेंस के आत अध्याय में, 28 पद देखेंगे.", "Romans 8:28"),  # real (small, CPU)
        ("Romans 8, verse 28.", "Romans 8:28"),  # real (small, CPU)
    ],
)
def test_observed_mishearings(parser, text, expected):
    assert fmt(parser.parse(text)) == expected
