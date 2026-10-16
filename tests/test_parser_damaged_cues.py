"""Damaged / missing cue words in live Whisper output (laptop mic, 2026-10-16).

The first case is verbatim: "पहला, ध्याय, दूसरा, वचन" lost the अ of अध्याय and Vachan showed
"John 1:1 (verse? guessed)" although the verse was in the transcript.
Every expected reference was checked against HIN_IRV in data/bibles.
"""

import pytest

from vachan.parser import ReferenceParser

REAL = "आप सभी को जै मसी की, आज हम पढ़ने जा यूहन्ना, रचित, सुसमाचार, पहला, ध्याय, दूसरा, वचन."


@pytest.fixture(scope="module")
def parser() -> ReferenceParser:
    return ReferenceParser()


def fmt(res) -> str | None:
    return res.reference.format_reference() if res and res.is_valid else None


# Damaged or variant cue words: full confidence is not required, the verse must be kept
DAMAGED = [
    (REAL, "John 1:2"),
    ("आप सभी को जै मसी की, आज हम पढ़ने जा रहे हैं, यूहन्ना, रचित सुसमाचार, पहला, ध्याय, दूसरा, वचन.", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला ध्याय दूसरा वचन", "John 1:2"),
    ("यूहन्ना, रचित, सुसमाचार, पहला, ध्याय, दूसरा, वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला अद्याय दूसरा वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला अध्यय दूसरा वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला अध्याय़ दूसरा वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला द्याय दूसरा वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला अध्याइ दूसरा वचन्", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला ध्याय दूसरा बचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला ध्याय दूसरा पद", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला ध्याय दूसरा आयत", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला ध्याय दूसरा", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला, अध्याय, दूसरा", "John 1:2"),
    ("यूहन्ना... रचित सुसमाचार... पहला ध्याय — दूसरा वचन।", "John 1:2"),
    ("“यूहन्ना रचित सुसमाचार” पहला ध्याय, “दूसरा वचन”", "John 1:2"),
    ("यूहन्ना   रचित   सुसमाचार,,  पहला ,ध्याय , दूसरा ,वचन", "John 1:2"),
    # ordinals with and without matras / Whisper spellings
    ("यूहन्ना रचित सुसमाचार पेहला ध्याय दुसरा वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहले ध्याय का दूसरा वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला ध्याय दूसरे वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार तिसरा ध्याय सोलवा वचन", "John 3:16"),
    ("यूहन्ना रचित सुसमाचार तीसरा ध्याय सोलहवां पद", "John 3:16"),
    ("यूहन्ना रचित सुसमाचार चोथा ध्याय पांचवा वचन", "John 4:5"),
    ("मत्ती रचित सुसमाचार पांचवा ध्याय तिसरा वचन", "Matthew 5:3"),
    ("लूका रचित सुसमाचार चौथा ध्याय सोलहवा पद", "Luke 4:16"),
    ("रोमियों की पत्री आठवा ध्याय अट्ठाईसवा पद", "Romans 8:28"),
    ("भजन संहिता तेईसवा ध्याय पहला पद", "Psalms 23:1"),
    ("उत्पत्ति पहला ध्याय पहला वचन", "Genesis 1:1"),
    # digits
    ("यूहन्ना रचित सुसमाचार 1 ध्याय 2 वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार, 1, ध्याय, 2, वचन.", "John 1:2"),
    ("यूहन्ना १ ध्याय २ वचन", "John 1:2"),
    ("रोमियों 8 द्याय 28 पद", "Romans 8:28"),
    ("यूहन्ना 3 ध्याय 16", "John 3:16"),
    # Roman
    ("yuhanna rachit susamachar pehla dhyay dusra vachan", "John 1:2"),
    ("yahunna rachit susamachar, pehla, dhyay, dusra, vachan.", "John 1:2"),
    ("yuhanna rachit susamachar pehla adyay doosra vachan", "John 1:2"),
    ("yuhanna rachit susamachar pehla adhyaay dusra pad", "John 1:2"),
    ("yuhanna rachit susamachar pehla adhya dusra vachan", "John 1:2"),
    ("yuhanna rachit susamachar pehla dusra", "John 1:2"),
    ("yuhanna rachit susamachar 1 dhyay 2 vachan", "John 1:2"),
    ("romiyon ki patri aathva dhyay atthaisva pad", "Romans 8:28"),
    # English / Hindi mixes
    ("John chapter 1 verse 2", "John 1:2"),
    ("John ch. 1 verse 2", "John 1:2"),
    ("John ke pehle chapter ka dusra verse", "John 1:2"),
    ("जॉन के पहले चैप्टर का दूसरा वर्स", "John 1:2"),
    ("यूहन्ना फर्स्ट चैप्टर सेकंड वर्स", "John 1:2"),
    # cue word missing entirely, or other words between chapter and verse
    ("यूहन्ना रचित सुसमाचार पहला दूसरा", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला, दूसरा.", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला अध्याय और उसका दूसरा वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला ध्याय हम देखते हैं दूसरा वचन", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार पहला ध्याय, आइए पढ़ें दूसरा वचन", "John 1:2"),
    # verse first, chapter second
    ("यूहन्ना रचित सुसमाचार दूसरा वचन पहला अध्याय", "John 1:2"),
    ("यूहन्ना रचित सुसमाचार दूसरा वचन पहले ध्याय का", "John 1:2"),
    ("yuhanna rachit susamachar dusra vachan pehla adhyay", "John 1:2"),
    ("यूहन्ना सोलहवा पद तीसरा अध्याय", "John 3:16"),
    # reference said before the book, damaged cue
    ("पहला ध्याय दूसरा वचन यूहन्ना रचित सुसमाचार", "John 1:2"),
    ("pehla dhyay dusra vachan, yuhanna rachit susamachar", "John 1:2"),
    # other books, damaged cues
    ("प्रेरितों के काम दूसरा ध्याय चौथा वचन", "Acts 2:4"),
    ("इफिसियों दूसरा द्याय आठवा और नौवा पद", "Ephesians 2:8-9"),
    ("कुरिन्थियों की पहली पत्री तेरहवा ध्याय चौथा पद", "1 Corinthians 13:4"),
    ("प्रकाशित वाक्य इक्कीसवा ध्याय चौथा पद", "Revelation 21:4"),
    ("नीतिवचन तीसरा ध्याय पांचवा और छठा पद", "Proverbs 3:5-6"),
]


@pytest.mark.parametrize("text,expected", DAMAGED)
def test_verse_kept_despite_damaged_cue(parser, text, expected):
    res = parser.parse(text)
    assert fmt(res) == expected
    assert "chapter_only_no_verse" not in res.low_confidence_reasons


def test_real_case_is_trusted(parser):
    # "ध्याय" is a known Whisper damage of अध्याय: nothing else is unusual here
    res = parser.parse(REAL)
    assert res.confidence >= 0.9, res


@pytest.mark.parametrize(
    "text",
    [
        "यूहन्ना रचित सुसमाचार पहला दूसरा",
        "yuhanna rachit susamachar pehla dusra",
        "यूहन्ना रचित सुसमाचार पहला ध्याय हम देखते हैं दूसरा वचन",
    ],
)
def test_missing_or_distant_cue_lowers_confidence(parser, text):
    res = parser.parse(text)
    assert fmt(res) == "John 1:2"
    assert res.confidence < 0.9 and res.low_confidence_reasons


@pytest.mark.parametrize(
    "text",
    [
        "यूहन्ना रचित सुसमाचार पहला ध्याय",
        "यूहन्ना रचित सुसमाचार, पहला, ध्याय.",
        "yuhanna rachit susamachar pehla dhyay",
    ],
)
def test_chapter_only_when_no_second_number(parser, text):
    res = parser.parse(text)
    assert fmt(res) == "John 1" and res.reference.verse_start is None


@pytest.mark.parametrize(
    "text",
    [
        "परमेश्वर का वचन हमें सिखाता है",
        "पहला वचन परमेश्वर ने दिया",
        "यह पहला अध्याय है हमारे जीवन का",
        "यूहन्ना ने बपतिस्मा दिया",
        "दो तीन बातें याद रखिए",
        "मैं तीन बजे आया",
        "हमारे जीवन का पहला ध्याय, दूसरा वचन",
        "ध्यान दीजिए, पहला और दूसरा",
        "I sat down with John yesterday and we talked for 2 hours",
        "Mark 2 people as present",
        # Hindi Bible prose that the looser rules first matched (scan of all 31,104 HIN_IRV verses)
        "यहोयाकीम के राज्य के चौथे वर्ष के पांचवें महीने में",
        "गिनती में वे बाईस हजार छः सौ थे",
        "पहलौठा योहानान दूसरा यहोयाकीम तीसरा सिदकिय्याह",
        "तब शमूएल शाऊल के पास गया और शाऊल ने उससे कहा",
        "वहाँ राजा का एक कर्मचारी था जिसका पुत्र बीमार था",
        "आइए हम सब मिलकर दूसरा भजन गाएं",
    ],
)
def test_no_false_positive(parser, text):
    assert fmt(parser.parse(text)) is None


@pytest.mark.parametrize("word", ["ध्याय", "अद्याय", "अध्यय", "द्याय", "dhyay", "adhya", "adyay"])
def test_damaged_chapter_cue_is_a_cue(parser, word):
    from vachan.parser.parser import CHAPTER_CUES

    (token,) = parser._tokens(word)
    assert token in CHAPTER_CUES


@pytest.mark.parametrize("word", ["ध्यान", "अध्यापक", "दया", "आया", "बचपन", "पढ़"])
def test_ordinary_words_are_not_cues(parser, word):
    from vachan.parser.parser import CHAPTER_CUES, VERSE_CUES

    assert parser._canonical(word) not in CHAPTER_CUES | VERSE_CUES


def test_real_mic_galatians_spelling(parser):
    res = parser.parse("गलाथियों की पत्री दूसरा अध्याय, पहला वचन.")  # live mic, 11:22
    assert fmt(res) == "Galatians 2:1" and res.confidence >= 0.9
