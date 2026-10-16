"""Natural spoken-Hindi citations: descriptor book names ("यूहन्ना रचित सुसमाचार"),
ordinals ("दूसरा अध्याय पहला वचन"), both scripts, and Whisper-style noise.

Every expected reference was checked against the HIN_IRV text in data/bibles.
"""

import pytest

from vachan.core.models import BibleReference
from vachan.parser import ReferenceParser


@pytest.fixture(scope="module")
def parser() -> ReferenceParser:
    return ReferenceParser()


def fmt(res) -> str | None:
    return res.reference.format_reference() if res and res.is_valid else None


SEED = [
    ("यूहन्ना रचित सुसमाचार दूसरा अध्याय पहला वचन", "John 2:1"),
    ("yahunna racchit susamachar dusra adhyay pehla vachan", "John 2:1"),
    ("आइए हम खोलें मत्ती रचित सुसमाचार के पाँचवें अध्याय का तीसरा वचन", "Matthew 5:3"),
    ("luka rachit susamachar chautha adhyay solahwa pad", "Luke 4:16"),
    ("रोमियों की पत्री आठवें अध्याय का अट्ठाईसवाँ पद", "Romans 8:28"),
    ("कुरिन्थियों की पहली पत्री तेरहवाँ अध्याय चौथा पद", "1 Corinthians 13:4"),
    ("भजन संहिता इक्यानवे अध्याय दूसरा पद", "Psalms 91:2"),
    ("प्रेरितों के काम दूसरे अध्याय का चौथा वचन", "Acts 2:4"),
    ("उत्पत्ति पहले अध्याय का पहला वचन", "Genesis 1:1"),
    ("इफिसियों दूसरे अध्याय का आठवाँ और नौवाँ पद", "Ephesians 2:8-9"),
    ("नीतिवचन तीसरे अध्याय पाँचवें और छठे पद", "Proverbs 3:5-6"),
    ("यूहन्ना की पहली पत्री चौथा अध्याय आठवाँ वचन", "1 John 4:8"),
    ("प्रकाशित वाक्य इक्कीसवाँ अध्याय चौथा पद", "Revelation 21:4"),
    ("john ke teesre adhyay ka solahwa vachan", "John 3:16"),
    ("chapter 3 verse 16 of John", "John 3:16"),
]

MORE = [
    # Gospels with descriptors, both scripts
    ("मत्ती का सुसमाचार पाँचवाँ अध्याय तीसरा पद", "Matthew 5:3"),
    ("मरकुस रचित सुसमाचार सोलहवें अध्याय का पंद्रहवाँ पद", "Mark 16:15"),
    ("लूका रचित सुसमाचार दूसरे अध्याय का दसवाँ वचन", "Luke 2:10"),
    ("यूहन्ना का सुसमाचार तीसरा अध्याय सोलहवाँ पद", "John 3:16"),
    ("matti rachit susamachar paanchva adhyay teesra pad", "Matthew 5:3"),
    ("mathi rachit susmachar ke panchve adhyay ka tisra vachan", "Matthew 5:3"),
    ("markus ka susamachar pehla adhyay pehla pad", "Mark 1:1"),
    ("yuhanna rachit susamachar teesra adhyay solahwa vachan", "John 3:16"),
    ("yuhanna ka susamachar teesra adhyay solahvan pad", "John 3:16"),
    ("yohanna rachit susamachar chaudahva adhyay chhatha pad", "John 14:6"),
    ("यूहन्ना रचित सुसमाचार चौदहवें अध्याय का छठा पद", "John 14:6"),
    ("यहुन्ना रचित सुसमाचार चौदहवाँ अध्याय छठा वचन", "John 14:6"),
    ("युहन्ना रचित सुसमाचार 11 अध्याय 25 पद", "John 11:25"),
    ("लुका रचित सुसमाचार पंद्रहवाँ अध्याय ग्यारहवाँ पद", "Luke 15:11"),
    # Epistles: "X की पत्री", "X के नाम पत्री", numbered
    ("रोमियों के नाम पत्री बारहवाँ अध्याय पहला पद", "Romans 12:1"),
    ("रोमियों के नाम पौलुस की पत्री बारहवें अध्याय का दूसरा पद", "Romans 12:2"),
    ("गलतियों की पत्री पाँचवें अध्याय का बाईसवाँ पद", "Galatians 5:22"),
    ("फिलिप्पियों की पत्री चौथे अध्याय का तेरहवाँ पद", "Philippians 4:13"),
    ("इब्रानियों की पत्री ग्यारहवाँ अध्याय पहला पद", "Hebrews 11:1"),
    ("याकूब की पत्री पहला अध्याय पाँचवाँ पद", "James 1:5"),
    ("कुरिन्थियों की दूसरी पत्री पाँचवाँ अध्याय सत्रहवाँ पद", "2 Corinthians 5:17"),
    ("दूसरा कुरिन्थियों पाँचवाँ अध्याय सत्रहवाँ पद", "2 Corinthians 5:17"),
    ("पहला कुरिन्थियों तेरहवें अध्याय का चौथा पद", "1 Corinthians 13:4"),
    ("पहला यूहन्ना पहला अध्याय नौवाँ पद", "1 John 1:9"),
    ("यूहन्ना की पहली पत्री पहले अध्याय का नौवाँ पद", "1 John 1:9"),
    ("यूहन्ना की तीसरी पत्री पहला अध्याय दूसरा पद", "3 John 1:2"),
    ("तीमुथियुस की दूसरी पत्री तीसरा अध्याय सोलहवाँ पद", "2 Timothy 3:16"),
    ("दूसरा तीमुथियुस पहला अध्याय सातवाँ पद", "2 Timothy 1:7"),
    ("पतरस की पहली पत्री पाँचवाँ अध्याय सातवाँ पद", "1 Peter 5:7"),
    ("थिस्सलुनीकियों की पहली पत्री पाँचवाँ अध्याय सत्रहवाँ पद", "1 Thessalonians 5:17"),
    ("kurinthiyon ki pehli patri terahva adhyay chautha pad", "1 Corinthians 13:4"),
    ("romiyon ki patri aathve adhyay ka atthaisva pad", "Romans 8:28"),
    ("yuhanna ki pehli patri chautha adhyay aathva vachan", "1 John 4:8"),
    ("pehla yuhanna chautha adhyay aathva vachan", "1 John 4:8"),
    ("doosra timothy teesra adhyay solahva pad", "2 Timothy 3:16"),
    ("ifisiyon ki patri doosre adhyay ka aathva aur nauva pad", "Ephesians 2:8-9"),
    # Old Testament
    ("उत्पत्ति की पुस्तक पहला अध्याय पहला पद", "Genesis 1:1"),
    ("utpatti pehla adhyay pehla vachan", "Genesis 1:1"),
    ("निर्गमन बीसवें अध्याय का तीसरा पद", "Exodus 20:3"),
    ("यहोशू पहला अध्याय नौवाँ पद", "Joshua 1:9"),
    ("भजन संहिता तेईसवाँ अध्याय पहला पद", "Psalms 23:1"),
    ("भजन संहिता के एक सौ उन्नीसवें अध्याय का एक सौ पाँचवाँ पद", "Psalms 119:105"),
    ("bhajan samhita ikyanve adhyay dusra pad", "Psalms 91:2"),
    ("bhajan sanhita teisva adhyay pehla pad", "Psalms 23:1"),
    ("नीतिवचन तीसरा अध्याय पाँचवाँ पद", "Proverbs 3:5"),
    ("nitivachan teesre adhyay paanchve aur chhathe pad", "Proverbs 3:5-6"),
    ("यशायाह तिरपनवें अध्याय का पाँचवाँ पद", "Isaiah 53:5"),
    ("यशायाह चालीसवें अध्याय का इकतीसवाँ पद", "Isaiah 40:31"),
    ("यिर्मयाह उनतीसवाँ अध्याय ग्यारहवाँ पद", "Jeremiah 29:11"),
    ("विलापगीत तीसरे अध्याय का बाईसवाँ और तेईसवाँ पद", "Lamentations 3:22-23"),
    # Acts / Revelation
    ("प्रेरितों के काम पहले अध्याय का आठवाँ पद", "Acts 1:8"),
    ("preriton ke kaam doosre adhyay ka chautha vachan", "Acts 2:4"),
    ("prakashit vakya ikkisva adhyay chautha pad", "Revelation 21:4"),
    ("प्रकाशितवाक्य तीसरे अध्याय का बीसवाँ पद", "Revelation 3:20"),
    # Order: reference before the book
    ("दूसरा अध्याय पहला वचन, यूहन्ना रचित सुसमाचार", "John 2:1"),
    ("तीसरा अध्याय सोलहवाँ पद यूहन्ना", "John 3:16"),
    ("teesra adhyay solahwa pad yuhanna rachit susamachar", "John 3:16"),
    ("आठवें अध्याय का अट्ठाईसवाँ पद रोमियों की पत्री", "Romans 8:28"),
    # "Book के N अध्याय का M वचन"
    ("यूहन्ना के तीसरे अध्याय के सोलहवें पद में", "John 3:16"),
    ("रोमियों के 8 अध्याय का 28 पद", "Romans 8:28"),
    ("मत्ती के 5वें अध्याय का 3रा पद", "Matthew 5:3"),
    # English / Hindi mixes
    ("John ke 3rd chapter ka 16th verse", "John 3:16"),
    ("Romans ke 8th chapter ka 28th verse dekhiye", "Romans 8:28"),
    ("Matthew ke fifth chapter ka third verse", "Matthew 5:3"),
    ("chapter 8 verse 28 of Romans", "Romans 8:28"),
    ("third chapter sixteenth verse of John", "John 3:16"),
    ("जॉन के थर्ड चैप्टर का सोलहवाँ वर्स", "John 3:16"),
    ("psalm ke 23rd chapter ka first verse", "Psalms 23:1"),
    # Devanagari digits
    ("यूहन्ना ३ अध्याय १६ पद", "John 3:16"),
    ("रोमियों ८ अध्याय २८ वचन", "Romans 8:28"),
    # Whisper-style: commas, fillers, split sentences, no punctuation
    ("यूहन्ना, रचित, सुसमाचार, दूसरा, अध्याय, पहला, वचन।", "John 2:1"),
    ("तो भाइयों और बहनों आइए हम खोलें यूहन्ना रचित सुसमाचार। दूसरा अध्याय। पहला वचन।", "John 2:1"),
    ("आज हम देखेंगे मत्ती रचित सुसमाचार. पाँचवाँ अध्याय. तीसरा पद.", "Matthew 5:3"),
    ("अपनी बाइबल खोलिए रोमियों की पत्री आठवाँ अध्याय और अट्ठाईसवाँ पद", "Romans 8:28"),
    ("toh aaiye hum kholte hain yuhanna rachit susamachar dusra adhyay pehla vachan", "John 2:1"),
    ("उत्पत्ति... पहला अध्याय... पहला पद", "Genesis 1:1"),
    ("भजन संहिता। इक्यानवे अध्याय। दूसरा पद।", "Psalms 91:2"),
    ("यूहन्ना रचित सुसमाचार दुसरा अध्याय पहला वचन", "John 2:1"),
    ("यूहन्ना रचित सुसमाचार दूसरा अध्याइ पहला बचन", "John 2:1"),
    ("यूहन्ना रचीत सुसमाचार दूसरा अध्याय पहला वचन", "John 2:1"),
    ("yuhanna rachit susamachar dusraa adhyaay pahla vachan", "John 2:1"),
    ("yahunna racchit susamachar doosra adhyay pehla wachan", "John 2:1"),
    ("luka rachit susamachar chautha adhyay solahva aayat", "Luke 4:16"),
    ("luka rachit susmachar chautha adyay solva pad", "Luke 4:16"),
    # Chapter only stays chapter only
    ("यूहन्ना रचित सुसमाचार तीसरा अध्याय", "John 3"),
    ("भजन संहिता इक्यानवे अध्याय", "Psalms 91"),
    ("रोमियों की पत्री आठवाँ अध्याय", "Romans 8"),
]

# fmt: off
ORDINALS = [
    ("पहला", 1), ("पहले", 1), ("पहली", 1), ("pehla", 1), ("pahla", 1), ("pehle", 1), ("first", 1),
    ("दूसरा", 2), ("दूसरे", 2), ("दूसरी", 2), ("dusra", 2), ("doosra", 2), ("dusre", 2), ("second", 2),
    ("तीसरा", 3), ("तीसरे", 3), ("teesra", 3), ("teesre", 3), ("tisra", 3), ("3rd", 3),
    ("चौथा", 4), ("चौथे", 4), ("chautha", 4), ("chauthe", 4),
    ("पाँचवाँ", 5), ("पाँचवें", 5), ("पांचवां", 5), ("paanchva", 5), ("panchve", 5),
    ("छठा", 6), ("छठे", 6), ("chhatha", 6), ("chhathe", 6), ("chatha", 6),
    ("सातवाँ", 7), ("सातवें", 7), ("saatva", 7), ("आठवाँ", 8), ("aathva", 8),
    ("नौवाँ", 9), ("नौवें", 9), ("nauva", 9), ("दसवाँ", 10), ("dasva", 10),
    ("ग्यारहवाँ", 11), ("बारहवें", 12), ("तेरहवाँ", 13), ("सोलहवाँ", 16), ("solahwa", 16),
    ("सत्रहवाँ", 17), ("satrahva", 17), ("अट्ठाईसवाँ", 28), ("atthaisva", 28),
    ("इक्यानवें", 91), ("इक्यानवे", 91), ("16th", 16), ("१६", 16), ("२८वाँ", 28),
]
# fmt: on


@pytest.mark.parametrize("text,expected", SEED + MORE)
def test_spoken_hindi_reference(parser, text, expected):
    assert fmt(parser.parse(text)) == expected


@pytest.mark.parametrize("word,value", ORDINALS)
def test_ordinal_as_verse(parser, word, value):
    # Psalms 119 has 176 verses, so every value above is a valid verse
    text = f"भजन संहिता एक सौ उन्नीसवाँ अध्याय {word} पद"
    assert fmt(parser.parse(text)) == f"Psalms 119:{value}"


FALSE_POSITIVES = [
    "परमेश्वर का वचन हमें सिखाता है",
    "पहला वचन परमेश्वर ने दिया",
    "यह पहला अध्याय है हमारे जीवन का",
    "दो तीन बातें याद रखिए",
    "मैं तीन बजे आया",
    "यूहन्ना ने बपतिस्मा दिया",
    "parmeshwar ka vachan hamein sikhata hai",
    "pehla vachan parmeshwar ne diya",
    "yah pehla adhyay hai hamare jeevan ka",
    "do teen baatein yaad rakhiye",
    "main teen baje aaya",
    "yuhanna ne baptisma diya",
    "यूहन्ना रचित सुसमाचार बहुत सुंदर है",
    "मत्ती ने यीशु के बारे में लिखा",
    "हमारे जीवन का दूसरा अध्याय शुरू होता है",
    "लूका एक वैद्य था और उसने दो पुस्तकें लिखीं",
]


@pytest.mark.parametrize("text", FALSE_POSITIVES)
def test_no_false_positive(parser, text):
    assert fmt(parser.parse(text)) is None


@pytest.mark.parametrize("text", FALSE_POSITIVES[:6])
def test_no_false_positive_with_context(parser, text):
    """Only explicit verse follow-ups may use the verse on screen."""
    ctx = BibleReference("JHN", "John", 3, 16)
    res = parser.parse(text, context_reference=ctx)
    assert fmt(res) is None


JOHN_2_1 = BibleReference("JHN", "John", 2, 1)

FOLLOW_UPS = [
    ("अगला वचन", "John 2:2"),
    ("अगली आयत", "John 2:2"),
    ("अगले पद में", "John 2:2"),
    ("agla vachan", "John 2:2"),
    ("agli aayat padhiye", "John 2:2"),
    ("next verse", "John 2:2"),
    ("पिछला पद", "John 2:1"),  # verse 1 has no previous: stays nothing? see test below
    ("अब सातवें वचन में", "John 2:7"),
    ("अब सातवाँ पद देखिए", "John 2:7"),
    ("ab saatve vachan mein", "John 2:7"),
    ("ग्यारहवें पद में", "John 2:11"),
    ("पाँचवें और छठे पद", "John 2:5-6"),
]


@pytest.mark.parametrize("text,expected", [f for f in FOLLOW_UPS if f[0] != "पिछला पद"])
def test_follow_up(parser, text, expected):
    res = parser.parse(text, context_reference=JOHN_2_1)
    assert fmt(res) == expected
    assert res.is_follow_up


@pytest.mark.parametrize("text", ["पिछला पद", "पिछली आयत", "pichla vachan", "previous verse"])
def test_previous(parser, text):
    ctx = BibleReference("JHN", "John", 3, 16)
    assert fmt(parser.parse(text, context_reference=ctx)) == "John 3:15"


def test_verse_only_follow_up_is_lower_confidence(parser):
    res = parser.parse("अब सातवें वचन में", context_reference=JOHN_2_1)
    assert res.confidence < 0.9


def test_bare_yuhanna_with_numbered_epistle_cue(parser):
    assert fmt(parser.parse("यूहन्ना की दूसरी पत्री पहला अध्याय छठा पद")) == "2 John 1:6"


def test_epistle_ambiguity_falls_back_to_gospel(parser):
    # "यूहन्ना की पत्री" without a number: 1 John is the usual one, flagged as unsure
    res = parser.parse("यूहन्ना की पत्री चौथा अध्याय आठवाँ पद")
    assert fmt(res) == "1 John 4:8"
    assert res.confidence < 0.9


def test_chapter_only_keeps_verse_none(parser):
    res = parser.parse("यूहन्ना रचित सुसमाचार तीसरा अध्याय")
    assert res.reference.verse_start is None
    assert "chapter_only_no_verse" in res.low_confidence_reasons


def test_descriptor_is_high_confidence(parser):
    res = parser.parse("यूहन्ना रचित सुसमाचार दूसरा अध्याय पहला वचन")
    assert res.confidence >= 0.9, res
