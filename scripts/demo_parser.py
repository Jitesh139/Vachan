import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vachan.core.models import BibleReference
from vachan.parser import ReferenceParser


def main() -> None:
    # Set stdout encoding to utf-8 for Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = ReferenceParser()

    test_phrases = [
        ("English Standard", "Let us open our Bibles to John 3:16."),
        ("English Spoken Words", "Turn to Matthew chapter five verse seven."),
        ("English with Version", "Romans 8:28 according to ESV."),
        ("English Numbered Book", "Open to 2 Timothy chapter 1 verse 7."),
        ("Verse Range", "Reading Ephesians 2:8-10."),
        ("Devanagari Hindi Standard", "आइए हम यूहन्ना 3:16 पढ़ें।"),
        ("Devanagari Hindi Spoken", "रोमियों 8 अध्याय 28 पद निकालिए।"),
        ("Devanagari Numerals", "मत्ती ५:७ पद"),
        ("Hinglish Code-Switching", "Romans ke aath adhyay mein 28 pad dekhenge."),
        ("Hinglish Short Citation", "Ab hum Yuhanna teen solah padhenge."),
        ("Hinglish Numbered Book", "Pehla Samuel chapter 16 verse 7"),
        ("Canon Out-of-Bounds Rejection", "Please turn to John 99:1."),
    ]

    print("=" * 75)
    print("           VACHAN REFERENCE PARSER - DEMO & TEST RUN")
    print("=" * 75)

    for category, phrase in test_phrases:
        res = parser.parse(phrase)
        print(f"\n[{category}]")
        print(f"  Input:      \"{phrase}\"")
        if res and res.is_valid:
            ref = res.reference
            print(f"  Detected:   --> {ref.format_reference()} <--")
            print(f"              [Book: {ref.book_id} | Chapter: {ref.chapter} | Verse: {ref.verse_start}"
                  + (f"-{ref.verse_end}" if ref.verse_end else "")
                  + (f" | Version: {ref.version}" if ref.version else "")
                  + "]")
            print(f"  Confidence: {res.confidence * 100:.0f}% (Match span: \"{res.raw_match}\")")
        elif res and not res.is_valid:
            print("  Status:     [REJECTED - OUT OF CANONICAL BOUNDS]")
            print(f"  Reason:     {res.low_confidence_reasons}")
        else:
            print("  Status:     [NO SCRIPTURE DETECTED]")

    print("\n" + "=" * 75)
    print("Testing Context-Aware Follow-ups (e.g. Preacher says 'next verse'):")
    print("=" * 75)

    current_context = BibleReference(book_id="JHN", book_name="John", chapter=3, verse_start=16)
    print(f"\nInitial Verse Context: {current_context.format_reference()}")

    follow_ups = [
        "Now look at the next verse.",
        "Go back to the previous verse.",
        "Check verse 21.",
        "Read verses 19 to 21.",
    ]

    for phrase in follow_ups:
        res = parser.parse(phrase, context_reference=current_context)
        if res and res.is_valid:
            print(f"  Preacher: \"{phrase}\"")
            print(f"    --> Automatically Resolved: {res.reference.format_reference()} (Confidence: {res.confidence * 100:.0f}%)")

    print("\n" + "=" * 75)
    print("Demo execution finished successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
