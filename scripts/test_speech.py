"""End-to-end check: audio -> Whisper transcript -> parsed reference -> verse text.

    python scripts/test_speech.py                 # synthesized sample phrases (needs internet once)
    python scripts/test_speech.py --realtime      # same, streamed through the live pipeline + latency
    python scripts/test_speech.py --wav my.wav    # your own recording(s)
    python scripts/test_speech.py --mic 8         # record 8 s from the selected input
    python scripts/test_speech.py --list-devices
    python scripts/test_speech.py --set hindi --language hi   # natural spoken-Hindi citations
Options: --device cpu|cuda|auto, --model NAME (overrides the GPU/CPU default), --language auto|hi|en.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vachan.audio.source import FileAudioSource, LiveAudioSource, list_input_devices  # noqa: E402
from vachan.bible.provider import LocalSqliteBibleProvider  # noqa: E402
from vachan.core.models import BibleReference  # noqa: E402
from vachan.core.paths import APP_DIR  # noqa: E402
from vachan.core.pipeline import PipelineEvents, PipelineWorker  # noqa: E402
from vachan.core.settings import Settings  # noqa: E402
from vachan.parser import ReferenceParser  # noqa: E402
from vachan.stt.profiles import ProfileTranscriber  # noqa: E402

JOHN_3_16 = BibleReference("JHN", "John", 3, 16)
# (TTS voice, phrase, expected reference, context for follow-ups)
SAMPLES = [
    ("en", "Turn to John chapter three verse sixteen", "John 3:16", None),
    ("en", "Turn to John chapter 3 verse 16", "John 3:16", None),
    ("hi", "यूहन्ना तीन अध्याय सोलह पद", "John 3:16", None),
    ("hi", "यूहन्ना 3 अध्याय 16 पद", "John 3:16", None),
    ("hi", "Romans ke aath adhyay mein 28 pad dekhenge", "Romans 8:28", None),
    ("en", "Now look at the next verse", "John 3:17", JOHN_3_16),
    ("hi", "रोमियों आठ अध्याय अट्ठाईस पद", "Romans 8:28", None),
    ("en", "Open your Bibles to first Corinthians thirteen four", "1 Corinthians 13:4", None),
    ("en", "Please turn with me to Psalms 91 verse 2", "Psalms 91:2", None),
    ("hi", "आज हम भजन संहिता 91 पद 2 पढ़ेंगे", "Psalms 91:2", None),
    ("hi", "आज हम Psalms 91 verse 2 पढ़ेंगे", "Psalms 91:2", None),  # Hinglish
    ("en", "Please turn with me to Psalms 91 | verse 2", "Psalms 91:2", None),  # "|" = 1.5 s pause
    ("hi", "आज हम भजन संहिता 91 | पद 2 पढ़ेंगे", "Psalms 91:2", None),
    ("en", "Please turn with me to Psalms 91", "Psalms 91:1 (GUESS)", None),  # chapter only
]
JOHN_2_1 = BibleReference("JHN", "John", 2, 1)
NOTHING = "nothing"  # expected: no suggestion at all
# Natural Hindi preaching (tests/test_parser_spoken_hindi.py SEED), spoken by a Hindi voice
HINDI_SAMPLES = [
    ("hi", "यूहन्ना रचित सुसमाचार दूसरा अध्याय पहला वचन", "John 2:1", None),
    ("hi", "yahunna racchit susamachar dusra adhyay pehla vachan", "John 2:1", None),
    ("hi", "आइए हम खोलें मत्ती रचित सुसमाचार के पाँचवें अध्याय का तीसरा वचन", "Matthew 5:3", None),
    ("hi", "luka rachit susamachar chautha adhyay solahwa pad", "Luke 4:16", None),
    ("hi", "रोमियों की पत्री आठवें अध्याय का अट्ठाईसवाँ पद", "Romans 8:28", None),
    ("hi", "कुरिन्थियों की पहली पत्री तेरहवाँ अध्याय चौथा पद", "1 Corinthians 13:4", None),
    ("hi", "भजन संहिता इक्यानवे अध्याय दूसरा पद", "Psalms 91:2", None),
    ("hi", "प्रेरितों के काम दूसरे अध्याय का चौथा वचन", "Acts 2:4", None),
    ("hi", "उत्पत्ति पहले अध्याय का पहला वचन", "Genesis 1:1", None),
    ("hi", "इफिसियों दूसरे अध्याय का आठवाँ और नौवाँ पद", "Ephesians 2:8-9", None),
    ("hi", "नीतिवचन तीसरे अध्याय पाँचवें और छठे पद", "Proverbs 3:5-6", None),
    ("hi", "यूहन्ना की पहली पत्री चौथा अध्याय आठवाँ वचन", "1 John 4:8", None),
    ("hi", "प्रकाशित वाक्य इक्कीसवाँ अध्याय चौथा पद", "Revelation 21:4", None),
    ("hi", "john ke teesre adhyay ka solahwa vachan", "John 3:16", None),
    ("en", "chapter 3 verse 16 of John", "John 3:16", None),
    ("hi", "अगला वचन", "John 2:2", JOHN_2_1),
    ("hi", "पिछला पद", "John 3:15", JOHN_3_16),
    ("hi", "अब सातवें वचन में", "John 2:7", JOHN_2_1),
    # the live mic phrase (2026-10-16) and variations
    (
        "hi",
        "आप सभी को जय मसीह की, आज हम पढ़ने जा रहे हैं यूहन्ना रचित सुसमाचार, पहला अध्याय, दूसरा वचन",
        "John 1:2",
        None,
    ),  # fmt: skip
    ("hi", "yahunna rachit susamachar, pehla adhyay, dusra vachan", "John 1:2", None),
    ("hi", "यूहन्ना रचित सुसमाचार दूसरा वचन पहला अध्याय", "John 1:2", None),
    ("hi", "गलतियों की पत्री दूसरा अध्याय पहला वचन", "Galatians 2:1", None),
    ("hi", "यूहन्ना रचित सुसमाचार पहला अध्याय, आइए पढ़ें दूसरा वचन", "John 1:2", None),
    # must produce nothing, even with a verse on screen
    ("hi", "परमेश्वर का वचन हमें सिखाता है", NOTHING, JOHN_2_1),
    ("hi", "पहला वचन परमेश्वर ने दिया", NOTHING, JOHN_2_1),
    ("hi", "यह पहला अध्याय है हमारे जीवन का", NOTHING, JOHN_2_1),
    ("hi", "दो तीन बातें याद रखिए", NOTHING, JOHN_2_1),
    ("hi", "मैं तीन बजे आया", NOTHING, JOHN_2_1),
    ("hi", "यूहन्ना ने बपतिस्मा दिया", NOTHING, JOHN_2_1),
]
SAMPLE_DIR = APP_DIR / "tmp" / "speech_samples"


VOICES = {"en": "en-IN-PrabhatNeural", "hi": "hi-IN-MadhurNeural"}  # Indian accents


def synthesize(voice: str, text: str, index: int) -> Path:
    """edge-tts -> 16 kHz mono WAV with 0.5 s lead-in and 2.5 s tail of silence (cached).

    'a | b' speaks a, pauses 1.5 s, then speaks b (a reference split by a pause).
    """
    voice = VOICES.get(voice, voice)
    key = hashlib.md5(text.encode()).hexdigest()[:8]  # cache by text: lists get edited
    path = SAMPLE_DIR / f"sample_{index}_{voice}_{key}.wav"
    if "|" in text:
        if not path.exists():
            texts = [t.strip() for t in text.split("|")]
            parts = [read_wav(synthesize(voice, t, f"{index}_{k}")) for k, t in enumerate(texts)]
            # each part: 0.5 s lead-in + speech + 2.5 s tail -> trim to a 1.5 s pause between parts
            last = len(parts) - 1
            parts = [x[(8000 if k else 0) : len(x) - (16000 if k < last else 0)] for k, x in enumerate(parts)]
            write_wav(path, np.concatenate(parts))
        return path
    if path.exists():
        return path
    import asyncio

    import edge_tts
    import miniaudio
    from scipy.signal import resample_poly

    async def fetch() -> bytes:
        chunks = [c["data"] async for c in edge_tts.Communicate(text, voice).stream() if c["type"] == "audio"]
        return b"".join(chunks)

    for attempt in range(5):  # the TTS service drops rapid repeat connections now and then
        try:
            data = asyncio.run(fetch())
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(5 * 2**attempt)
    dec = miniaudio.decode(data, nchannels=1, output_format=miniaudio.SampleFormat.FLOAT32)
    audio = resample_poly(np.array(dec.samples, np.float32), 16000, dec.sample_rate)
    audio = np.concatenate([np.zeros(8000), audio, np.zeros(40000)])
    write_wav(path, audio)
    return path


def write_wav(path: Path, audio: np.ndarray) -> None:
    """Save 16 kHz mono float audio as 16-bit PCM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def read_wav(path: Path) -> np.ndarray:
    """Load a WAV file as 16 kHz mono float32."""
    src = FileAudioSource(path, loop=False)
    from scipy.signal import resample_poly

    sr = src._file_sr
    return src._audio if sr == 16000 else resample_poly(src._audio, 16000, sr).astype(np.float32)


ROWS: list[tuple[str, str, str, str, bool]] = []  # spoken, whisper, parsed, expected, ok


def run_direct(worker: PipelineWorker, audio: np.ndarray, expected: str | None) -> bool:
    """Whole clip through Whisper, then the pipeline's own parse + lookup code."""
    t = worker.transcriber.transcribe(audio)
    print(f"  transcript ({t.language}, {t.seconds:.2f}s): {t.text!r}")
    ROWS.append(("", f"[{t.language}] {t.text}", "", expected or "", False))
    line, suggestion = worker._handle_text(t.text)
    if suggestion is None:  # chapter only: let the verse wait run out
        worker._clock = lambda: time.monotonic() + worker.settings.verse_wait_sec + 1
        suggestion = worker._check_pending()
        worker._clock = time.monotonic
    return report(line, suggestion, expected)


def run_realtime(worker: PipelineWorker, path: Path, expected: str | None) -> bool:
    """Stream the file in real time through the live pipeline; the LAST suggestion counts
    (a guess may be upgraded). Latency is the pipeline's own end-of-phrase measurement."""
    got: list = []
    delays: list = []
    worker.events = PipelineEvents(
        suggestion=got.append,
        latency=delays.append,
        transcript=lambda text, lang: print(f"  transcript ({lang}): {text!r}"),
        error=lambda e: print(f"  ERROR: {e}"),
    )
    worker.audio = FileAudioSource(path, loop=False)
    worker.start()
    speech = len(read_wav(path)) / 16000 - 2.5  # tail silence added by synthesize()
    time.sleep(speech + worker.settings.verse_wait_sec + 4)  # room for a guess and its upgrade
    worker.stop(wait=5)
    for s in got:
        print(f"  suggestion: {label(s)} ({s.action})")
    if delays:
        print(f"  end of phrase -> suggestion: {delays[-1]:.2f}s")
    return report("no suggestion" if not got else "MATCH", got[-1] if got else None, expected)


def label(s) -> str:
    """'Psalms 91:2', or 'Psalms 91:1 (GUESS)' when the verse was not spoken."""
    return s.reference.format_reference() + (" (GUESS)" if s.is_guess else "")


def report(line: str, suggestion, expected: str | None) -> bool:
    """Print transcript -> reference -> verse and return pass/fail."""
    if suggestion is None:
        print(f"  parser: {line}")
        ok = expected in (None, NOTHING)
    else:
        ref = label(suggestion)
        print(f"  reference: {ref}  (confidence {suggestion.result.confidence:.2f})")
        print(f"  verse [{suggestion.primary.version}]: {suggestion.primary.text[:110]}")
        if suggestion.secondary:
            print(f"  verse [{suggestion.secondary.version}]: {suggestion.secondary.text[:110]}")
        ok = expected is None or ref == expected
    if expected:
        print(f"  {'PASS' if ok else 'FAIL'} (expected {expected})")
    if ROWS and not ROWS[-1][2]:
        ROWS[-1] = (ROWS[-1][0], ROWS[-1][1], label(suggestion) if suggestion else "-", ROWS[-1][3], ok)
    return ok


def main() -> int:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Speech -> verse end-to-end test")
    ap.add_argument("--wav", nargs="+", type=Path)
    ap.add_argument("--mic", type=float, metavar="SECONDS")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--profile", default=None, choices=["auto", "accurate", "balanced", "fast", "custom"])
    ap.add_argument("--language", default=None, choices=["auto", "hi", "en"])
    ap.add_argument("--set", default="default", choices=["default", "hindi"], help="sample phrases")
    args = ap.parse_args()

    if args.list_devices:
        for d in list_input_devices():
            print(f"{'*' if d['is_default'] else ' '} [{d['index']}] {d['name']}")
        return 0

    settings = Settings.load()
    if args.device:
        settings.device = args.device
    if args.model:
        settings.model_gpu = settings.model_cpu = args.model
    if args.device or args.model:
        settings.profile = "custom"
    if args.profile:
        settings.profile = args.profile
    if args.language:
        settings.language = args.language
    stt = ProfileTranscriber(settings)
    stt.load(lambda msg: print(f"[stt] {msg}"))
    if stt.fallback_reason:
        print(f"[stt] note: {stt.fallback_reason}")
    worker = PipelineWorker(
        LiveAudioSource(settings.input_device), stt, ReferenceParser(),
        LocalSqliteBibleProvider(), settings,
    )  # fmt: skip

    cases: list[tuple[str, Path | np.ndarray, str | None, BibleReference | None]] = []
    if args.mic:
        src = worker.audio
        src.start()
        print(f"Recording {args.mic:.0f}s from {src.device_name} - speak now...")
        time.sleep(args.mic)
        audio = src.read_latest(args.mic)
        src.stop()
        path = APP_DIR / "tmp" / "mic_recording.wav"
        write_wav(path, audio)
        print(f"saved {path} (RMS {float(np.sqrt(np.mean(audio**2))):.4f})")
        cases.append(("microphone", path, None, None))
    elif args.wav:
        cases += [(str(p), p, None, None) for p in args.wav]
    else:
        samples, prefix = (HINDI_SAMPLES, "h") if args.set == "hindi" else (SAMPLES, "")
        for i, (voice, text, expected, ctx) in enumerate(samples):
            cases.append((text, synthesize(voice, text, f"{prefix}{i}" if prefix else i), expected, ctx))

    passed = 0
    for label, path, expected, ctx in cases:
        print(f"\n=== {label}")
        worker._state = type(worker._state)()  # fresh buffers/cooldowns per case
        worker.set_context(ctx)
        if args.realtime:
            ok = run_realtime(worker, path, expected)
        else:
            ok = run_direct(worker, read_wav(path), expected)
        passed += ok
        if ROWS and not ROWS[-1][0]:
            ROWS[-1] = (label,) + ROWS[-1][1:]
    if ROWS:
        print("\n| spoken | Whisper wrote | parser | expected | ok |\n|---|---|---|---|---|")
        for spoken, heard, got, want, ok in ROWS:
            print(f"| {spoken} | {heard} | {got} | {want} | {'PASS' if ok else 'FAIL'} |")
    engine = f"{stt.profile}: Whisper {stt.model_name} on {stt.device.upper()}"
    print(f"\n{passed}/{len(cases)} passed  ({engine})")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
