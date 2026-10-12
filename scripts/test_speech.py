"""Test speech transcription, parsing, and Bible retrieval end-to-end.

Usage:
    # 1. Generate/test synthesized speech (English and Hindi):
    python scripts/test_speech.py --synth "Turn to John chapter 3 verse 16" --lang en
    python scripts/test_speech.py --synth "यूहन्ना 3 अध्याय 16 पद" --lang hi

    # 2. Test an existing WAV file:
    python scripts/test_speech.py --wav path/to/audio.wav

    # 3. Live record 5-10 seconds from mic:
    python scripts/test_speech.py --record 8
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

# Ensure stdout handles UTF-8 (especially for Hindi on Windows console)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from vachan.bible.provider import LocalSqliteBibleProvider
from vachan.parser.parser import ReferenceParser
from vachan.stt.transcriber import WhisperTranscriber


def synthesize_audio(phrase: str, lang: str = "en") -> np.ndarray:
    """Generate audio using gTTS and return as 16kHz float32 mono numpy array."""
    from gtts import gTTS
    import io
    import miniaudio
    from scipy.signal import resample_poly
    from math import gcd

    print(f"[*] Synthesizing speech via gTTS ({lang}): '{phrase}'...")
    tts = gTTS(text=phrase, lang=lang, slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    mp3_bytes = fp.getvalue()

    decoded = miniaudio.decode(mp3_bytes, nchannels=1, output_format=miniaudio.SampleFormat.FLOAT32)
    sample_rate = decoded.sample_rate
    data = np.array(decoded.samples, dtype=np.float32)

    if sample_rate != 16000:
        up = 16000 // gcd(16000, sample_rate)
        down = sample_rate // gcd(16000, sample_rate)
        data = resample_poly(data, up, down).astype(np.float32)

    return data

    return data


def record_microphone(duration_sec: float = 7.0, device: int = None) -> np.ndarray:
    """Record audio from the microphone and return 16kHz float32 mono."""
    import sounddevice as sd
    from scipy.signal import resample_poly
    from math import gcd

    target_sr = 16000
    native_sr = 16000

    if device is not None:
        dev_info = sd.query_devices(device, "input")
        native_sr = int(dev_info["default_samplerate"])
    else:
        try:
            sd.check_input_settings(samplerate=16000, channels=1)
            native_sr = 16000
        except Exception:
            dev_info = sd.query_devices(kind="input")
            native_sr = int(dev_info["default_samplerate"])

    channels = 1
    print(f"[*] Recording for {duration_sec}s from mic (Native SR: {native_sr}Hz)... Speak now!")
    audio = sd.rec(int(duration_sec * native_sr), samplerate=native_sr, channels=channels, dtype="float32", device=device)
    sd.wait()
    print("[*] Recording complete.")

    audio = audio.flatten()
    if native_sr != target_sr:
        up = target_sr // gcd(target_sr, native_sr)
        down = native_sr // gcd(target_sr, native_sr)
        audio = resample_poly(audio, up, down).astype(np.float32)

    return audio


def run_test(audio_16k: np.ndarray, model_size: str = "base"):
    """Run Whisper STT -> Reference Parser -> Bible Verse retrieval."""
    print("=" * 60)
    print(f"[*] Initializing WhisperTranscriber (model={model_size})...")
    transcriber = WhisperTranscriber(model_size=model_size)
    parser = ReferenceParser()
    bible = LocalSqliteBibleProvider()

    print("[*] Running Whisper transcription...")
    t0 = time.time()
    raw_text, detected_lang = transcriber.transcribe_with_info(audio_16k)
    duration = time.time() - t0

    print(f"[-] STT Time taken: {duration:.2f}s")
    print(f"[-] Raw Whisper text: '{raw_text}'")
    print(f"[-] Detected Language: '{detected_lang}'")

    if not raw_text.strip():
        print("[!] No speech recognized.")
        return

    print("\n[*] Parsing reference from text...")
    parse_result = parser.parse(raw_text)

    if parse_result and parse_result.reference:
        verse_ref = parse_result.reference
        print(f"[+] PARSED REFERENCE DETECTED: {verse_ref.book_id} {verse_ref.chapter}:{verse_ref.verse_start}"
              + (f"-{verse_ref.verse_end}" if verse_ref.verse_end else ""))
        print(f"    Confidence: {parse_result.confidence:.2f}")
        print(f"    Original matched text: '{parse_result.raw_match}'")

        # Determine version based on language or Devanagari characters in transcript
        is_hindi = detected_lang == "hi" or any('\u0900' <= char <= '\u097F' for char in raw_text)
        version = "HIN_IRV" if is_hindi else "KJV"
        verse_text = bible.get_verse_text(
            verse_ref.book_id,
            verse_ref.chapter,
            verse_ref.verse_start,
            verse_ref.verse_end,
            version=version,
        )

        print(f"\n[+] BIBLE VERSE RETRIEVED ({version}):")
        if verse_text:
            print(f"    \"{verse_text}\"")
        else:
            print("    [!] Verse not found in database.")
    else:
        print("[-] PARSER RESULT: No match found.")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="End-to-end Speech to Bible Verse test.")
    parser.add_argument("--synth", type=str, help="Text phrase to synthesize via gTTS and test")
    parser.add_argument("--lang", type=str, default="en", help="Language for synthesis (en or hi)")
    parser.add_argument("--wav", type=str, help="Path to WAV audio file to test")
    parser.add_argument("--record", type=float, help="Record N seconds from mic")
    parser.add_argument("--device", type=int, default=None, help="Input device ID for recording")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, large-v3)")

    args = parser.parse_args()

    if args.synth:
        audio = synthesize_audio(args.synth, lang=args.lang)
    elif args.wav:
        import soundfile as sf
        audio, sr = sf.read(args.wav, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            up = 16000 // gcd(16000, sr)
            down = sr // gcd(16000, sr)
            audio = resample_poly(audio, up, down).astype(np.float32)
    elif args.record:
        audio = record_microphone(duration_sec=args.record, device=args.device)
    else:
        # Default self-test: test English and Hindi synthesized samples
        print("[*] Running standard end-to-end self-test on John 3:16 (English & Hindi)...")
        en_audio = synthesize_audio("Turn to John chapter 3 verse 16", lang="en")
        run_test(en_audio, model_size=args.model)

        hi_audio = synthesize_audio("यूहन्ना 3 अध्याय 16 पद", lang="hi")
        run_test(hi_audio, model_size=args.model)
        return

    run_test(audio, model_size=args.model)


if __name__ == "__main__":
    main()
