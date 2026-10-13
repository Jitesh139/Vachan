"""Benchmark speech-to-text setups on THIS machine with the same test audio.

    python scripts/benchmark_stt.py                     # all candidates
    python scripts/benchmark_stt.py --only fw-small-int8 ov-small-int8-GPU
    python scripts/benchmark_stt.py --md docs/BENCHMARKS.md
    python scripts/benchmark_stt.py --threads 4 8 16 --only fw-small-int8   # thread sweep

Each candidate runs in its own process (clean memory numbers; a crash or a missing
device cannot take the others down). Reported per candidate: load time, seconds to
transcribe 5 s of audio, real-time factor, peak memory, word error rate and how many
test phrases produced the correct Bible reference. Test audio: edge-tts Indian-English
and Hindi voices (cached in tmp/speech_samples; needs internet once).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# (voice, text, language, expected reference)
PHRASES = [
    ("en", "Good morning church, please turn with me to Psalms 91 verse 2.", "en", "Psalms 91:2"),
    ("en", "Open your Bibles to John chapter 3 verse 16.", "en", "John 3:16"),
    ("en", "Let us read Romans chapter 8 verse 28 together.", "en", "Romans 8:28"),
    ("en", "Psalm 119 verse 105 says your word is a lamp to my feet.", "en", "Psalms 119:105"),
    ("en", "First Corinthians chapter 13 verse 4, love is patient.", "en", "1 Corinthians 13:4"),
    ("hi", "आज हम भजन संहिता 91 पद 2 पढ़ेंगे।", "hi", "Psalms 91:2"),
    ("hi", "यूहन्ना तीन अध्याय सोलह पद को देखिए।", "hi", "John 3:16"),
    ("hi", "रोमियों आठ अध्याय अट्ठाईस पद पढ़िए।", "hi", "Romans 8:28"),
    ("hi", "आज हम Psalms 91 verse 2 पढ़ेंगे।", "hi", "Psalms 91:2"),  # Hinglish
    ("hi", "चलिए John chapter 3 verse 16 को पढ़ते हैं।", "hi", "John 3:16"),
    ("hi", "Bhajan Sanhita 91 pad 2 kholiye.", "hi", "Psalms 91:2"),
]
OV = "OpenVINO/whisper-{}-int8-ov"
MAX_TOKENS = 96  # greedy, no temperature fallback: a repetition loop must not cost 25 s
CANDIDATES = {
    "fw-large-v3-turbo-int8": ("faster-whisper", "large-v3-turbo", "cpu"),
    "fw-distil-large-v3-int8": ("faster-whisper", "distil-large-v3", "cpu"),
    "fw-medium-int8": ("faster-whisper", "medium", "cpu"),
    "fw-small-int8": ("faster-whisper", "small", "cpu"),
    "fw-base-int8": ("faster-whisper", "base", "cpu"),
    **{
        f"ov-{m}-int8-{d}": ("openvino", OV.format(m), d)
        for m in ("large-v3-turbo", "medium", "small", "base")
        for d in ("CPU", "GPU", "NPU")
    },
    "ov-distil-large-v3-int8-CPU": ("openvino", "OpenVINO/distil-whisper-large-v3-int8-ov", "CPU"),
}


def physical_cores() -> int:
    import psutil

    return psutil.cpu_count(logical=False) or os.cpu_count() or 4


def clips() -> list[tuple[np.ndarray, str, str, str]]:
    """(16 kHz audio without the padding tail, language, reference text, expected ref)."""
    from test_speech import read_wav, synthesize

    out = []
    for i, (voice, text, lang, expected) in enumerate(PHRASES):
        audio = read_wav(synthesize(voice, text, 100 + i))
        out.append((audio[: len(audio) - 32000], lang, text, expected))  # keep 0.5 s tail
    return out


def words(text: str) -> list[str]:
    """Normalized tokens for WER: the parser's own normalization, numbers as digits."""
    from vachan.parser.normalizer import NumberNormalizer

    n = NumberNormalizer()
    tokens, out, i = n.tokenize(n.normalize(text)), [], 0
    while i < len(tokens):
        num = n.parse_number(tokens, i)
        out.append(str(num[0]) if num else tokens[i])
        i = num[1] if num else i + 1
    return out


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance, reference length) over normalized words."""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


# ------------------------------------------------------------------ one candidate (child)
def load(kind: str, model: str, device: str, threads: int):
    """Return transcribe(audio, language) -> text, using the app's own Whisper prompt."""
    from vachan.core.settings import DEFAULT_PROMPT

    if kind == "faster-whisper":
        from faster_whisper import WhisperModel

        m = WhisperModel(model, device=device, compute_type="int8", cpu_threads=threads)

        def run(audio: np.ndarray, lang: str) -> str:
            segments, _ = m.transcribe(
                audio, language=lang, beam_size=1, condition_on_previous_text=False,
                without_timestamps=True, vad_filter=False, initial_prompt=DEFAULT_PROMPT,
                temperature=0.0, max_new_tokens=MAX_TOKENS,
            )  # fmt: skip
            return " ".join(s.text.strip() for s in segments)

        def detect(audio: np.ndarray) -> str:
            return m.detect_language(audio)[0]

        return run, detect
    import openvino_genai as ov_genai
    from huggingface_hub import snapshot_download

    cache = ROOT / "tmp" / "ov_cache"  # compiled GPU/NPU blobs: later loads are fast
    pipe = ov_genai.WhisperPipeline(snapshot_download(model), device, CACHE_DIR=str(cache))

    def run(audio: np.ndarray, lang: str) -> str:
        res = pipe.generate(
            audio.tolist(), language=f"<|{lang}|>", initial_prompt=DEFAULT_PROMPT, max_new_tokens=MAX_TOKENS
        )
        return str(res).strip()

    def detect(audio: np.ndarray) -> str:
        return "auto"

    return run, detect


def bench_one(name: str, threads: int) -> dict:
    import psutil

    from vachan.parser import ReferenceParser

    kind, model, device = CANDIDATES[name]
    proc = psutil.Process()
    data = clips()
    base_mem = proc.memory_info().rss
    t0 = time.perf_counter()
    run, detect = load(kind, model, device, threads)
    run(np.zeros(16000, np.float32), "en")  # warm-up (GPU/NPU compile, first-call init)
    load_s = time.perf_counter() - t0
    parser = ReferenceParser()
    secs = dur = errs = n_words = hits = 0.0
    rows = []
    for audio, lang, text, expected in data:
        t = time.perf_counter()
        hyp = run(audio, lang)
        took = time.perf_counter() - t
        secs, dur = secs + took, dur + len(audio) / 16000
        e, n = wer(text, hyp)
        errs, n_words = errs + e, n_words + n
        res = parser.parse(hyp)
        got = res.reference.format_reference() if res and res.is_valid else None
        hits += got == expected
        rows.append({"lang": lang, "took": round(took, 2), "text": hyp, "ref": got, "ok": got == expected})
    t = time.perf_counter()
    detect(data[0][0])
    detect_s = time.perf_counter() - t
    mem = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss)
    return {
        "name": name, "ok": True, "load_s": round(load_s, 1), "per_5s": round(secs / dur * 5, 2),
        "rtf": round(secs / dur, 3), "mem_mb": round((mem - base_mem) / 1e6), "wer": round(errs / n_words, 3),
        "refs": f"{int(hits)}/{len(data)}", "detect_s": round(detect_s, 2), "threads": threads, "rows": rows,
    }  # fmt: skip


# ------------------------------------------------------------------ driver
def power_mode() -> str:
    """Windows 11 power-mode slider (an overlay on the power plan)."""
    import winreg

    modes = {"00000000-0000-0000-0000-000000000000": "Balanced",
             "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
             "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency"}  # fmt: skip
    try:
        path = r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
        guid = winreg.QueryValueEx(key, "ActiveOverlayAcPowerScheme")[0]
        return modes.get(guid, guid)
    except OSError:
        return "unknown"


def machine_info() -> str:
    import psutil

    vm = psutil.virtual_memory()
    batt = psutil.sensors_battery()
    power = "unknown" if batt is None else "plugged in" if batt.power_plugged else f"BATTERY {batt.percent}%"
    plan = ""
    if platform.system() == "Windows":
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout
        plan = out.strip().split("(")[-1].rstrip(")") if "(" in out else out.strip()
        plan += f", power mode: {power_mode()}"
    return (
        f"{platform.processor() or platform.machine()} | {physical_cores()} physical cores, "
        f"{os.cpu_count()} logical | RAM {vm.total / 1e9:.1f} GB, {vm.available / 1e9:.1f} GB free | "
        f"power: {power}, plan: {plan}"
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=list(CANDIDATES), help="subset of candidates")
    ap.add_argument("--threads", nargs="+", type=int, help="CPU threads (default: physical cores)")
    ap.add_argument("--md", type=Path, help="write a markdown table here")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--timeout", type=float, default=1200)
    args = ap.parse_args()

    if args.one:
        print("RESULT " + json.dumps(bench_one(args.one, (args.threads or [physical_cores()])[0])))
        return 0

    clips()  # synthesize test audio once, before the timed children
    info = machine_info()
    print(info)
    results = []
    for name in args.only or list(CANDIDATES):
        for threads in args.threads or [physical_cores()]:
            print(f"\n=== {name} (threads={threads})", flush=True)
            cmd = [sys.executable, __file__, "--one", name, "--threads", str(threads)]
            try:
                p = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=args.timeout)
                line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
                if line:
                    r = json.loads(line[7:])
                else:
                    err = [x for x in (p.stderr or "").strip().splitlines() if x.strip()]
                    r = {"name": name, "ok": False, "error": err[-1][:160] if err else f"exit {p.returncode}"}
            except subprocess.TimeoutExpired:
                r = {"name": name, "ok": False, "error": f"timeout after {args.timeout:.0f}s"}
            r["threads"] = threads
            results.append(r)
            if r["ok"]:
                for row in r["rows"]:
                    print(f"  [{row['lang']}] {row['took']:5.2f}s {'OK ' if row['ok'] else 'BAD'} "
                          f"{row['ref']!s:20} {row['text']!r}")  # fmt: skip
            print("  " + (summary(r) if r["ok"] else f"FAILED: {r['error']}"), flush=True)

    table = render(results, info)
    print("\n" + table)
    if args.md:
        args.md.write_text(table, encoding="utf-8")
        (args.md.with_suffix(".json")).write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    return 0


def summary(r: dict) -> str:
    return (f"load {r['load_s']}s | {r['per_5s']}s per 5 s audio | RTF {r['rtf']} | {r['mem_mb']} MB | "
            f"WER {r['wer']:.0%} | refs {r['refs']} | lang-detect {r['detect_s']}s")  # fmt: skip


def render(results: list[dict], info: str) -> str:
    lines = [
        f"Machine: {info}",
        "",
        "| Setup | Threads | Works | Load (s) | s per 5 s clip | RTF | Peak RAM (MB) | WER | Refs correct "
        "| Lang detect (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["ok"]:
            lines.append(
                f"| {r['name']} | {r['threads']} | yes | {r['load_s']} | {r['per_5s']} | {r['rtf']} | "
                f"{r['mem_mb']} | {r['wer']:.0%} | {r['refs']} | {r['detect_s']} |"
            )
        else:
            lines.append(f"| {r['name']} | {r['threads']} | **no**: {r['error']} | | | | | | | |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
