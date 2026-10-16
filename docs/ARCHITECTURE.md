# Architecture

Ports and adapters. The domain core (`vachan/core`, `vachan/parser`) imports no Qt,
faster-whisper or sounddevice; adapters implement the ports in `vachan/core/ports.py`.

```
 Microphone/mixer ──► LiveAudioSource ──┐          (AudioSource)
                     (ring buffer, 16k) │
                                        ▼
                         PipelineWorker (worker thread, tick 0.25 s)
                          │  VAD endpoint → phrase → ProfileTranscriber     (Transcriber)
                          │     NVIDIA CUDA → Intel iGPU (OpenVINO) → CPU
                          │  last 15 s of text → ReferenceParser            (ReferenceResolver)
                          │  cooldown / follow-up / chapter-wait-then-guess
                          │  LocalSqliteBibleProvider → Suggestion          (BibleProvider)
                          ▼
              PipelineEvents callbacks ──► Qt Bridge signals ──► ControlWindow
                                                                   │ Enter / auto-show policy
                                                                   ▼
                                                     VerseDisplayWindow     (OutputSink)
```

## Modules

| Path | Role |
|---|---|
| `core/models.py` | `BibleReference`, `ParseResult`, `Passage`, `Suggestion`, `Transcript`, `ChunkDebugInfo` |
| `core/ports.py` | Protocols: AudioSource, Transcriber, BibleProvider, ReferenceResolver, OutputSink |
| `core/pipeline.py` | Worker thread: VAD endpointing, partials, rolling text buffer, verse wait/guess/upgrade, duplicate suppression, stall recovery, `logs/pipeline.log` |
| `core/approval.py` | When auto-show is allowed; operator-friendly warning text |
| `core/settings.py` | `settings.json` (per user, git-ignored) |
| `parser/normalizer.py` | Whisper-output normalization, English/Hindi/Hinglish number parsing |
| `parser/parser.py` | Token scanner: book aliases (exact then fuzzy), chapter/verse structure, follow-ups, confidence |
| `parser/validator.py` | Chapter/verse bounds from `data/verse_counts.json` |
| `audio/source.py` | sounddevice input (WASAPI, native rate, mono ring buffer); `FileAudioSource` for tests |
| `stt/profiles.py` | Speed profiles (accurate / balanced / fast / auto): first engine of the chain that loads |
| `stt/transcriber.py` | faster-whisper: GPU warm-up check, CPU fallback, model download progress, hi/en language choice |
| `stt/openvino_transcriber.py` | OpenVINO GenAI Whisper on the Intel Arc iGPU (optional, `requirements-openvino.txt`) |
| `bible/provider.py`, `bible/builder.py` | SQLite lookups with KJV fallback; DB builder from eBible.org |
| `ui/` , `output/` | PySide6 operator window, settings dialog, theme; broadcast window |

## Design decisions (and why)

- **Pull, not push, audio.** The PortAudio callback only copies samples into a ring buffer with a
  stream clock; the worker reads `read_since(t)` and resamples there.
- **Endpointing instead of overlapping windows.** Silero VAD finds the end of a phrase
  (`endpoint_sec` pause) and the phrase is transcribed once, so CPU time is not spent re-transcribing
  overlaps. Run-on speech is cut at `max_segment_sec`. Partials (`partial_sec`) only feed the live
  transcript, plus an early commit when two partials agree and more words follow the reference.
  The parser reads the last `context_sec` of text, so "Psalms 91 …pause… verse 2" still joins.
- **Chapter-only matches wait `verse_wait_sec` (3 s) by the clock**, then show `91:1` marked
  *verse? guessed* and keep the text. A later "verse 2" resolves against that chapter and upgrades
  the card, even if the guess is already on air. Every decision is logged in `logs/pipeline.log`.
- **Profiles pick the engine, the benchmark picks the model** (docs/BENCHMARKS.md). A forced CUDA
  load that fails raises, so the chain moves on instead of loading a large model on the CPU.
  The Intel NPU is never used (garbage output / errors).
- **Language per stretch of speech restricted to Hindi/English** (reused until `lang_reset_sec`
  of silence) (Urdu probability counted as Hindi) with
  hysteresis. Unrestricted detection turned Hindi speech into romanized gibberish (see AUDIT.md).
- **Prompt without numbers.** Whisper sometimes echoes `initial_prompt`; a prompt containing
  "Romans 8:28" produced that reference from unrelated speech. The prompt is a vocabulary list
  only, and short: Whisper keeps ~223 prompt tokens and one Hindi word costs 9-17 of them.
- **One spelling key for both scripts** (`normalizer.phonetic`): "पहला"/"pehla"/"pahla" -> "phl",
  "सुसमाचार"/"susmachar" -> "susmchr". Variants snap onto the known cue, number and ordinal words;
  book names matched only through the key are flagged like fuzzy matches. Keys shared by two
  numbers or two books (sat/saath, योना/जॉन) are dropped, never guessed.
- **Descriptors pick the book.** "यूहन्ना रचित सुसमाचार" = John, "यूहन्ना की पहली पत्री" = 1 John,
  "कुरिन्थियों की दूसरी पत्री" = 2 Corinthians; a letter without its number is 1 X, flagged.
- **"वचन" also means "word".** A verse-only follow-up with it ("अब सातवें वचन में") needs an anchor
  word (अब, में, देखिए...), so "पहला वचन परमेश्वर ने दिया" stays nothing.
- **Missed verses are data.** F7 stores the last 30 s of transcript with the right reference in
  `data/corpus/missed.jsonl` (local); `scripts/eval_corpus.py` replays it as a regression set.
- **GPU verified by a warm-up inference** because missing cuBLAS/cuDNN DLLs only fail at the first
  transcription, not at model load. CUDA errors at runtime (e.g. out of memory) switch to CPU.
- **Abbreviations only for typed search.** "he", "am", "is" are book abbreviations but also common
  words; in speech only full names / Hindi names / transliterations count.
- **Opaque output window.** Translucent top-level windows capture as white/black in screen capture.

## Threads

GUI thread (Qt), PortAudio callback thread (copy only), pipeline worker thread (VAD, Whisper,
parsing, SQLite), one-off model preload thread at start-up. Worker → GUI only via queued Qt
signals. `stop()` never blocks the GUI; a quick Stop → Start cannot let the old worker close the
new audio stream.
