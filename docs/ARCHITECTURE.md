# Architecture

Ports and adapters. The domain core (`vachan/core`, `vachan/parser`) imports no Qt,
faster-whisper or sounddevice; adapters implement the ports in `vachan/core/ports.py`.

```
 Microphone/mixer ──► LiveAudioSource ──┐          (AudioSource)
                     (ring buffer, 16k) │
                                        ▼
                         PipelineWorker (worker thread, every 1.5 s)
                          │  newest 7 s window → VAD → WhisperTranscriber   (Transcriber)
                          │  prev+current text → ReferenceParser            (ReferenceResolver)
                          │  cooldown / follow-up / chapter-only checks
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
| `core/pipeline.py` | Worker thread: windowing, rolling text buffer, duplicate suppression, stall recovery, `logs/pipeline.log` |
| `core/approval.py` | When auto-show is allowed; operator-friendly warning text |
| `core/settings.py` | `settings.json` (per user, git-ignored) |
| `parser/normalizer.py` | Whisper-output normalization, English/Hindi/Hinglish number parsing |
| `parser/parser.py` | Token scanner: book aliases (exact then fuzzy), chapter/verse structure, follow-ups, confidence |
| `parser/validator.py` | Chapter/verse bounds from `data/verse_counts.json` |
| `audio/source.py` | sounddevice input (WASAPI, native rate, mono ring buffer); `FileAudioSource` for tests |
| `stt/transcriber.py` | faster-whisper: GPU warm-up check, CPU fallback, model download progress, hi/en language choice |
| `bible/provider.py`, `bible/builder.py` | SQLite lookups with KJV fallback; DB builder from eBible.org |
| `ui/` , `output/` | PySide6 operator window, settings dialog, theme; broadcast window |

## Design decisions (and why)

- **Pull, not push, audio.** The PortAudio callback only copies samples into a ring buffer; the worker
  pulls the newest 7 s every 1.5 s and resamples there. A slow transcription never queues stale audio:
  the next step simply takes the newest window.
- **Overlap instead of stitching.** A reference up to ~5 s long is always complete in some window.
  The parser sees the previous + current window text, picks the *last* reference, and a 45 s cooldown
  per reference removes repeats from overlapping windows. After a match the text buffer is cleared.
- **Chapter-only matches wait one step** so "Turn to Romans 8 … verse 28" does not first suggest 8:1.
- **Language per window restricted to Hindi/English** (Urdu probability counted as Hindi) with
  hysteresis. Unrestricted detection turned Hindi speech into romanized gibberish (see AUDIT.md).
- **Prompt without numbers.** Whisper sometimes echoes `initial_prompt`; a prompt containing
  "Romans 8:28" produced that reference from unrelated speech.
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
