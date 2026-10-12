# Vachan Audit — 2026-10-12

Scope: every tracked file read; tests, linters, the pipeline, real faster-whisper runs and the UI exercised.
Audit machine: Intel Core Ultra 7 155H, Intel Arc iGPU, **no NVIDIA GPU**, Python 3.11.9,
ctranslate2 4.8.2, faster-whisper 1.2.1, PySide6 6.12. GPU behaviour on the RTX 5060 could not be
measured here — see "Unverified on target hardware" at the bottom.

## Root cause of "audio is captured but no verse is detected"

There are **two independent bugs**, either of which alone kills the feature:

1. `vachan/core/pipeline.py:187` calls `time.time()` but `time` is never imported. Every chunk raises
   `NameError` *after* transcription and *before* the transcript is emitted or logged, so the UI shows
   no transcript, no suggestion, and `logs/pipeline.log` is never written (the `logs/` folder on disk
   is empty). The exception is swallowed into a status-bar "Notice". Evidence (pipeline driven with a
   fake transcriber that returns a perfect "Turn to John chapter 3 verse 16."):

   ```
   File "vachan\core\pipeline.py", line 187, in _process_loop
       now_epoch = time.time()
   NameError: name 'time' is not defined
   ON_ERROR: name 'time' is not defined
   ```
   `ruff check` reports it as `F821 Undefined name 'time'` — CI never ran ruff on real deps.

2. The Bible database is a stub: `data/bibles/vachan_bibles.db` is 36 KB with
   `[('ASV', 1), ('HIN_IRV', 13), ('KJV', 13), ('WEB', 13)]` verses. Both download URLs in
   `scripts/download_bibles.py` return **HTTP 404**. The pipeline only emits a suggestion
   `if text_primary` (`pipeline.py:275`), so even a correct parse of e.g. "Acts 2:38" is silently dropped.

Fixing (1) alone would have made only 13 hand-seeded verses work — which is why the earlier fix
"didn't work" in practice.

## Critical (breaks the core flow)

| # | Where | Problem | Evidence | Planned fix |
|---|---|---|---|---|
| C1 | `vachan/core/pipeline.py:187` | `time` not imported → every chunk crashes | traceback above; ruff F821 | Rewrite pipeline loop (see H-items); add a pipeline test with fake STT |
| C2 | `scripts/download_bibles.py:35-60`, `data/bibles/vachan_bibles.db` | DB has 13 verses; source URLs 404 | sqlite count + 404s above | Build DB from eBible.org VPL files (KJV, ASV, WEB, Hindi IRV); map eBible book codes (`JOH`,`MAR`,`SOL`,`EZE`,`JOE`,`NAH`,`JAM`,`PHI`,`1JO`..) to USFM; strip `¶` and Hindi cross-refs `(इब्रा. 1:10)` |
| C3 | `vachan/stt/transcriber.py:21` | `initial_prompt` lists concrete references; Whisper **echoes the prompt** → wrong verse | `small`, prompt on, audio "Romans ke aath adhyay mein 28 pad" → `'रोमियों 8, भजन संहिता 23'` → parsed **Psalms 23:1** | Vocabulary-only prompt (book names + chapter/verse words, no numbers), configurable |
| C4 | `vachan/stt/transcriber.py:127-165` | Language auto-detect on short Hindi clips picks `en`, producing romanized mush; `ur` accepted (Urdu script, unparseable) | `small`: "यूहन्ना तीन अध्याय सोलह पद" → `lang=en(0.33)` `'Yuhanna Tiin Adhyay Sola Hapad'` | Detect language restricted to {hi, en} (hi+ur vs en), configurable force; verified: turbo → `'यूहन्ना, तीन, अध्याय, सोलह, पद.'` |
| C5 | `vachan/parser/parser.py` (aliases) | Whisper writes English book names in Devanagari and misspells अध्याय; parser misses them | `'रूमेंस के आठ अध्याय में 28 पद देखेंगे.'` → None; `'रोमेंस के …'` → None | Normalise Whisper output (अद्याय/अध्याइ→अध्याय etc.), add Devanagari transliterations of English names (रोमन्स/रोमेंस/जॉन/…) |
| C6 | `vachan/stt/transcriber.py:55-108` | GPU failure is only caught at model *load*. With faster-whisper on Windows, missing `cublas64_12.dll`/cuDNN fails at the **first transcribe**, so every chunk errors and there is no CPU fallback. `requirements.txt` has no `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` | code path; ctranslate2 raises `CUDA driver version is insufficient` on this PC at compute-type query | Warm-up inference inside load; fall back to CPU on any failure; add CUDA wheels + DLL dir registration; report device in UI |
| C7 | `main.py:36,39` | Window 2.5 s (too short: a reference is often cut in half), `large-v3` hard-coded (≈3 GB, very slow on CPU), mic index `1` hard-coded (a different PC's Intel array mic) | file | Settings-driven: `large-v3-turbo` on CUDA, `small` on CPU, 7 s window / 2 s step, default input device |

## High

| # | Where | Problem | Evidence | Planned fix |
|---|---|---|---|---|
| H1 | `vachan/audio/source.py:233-247` | Sample-rate fallback (most WASAPI/USB interfaces reject 16 kHz) sets `self.chunk_samples` (unused) instead of `window_samples`/`step_samples` → at 48 kHz the "2.5 s" window is 0.83 s. `dev_info` may be unbound → `NameError` | code reading | Compute windows from the *stream* rate after opening; always resample to 16 kHz |
| H2 | `vachan/audio/source.py:145-190` | Concatenate + resample + normalize + callback all run inside the PortAudio callback; `status` (overflow) ignored | code | Callback only copies into a ring buffer; windowing/resampling on the worker thread |
| H3 | `vachan/audio/source.py` | Mic unplug / stream error is never detected — app keeps "listening" to nothing | no `finished_callback`, no watchdog | Stream finished-callback + "no audio for N s" watchdog → clear on-screen error |
| H4 | `vachan/core/pipeline.py:186-243` | Rolling text buffer concatenates transcripts of *overlapping* windows, so the same words appear 2–3×; dedupe is only against the immediately previous match; "consume" by substring search can fail | design | Parse the current 7 s window's text plus the tail of the previous one; dedupe by (ref, time) cooldown |
| H5 | `vachan/parser/parser.py:334` | Aliases tried longest-first, not by position → in a buffer with two references the older/random one wins | `'John 3:16 … And Romans 8 28 says'` → Romans 8:28; | Pick the **last** valid reference by text position |
| H6 | `vachan/parser/parser.py` aliases | ≤2-letter abbreviations ("he", "am", "is", "so"…) match ordinary words in speech | `'He gave me one of those looks. Mark my words, in 2 weeks'` → **Hebrews 1:2 (0.70)** | Short abbreviations only for typed (manual) search, never for speech |
| H7 | `vachan/core/pipeline.py:54` | Queue of 15 × 7 s windows: on a slow model, latency grows to minutes before dropping | design | Keep only the newest window (maxsize 1-2, drop oldest) — stale audio is worthless live |
| H8 | `vachan/ui/main_window.py:515-525` | Model loads lazily on the first chunk inside the worker; status says "Listening on CPU" before anything loaded; first-run 1.5–3 GB download has no progress | code | Load model on a background thread at start with status/progress and errors shown |
| H9 | `vachan/ui/main_window.py:349,669` | Enter in the manual-search box also bubbles to `keyPressEvent` → `_approve_verse` (to verify in Phase 4) | code reading | Separate Enter handling; search only fills the suggestion card |
| H10 | `vachan/ui/main_window.py:584` | Auto-show threshold 0.80 while almost every parse scores 0.85–0.95 (even "no cue words") → auto mode shows nearly everything | parser probe | Recalibrate confidence; auto mode needs ≥0.9 and no warnings |
| H11 | `vachan/bible/provider.py:63-74` | Unknown version silently falls back to KJV, no notice; new connection per query; `_ensure_db` imports `scripts.` package at runtime | code | Return used version + fallback flag; UI notice; single connection |
| H12 | `run.bat` | Clean clone: no venv creation, no `pip install` → `ModuleNotFoundError` on a fresh machine | file | run.bat creates venv, installs pinned deps, GPU check, builds Bible DB, launches |
| H13 | `requirements.txt`, `pyproject.toml` | Unpinned; `scipy` used but missing; `rapidfuzz` missing from pyproject; CUDA wheels missing | files | Pin exact tested versions |

## Medium

| # | Where | Problem | Planned fix |
|---|---|---|---|
| M1 | `vachan/output/verse_window.py:108,150` | Second `setGraphicsEffect` destroys the drop shadow (Qt allows one effect per widget) | Shadow on card, opacity on inner wrapper |
| M2 | `verse_window.py:89` | `WA_TranslucentBackground` renders white in grabs/window capture (see `docs/screenshots/verse_window.png`) | Opaque themed background (chroma green option) |
| M3 | `verse_window.py:101` | Card fixed 980 px; no auto-fit → long verses/ranges overflow, fullscreen 1080p/4K looks small | Card width relative to window, font auto-fit |
| M4 | `vachan/ui/main_window.py` | Top bar ~300 px tall of empty space (see `docs/screenshots/control_window.png`); no history, no settings dialog, no confidence warnings | Phase 4 redesign |
| M5 | `settings.json` | Not read by any code (`grep` finds no loader); values contradict `main.py` | Real settings load/save |
| M6 | `vachan/stt/transcriber.py:88` | `cpu_threads=4` hard-coded | Use physical-core count |
| M7 | `vachan/parser/normalizer.py` | Romanized Hindi number words (`sat`, `char`, `tin`, `das`) rewritten inside English sentences | Acceptable only near cue words; covered by tests |
| M8 | `vachan/core/pipeline.py:92` | `stop()` joins 1 s while a transcribe may run for longer; daemon thread keeps using model | Join properly; ignore results after stop |
| M9 | `verse_window.py:240` | Esc on the output window hides it (easy to blank a live stream by accident); F11 does not target monitor 2 as README claims | Esc does nothing live; F11 fullscreen on the window's current screen (documented) |
| M10 | `.github/workflows/ci.yml` | Installs only ruff+pytest; tests import `rapidfuzz` → CI cannot pass | Install requirements |

## Low

| # | Where | Problem | Planned fix |
|---|---|---|---|
| L1 | repo | ruff: 297 issues (253 E501, 17 F401, 16 I001, 5 F601 duplicate keys in `scripts/build_numbers.py`); black: 21 files unformatted | ruff + black clean |
| L2 | `verse_window.py:246` | Compares mouse button to `Qt.Key.Key_Left` | Remove |
| L3 | `scripts/test_speech.py:62` | Duplicate `return data`; English-only default test | Rewrite (Phase 2) |
| L4 | `README.md` | Claims 44 tests, "notifies operator" on unavailable version, F11 "on monitor 2", full offline Bible — none true | Rewrite (Phase 7) |
| L5 | `.env.example` | Advertises ESV/API.Bible/LLM keys that no code uses | Remove |
| L6 | `scripts/audio_self_test.py`, `scripts/probe_devices.py`, `scripts/capture_ui.py` | Hard-coded device 1, overlapping purposes | Fold into `test_speech.py --list-devices`; keep capture_ui |
| L7 | `vachan/core/models.py` | `ChunkDebugInfo` lacks device name / timing | Add device + STT latency |
| L8 | `.gitignore` | Good: models, wav, logs, db, .env ignored. `*.spec` ignored would hide the PyInstaller spec | Un-ignore `vachan.spec` |

## Parser vs real Whisper output (measured)

`small` int8 CPU, gTTS audio, current parser:

| Audio | Whisper text | Parsed |
|---|---|---|
| Turn to John chapter three verse sixteen | `Turn to John chapter 3 verse 16` | John 3:16 ✔ |
| यूहन्ना तीन अध्याय सोलह पद | `Yuhanna Tiin Adhyay Sola Hapad` (lang=en 0.33) | None ✘ |
| Romans ke aath adhyay mein 28 pad dekhenge | `रूमेन्स के 8 अद्याय में 28 पद देखेंगे` | None ✘ |
| first Corinthians thirteen four | `1 Corinthians 13-4.` | 1 Cor 13:4 ✔ |

`large-v3-turbo`, vocabulary prompt, language restricted to hi/en:

| Audio | Whisper text |
|---|---|
| यूहन्ना तीन अध्याय सोलह पद | `यूहन्ना, तीन, अध्याय, सोलह, पद.` |
| Romans ke aath adhyay mein 28 pad dekhenge | `रूमेंस के आठ अध्याय में 28 पद देखेंगे.` |
| रोमियों आठ अध्याय अट्ठाईस पद | `रोमियो आठ अध्याय, अध्याय, अध्याइस पद.` |

Takeaways: Hindi audio must be decoded as Hindi; the parser must accept Devanagari transliterations of
English book names, commas between tokens, and misspelled number words.

Latency on this CPU-only PC: `small` ≈ 8 s, `large-v3-turbo` 16–37 s per 3 s clip. The 3-second
target is only reachable on the GPU.

## Unverified on target hardware

- CUDA on the RTX 5060 (Blackwell, sm_120): needs ctranslate2 ≥ 4.6 with CUDA 12.x cuBLAS + cuDNN 9.
  Cannot be tested on this machine.
- Real microphone / church mixer input, Wirecast capture.
