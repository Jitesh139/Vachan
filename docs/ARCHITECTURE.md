# Vachan Architecture

This document outlines the system architecture and data processing pipeline for **Vachan**, an automated Bible reference detection and live stream display application.

---

## 🏗 High-Level Pipeline

```mermaid
graph LR
    A[Microphone / Line In] --> B[Audio Module]
    B --> C[STT Module faster-whisper]
    C --> D[Reference Parser Module]
    D --> E[Bible Service SQLite / APIs]
    E --> F[Operator UI PySide6]
    F -->|Approve / Live| G[Output Window / Stream Overlay]
```

### Pipeline Flow:
1. **Audio Capture (`vachan.audio`)**:
   - Streams real-time audio chunks from the selected input device (e.g., soundboard, virtual audio cable, or microphone) using `sounddevice`.
   - Normalizes audio format (16kHz, mono 16-bit PCM) and handles silence filtering.

2. **Speech-to-Text (`vachan.stt`)**:
   - Transcribes chunks in real-time using `faster-whisper` (CTranslate2).
   - Optimized for code-switching (Hindi + English / Hinglish).
   - Outputs rolling transcripts with timestamps.

3. **Reference Parser (`vachan.parser`)**:
   - Analyzes transcribed text using regex, phonetic matching, and fuzzy search across canonical Bible books.
   - Detects book names in Hindi (e.g., *उत्पत्ति*, *यूहन्ना*, *भजन संहिता*), Romanized transliteration (*Yuhanna*, *Bhajan Sanhita*), and English (*Genesis*, *John*, *Psalms*).
   - Extracts chapter and verse numbers (supporting Hindi numerals and spoken number words).

4. **Bible Verse Lookup (`vachan.bible`)**:
   - Retrieves the full text of matched references from local SQLite databases (offline first).
   - Supports switching translations (e.g., Hindi IRV, English ESV, KJV).
   - Optional fallback to external scripture APIs (API.Bible, ESV API).

5. **Operator Approval UI (`vachan.ui`)**:
   - Operator dashboard built in PySide6 with dark mode styling.
   - Displays real-time audio levels, live transcript stream, candidate verse cards, and history.
   - Modes:
     - **Semi-Automatic (Default)**: Operator reviews verse suggestions and clicks "Present" or uses hotkeys (Space/Enter).
     - **Automatic Mode**: High-confidence detections are pushed directly to output after a customizable delay.

6. **Output Window / Broadcast Overlay (`vachan.output`)**:
   - Frameless, transparent overlay window designed for secondary monitors or screen capture in Wirecast/OBS/vMix.
   - Configurable lower-third or full-screen scripture layout with customizable typography, colors, animations, and auto-dismiss timers.

---

## 📂 Module Structure

```text
vachan/
├── __init__.py
├── audio/          # Device discovery, stream listener, audio ring buffer
├── stt/            # Whisper model loader, worker threads, inference pipeline
├── parser/         # Regex extractors, phonetic dictionaries, scripture matcher
├── bible/          # SQLite manager, translation registry, API connectors
├── output/         # Transparent overlay window, styling engine, NDI/OBS hook
├── ui/             # PySide6 main window, operator cards, settings dialogs
└── core/           # Event bus, configuration manager, logging, models
```

---

## ⚡ Concurrency & Threading Model

- **GUI Thread (Main)**: Handles UI updates, Qt event loop, user interaction.
- **Audio Worker Thread**: Background stream loop feeding raw audio frames into a ring buffer.
- **STT Worker Thread**: Batches incoming audio frames and runs inference without blocking the UI.
- **Parser & Lookup Thread**: Processes text snippets and queries SQLite asynchronously, emitting Qt signals (`Signal(DetectedVerse)`) back to the UI.
