# Vachan

A Windows desktop app that listens to sermon audio in Hindi and English, detects Bible references, and displays the verse on screen for live streams.

> **Status:** In development 🚧

---

## 📖 Overview

**Vachan** (वचन) is designed for live church production and broadcasting. During a sermon, preachers frequently switch naturally between Hindi and English ("Hinglish") and quote scriptures on the fly. Vachan captures live audio, transcribes speech in real time, extracts chapter and verse citations, fetches scripture text, and presents it to stream operators for live broadcast display.

## 🚀 Planned Features

- **Live Multi-Lingual Audio Stream**: Real-time microphone/line-in capture with low latency.
- **Hinglish Speech-to-Text**: Mixed Hindi + English audio transcription via `faster-whisper`.
- **Intelligent Bible Reference Parsing**: Robust regex and phonetic/NLP pattern matching capable of identifying book names, chapters, and verses in Hindi, English, and Romanized Hindi.
- **Offline & Online Scripture Lookup**: Fast multi-version scripture search with local SQLite storage (Hindi BSI / IRV, English KJV, ESV, NIV).
- **Operator-in-the-Loop Approval UI**: PySide6 dark-mode desktop dashboard allowing technicians to preview, approve, edit, or reject verses before going live.
- **Live Stream Integration**: Lower-third overlays, full-screen graphics, and broadcast outputs compatible with Wirecast, OBS, and vMix (via NDI, transparent window, or browser source).

## 🛠 Tech Stack

- **Platform:** Windows Desktop (Windows 10/11)
- **Language & Runtime:** Python 3.11+
- **GUI Framework:** PySide6 (Qt for Python)
- **Speech-to-Text Engine:** `faster-whisper` (CTranslate2)
- **Database:** SQLite (local Bible databases & caching)
- **Styling & Overlay Output:** Qt Graphics / Transparent Window / HTML OBS dock

## 🖥 Target Hardware & Acceleration Note

- **Recommended GPU:** NVIDIA GeForce RTX 5060 (Blackwell Architecture) or modern NVIDIA RTX GPU.
- **CUDA Requirement:** RTX 50-series Blackwell GPUs require **CUDA 12.8+** compatible PyTorch and CTranslate2 builds.
- *Note:* Heavy GPU dependencies (PyTorch with CUDA 12.8, cuDNN) are not installed by default in the starter dependencies to keep initial setup light and portable. Detailed GPU configuration instructions will be provided as STT models are integrated.

## ⚙️ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Jitesh139/Vachan.git
cd Vachan
```

### 2. Set Up Virtual Environment (Windows)
Run the automated setup script:
```cmd
setup.bat
```
Or manually:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```cmd
copy .env.example .env
```
Add optional API keys if online lookups or LLM fallback are desired.

### 4. Run the Application
```cmd
python main.py
```

## 🧪 Development & Testing

```cmd
# Run linter
ruff check .

# Run format check
black --check .

# Run unit tests
pytest
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
