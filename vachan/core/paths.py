"""Filesystem locations, valid both from source and inside a PyInstaller build."""

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
    RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
else:
    APP_DIR = Path(__file__).resolve().parent.parent.parent
    RESOURCE_DIR = APP_DIR

DATA_DIR = RESOURCE_DIR / "data"
LOGS_DIR = APP_DIR / "logs"
DEBUG_AUDIO_DIR = APP_DIR / "debug_audio"
SETTINGS_FILE = APP_DIR / "settings.json"
