"""Vachan - live sermon Bible reference detection. Composition root."""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from vachan.core.paths import LOGS_DIR


def setup_logging() -> None:
    """App log to logs/vachan.log; uncaught exceptions are logged, never shown raw."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOGS_DIR / "vachan.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()])

    def excepthook(exc_type, exc, tb):
        logging.getLogger("vachan").critical("Unhandled error", exc_info=(exc_type, exc, tb))

    sys.excepthook = excepthook
    threading.excepthook = lambda args: excepthook(args.exc_type, args.exc_value, args.exc_traceback)


def ensure_bible_db(app: QApplication) -> str:
    """Build the offline Bible DB on first run. Returns an error message or ''."""
    from vachan.bible.builder import DB_PATH, build

    if DB_PATH.exists():
        return ""
    dialog = QProgressDialog("Downloading Bible texts (one time, about 20 MB)...", None, 0, 0)
    dialog.setWindowTitle("Vachan")
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.show()
    result: dict = {}

    def work() -> None:
        try:
            build()
        except Exception as e:  # network/disk problems are reported in the UI
            result["error"] = e

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    while thread.is_alive():
        app.processEvents()
        thread.join(0.05)
    dialog.close()
    if "error" in result:
        logging.getLogger("vachan").error("Bible download failed: %s", result["error"])
        return (
            f"Bible texts could not be downloaded ({result['error']}). Connect to the internet "
            "and restart Vachan, or run: python scripts\\download_bibles.py"
        )
    return ""


def main() -> int:
    """Wire adapters and windows together and run the Qt event loop."""
    setup_logging()
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("Vachan")

    from vachan.bible.provider import BibleDataMissing, LocalSqliteBibleProvider
    from vachan.core.settings import Settings
    from vachan.output.verse_window import VerseDisplayWindow
    from vachan.parser import ReferenceParser
    from vachan.stt.transcriber import WhisperTranscriber
    from vachan.ui import theme
    from vachan.ui.main_window import ControlWindow

    theme.load_fonts()
    app.setStyleSheet(theme.STYLESHEET)
    settings = Settings.load()
    error = ensure_bible_db(app)
    try:
        bible = LocalSqliteBibleProvider()
    except (BibleDataMissing, OSError) as e:
        bible, error = None, error or str(e)
    try:
        parser = ReferenceParser()
    except Exception as e:  # corrupted data files
        logging.getLogger("vachan").exception("Parser data failed to load")
        QMessageBox.critical(None, "Vachan", f"Reference data in data/ could not be loaded:\n{e}")
        return 1

    output = VerseDisplayWindow(settings.theme, settings.display_mode)
    window = ControlWindow(settings, WhisperTranscriber(settings), parser, bible, output, error)
    window.show()
    output.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
