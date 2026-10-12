"""Shared look: palette, Qt stylesheet and bundled Noto fonts."""

from __future__ import annotations

import logging

from PySide6.QtGui import QFontDatabase

from vachan.core.paths import RESOURCE_DIR

log = logging.getLogger(__name__)

BG = "#0E1016"
SURFACE = "#161922"
SURFACE_2 = "#1D212C"
BORDER = "#2A2F3D"
TEXT = "#E9EBF1"
MUTED = "#8B91A3"
ACCENT = "#E3B75C"  # warm gold
ACCENT_DIM = "#3A3020"
GREEN = "#3FBF7F"
RED = "#E5605A"
AMBER = "#E8A33D"

UI_FONT = "Segoe UI"
SERIF = "Noto Serif"
SERIF_DEVA = "Noto Serif Devanagari"
SANS_DEVA = "Noto Sans Devanagari"


def tint(color: str, alpha: int) -> str:
    """CSS rgba() for a #RRGGBB color (Qt reads 8-digit hex as #AARRGGBB)."""
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def load_fonts() -> None:
    """Register the bundled OFL fonts; fall back to system fonts if missing."""
    for path in sorted((RESOURCE_DIR / "assets" / "fonts").glob("*.ttf")):
        if QFontDatabase.addApplicationFont(str(path)) < 0:
            log.warning("Could not load font %s", path)


STYLESHEET = f"""
* {{ font-family: '{UI_FONT}', '{SANS_DEVA}'; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; }}
QFrame#Card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#Suggestion {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #22242C, stop:1 #1A1C24);
    border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#Suggestion[state="active"] {{ border: 1px solid {ACCENT}; }}
QFrame#Suggestion[state="warn"] {{ border: 1px solid {AMBER}; }}
QFrame#Header {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
QFrame#Banner {{ background: #3A1F22; border: 1px solid {RED}; border-radius: 10px; }}
QLabel#Title {{ font-size: 15pt; font-weight: 600; }}
QLabel#Section {{ color: {MUTED}; font-size: 8.5pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#RefBig {{ font-size: 17pt; font-weight: 600; color: {ACCENT}; }}
QLabel#Badge {{ border-radius: 9px; padding: 2px 10px; font-weight: 600; font-size: 9pt; }}
QLabel#Warning {{ color: {AMBER}; }}
QLabel#VerseText {{ font-family: '{SERIF}', '{SERIF_DEVA}'; font-size: 12.5pt; }}
QLabel#VerseTextHi {{ font-family: '{SERIF_DEVA}', '{SERIF}'; font-size: 12.5pt; color: #C9CCD6; }}
QPushButton {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 9px;
    padding: 7px 14px; font-weight: 600; }}
QPushButton:hover {{ border-color: #4A5164; background: #232836; }}
QPushButton:pressed {{ background: #1A1E28; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #5A6072; border-color: #23272F; }}
QPushButton#Primary {{ background: {ACCENT}; color: #1B1508; border: 1px solid {ACCENT}; }}
QPushButton#Primary:hover {{ background: #EEC673; }}
QPushButton#Primary:disabled {{ background: #4A4030; color: #7A705C; border-color: #4A4030; }}
QPushButton#Start {{ background: {GREEN}; color: #08140E; border: 1px solid {GREEN}; padding: 8px 18px; }}
QPushButton#Start:hover {{ background: #55D493; }}
QPushButton#Stop {{ background: {RED}; color: #1E0807; border: 1px solid {RED}; padding: 8px 18px; }}
QPushButton#Ghost {{ background: transparent; border: 1px solid transparent; color: {MUTED}; }}
QPushButton#Ghost:hover {{ color: {TEXT}; border-color: {BORDER}; }}
QPushButton#Seg {{ border-radius: 0; padding: 6px 12px; }}
QPushButton#Seg:checked {{ background: {ACCENT_DIM}; color: {ACCENT}; border-color: {ACCENT}; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 10px; }}
QComboBox:hover, QLineEdit:hover {{ border-color: #4A5164; }}
QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM}; selection-color: {ACCENT}; outline: none; }}
QTextBrowser, QListWidget, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: transparent; border: none; }}
QListWidget::item {{ padding: 8px 6px; border-radius: 8px; }}
QListWidget::item:hover {{ background: {SURFACE_2}; }}
QListWidget::item:selected {{ background: {ACCENT_DIM}; color: {ACCENT}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #4A5164;
    background: {SURFACE_2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2E3342; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #3B4255; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QSplitter::handle {{ background: transparent; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 14px; border: none;
    border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
"""
