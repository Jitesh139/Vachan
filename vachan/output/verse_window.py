"""Broadcast output window (Wirecast Window Capture / second monitor).

Opaque themed background (translucent windows capture as white/black), one centered
card painted by hand: rounded corners, soft shadow, subtle gradient, serif text that
auto-fits any verse length, reference line below, fade + scale animation.
Keys: F11 / double-click fullscreen, Esc leaves fullscreen, right-click for sizes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Property, QEasingCurve, QPoint, QPropertyAnimation, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QMenu, QSizeGrip, QWidget

from vachan.core.models import Suggestion
from vachan.core.paths import DATA_DIR
from vachan.ui.theme import SERIF, SERIF_DEVA, UI_FONT


@dataclass(frozen=True)
class OutputTheme:
    """Colors for the broadcast card."""

    background: str
    card_top: str
    card_bottom: str
    border: QColor
    text: str
    text_secondary: str
    accent: str
    shadow: bool = True


THEMES = {
    "Midnight": OutputTheme("#07090F", "#1E2233", "#121521", QColor(255, 255, 255, 26),
                            "#F7F5F0", "#C9CDD9", "#E3B75C"),
    "Royal Blue": OutputTheme("#030A1A", "#173170", "#0A1838", QColor(150, 185, 255, 48),
                              "#FFFFFF", "#D2DEFF", "#9EC1FF"),
    "Parchment": OutputTheme("#2A231B", "#FBF6EC", "#EFE4CF", QColor(120, 80, 30, 60),
                             "#2B2118", "#4D3E2F", "#9A5B1E"),
    "Green Screen (Chroma)": OutputTheme("#00FF00", "#1E2233", "#121521", QColor(255, 255, 255, 26),
                                         "#F7F5F0", "#C9CDD9", "#E3B75C", shadow=False),
}  # fmt: skip
VERSION_LABEL = {"HIN_IRV": "IRV हिंदी"}


def _hindi_book_names() -> dict[str, str]:
    books = json.loads((DATA_DIR / "books.json").read_text(encoding="utf-8"))
    return {b["id"]: b["hindi_name"] for b in books}


class VerseDisplayWindow(QWidget):
    """Implements the OutputSink port."""

    def __init__(self, theme_name: str = "Midnight", display_mode: str = "both") -> None:
        super().__init__()
        self.setWindowTitle("Vachan - Verse Output")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMinimumSize(480, 270)
        self.resize(1280, 720)
        self.theme = THEMES.get(theme_name, THEMES["Midnight"])
        self.display_mode = display_mode
        self.current: Optional[Suggestion] = None
        self._hindi_names = _hindi_book_names()
        self._progress = 0.0
        self._drag: Optional[QPoint] = None
        self._anim = QPropertyAnimation(self, b"progress", self)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._grip = QSizeGrip(self)
        self._grip.resize(18, 18)

    # ------------------------------------------------------------------ OutputSink
    def show_suggestion(self, suggestion: Suggestion) -> None:
        """Put a verse on air with a fade/scale-in."""
        self.current = suggestion
        self._animate(0.0, 1.0, 520)

    def clear(self) -> None:
        """Fade the card out, leaving only the background."""
        self._animate(self._progress, 0.0, 300)

    def set_theme(self, name: str) -> None:
        """Switch output theme."""
        self.theme = THEMES.get(name, THEMES["Midnight"])
        self.update()

    def set_display_mode(self, mode: str) -> None:
        """'both', 'primary' or 'secondary'."""
        self.display_mode = mode
        self.update()

    # ------------------------------------------------------------------ animation
    def _get_progress(self) -> float:
        return self._progress

    def _set_progress(self, value: float) -> None:
        self._progress = value
        self.update()

    progress = Property(float, _get_progress, _set_progress)

    def _animate(self, start: float, end: float, ms: int) -> None:
        self._anim.stop()
        self._anim.setDuration(ms)
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.start()

    # ------------------------------------------------------------------ painting
    def _blocks(self) -> list[tuple[str, str, str]]:
        """(text, font family, color) blocks for the current display mode."""
        s, t = self.current, self.theme
        blocks = []
        if self.display_mode in ("both", "primary") or not s.secondary:
            family = SERIF_DEVA if s.primary.version.startswith("HIN") else SERIF
            blocks.append((s.primary.text, family, t.text))
        if s.secondary and self.display_mode in ("both", "secondary"):
            family = SERIF_DEVA if s.secondary.version.startswith("HIN") else SERIF
            color = t.text_secondary if blocks else t.text
            blocks.append((s.secondary.text, family, color))
        return blocks

    def _reference_line(self) -> str:
        s = self.current
        ref = s.reference
        versions = [s.primary.version]
        if s.secondary and self.display_mode in ("both", "secondary"):
            versions = [s.secondary.version] if self.display_mode == "secondary" else versions + [s.secondary.version]
        if self.display_mode == "secondary" and s.secondary and s.secondary.version.startswith("HIN"):
            name = f"{self._hindi_names.get(ref.book_id, ref.book_name)} {ref.format_reference().split(' ')[-1]}"
        else:
            name = ref.format_reference()
        return "  ·  ".join([name] + [VERSION_LABEL.get(v, v) for v in versions])

    def _fit(self, blocks, width: float, max_height: float) -> tuple[float, list[float]]:
        """Largest font size (px) at which all blocks + reference fit."""
        lo, hi = 14.0, max(16.0, self.height() * 0.085)
        heights: list[float] = []
        for _ in range(18):
            mid = (lo + hi) / 2
            hs = [self._text_height(text, fam, mid, width) for text, fam, _ in blocks]
            total = sum(hs) + mid * 0.9 * (len(blocks) - 1) + mid * 2.2
            if total <= max_height:
                lo, heights = mid, hs
            else:
                hi = mid
        if not heights:
            heights = [self._text_height(text, fam, lo, width) for text, fam, _ in blocks]
        return lo, heights

    @staticmethod
    def _font(family: str, px: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
        font = QFont(family)
        font.setPixelSize(max(1, round(px)))
        font.setWeight(weight)
        return font

    def _text_height(self, text: str, family: str, px: float, width: float) -> float:
        fm = QFontMetricsF(self._font(family, px))
        flags = int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignHCenter)
        return fm.boundingRect(QRectF(0, 0, width, 1e6), flags, text).height()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Background, then the animated card."""
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        t = self.theme
        p.fillRect(self.rect(), QColor(t.background))
        if self.current is None or self._progress <= 0.001:
            return
        w, h = self.width(), self.height()
        card_w = min(w * 0.86, h * 2.1)
        pad = max(28.0, card_w * 0.055)
        inner_w = card_w - 2 * pad
        blocks = self._blocks()
        px, heights = self._fit(blocks, inner_w, h * 0.80 - 2 * pad)
        ref_px = max(13.0, px * 0.52)
        content_h = sum(heights) + px * 0.9 * (len(blocks) - 1) + px * 2.2
        card_h = content_h + 2 * pad
        card = QRectF((w - card_w) / 2, (h - card_h) / 2, card_w, card_h)

        p.setOpacity(self._progress)
        scale = 0.965 + 0.035 * self._progress
        p.translate(card.center())
        p.scale(scale, scale)
        p.translate(-card.center())

        radius = 24.0
        if t.shadow:
            for i in range(14, 0, -1):  # soft layered shadow
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, int(9 * (15 - i) / 14 * 1.6)))
                p.drawRoundedRect(card.adjusted(-i, -i + 10, i, i + 10), radius + i, radius + i)
        path = QPainterPath()
        path.addRoundedRect(card, radius, radius)
        grad = QLinearGradient(card.topLeft(), card.bottomLeft())
        grad.setColorAt(0, QColor(t.card_top))
        grad.setColorAt(1, QColor(t.card_bottom))
        p.fillPath(path, grad)
        p.setPen(QPen(t.border, 1.2))
        p.drawPath(path)
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))  # glass highlight
        p.drawLine(card.left() + radius, card.top() + 1.5, card.right() - radius, card.top() + 1.5)

        y = card.top() + pad
        flags = int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignHCenter)
        for (text, family, color), bh in zip(blocks, heights):
            p.setFont(self._font(family, px))
            p.setPen(QColor(color))
            p.drawText(QRectF(card.left() + pad, y, inner_w, bh + 4), flags, text)
            y += bh + px * 0.9
        y = card.bottom() - pad - ref_px * 1.6
        p.setPen(QPen(QColor(t.accent), 2))
        p.drawLine(card.center().x() - 32, y - px * 0.35, card.center().x() + 32, y - px * 0.35)
        ref_font = self._font(UI_FONT, ref_px, QFont.Weight.DemiBold)
        ref_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 104)
        p.setFont(ref_font)
        p.setPen(QColor(t.accent))
        p.drawText(QRectF(card.left(), y, card_w, ref_px * 1.8), int(Qt.AlignmentFlag.AlignHCenter), self._reference_line())

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Keep the resize grip in the corner."""
        self._grip.move(self.width() - self._grip.width(), self.height() - self._grip.height())
        super().resizeEvent(event)

    # ------------------------------------------------------------------ window handling
    def toggle_fullscreen(self) -> None:
        """Fullscreen on the monitor the window is currently on."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self._grip.setVisible(not self.isFullScreen())

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """F11 toggles fullscreen; Esc only leaves fullscreen (never blanks the stream)."""
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Double-click toggles fullscreen."""
        self.toggle_fullscreen()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Drag the frameless window."""
        if event.button() == Qt.MouseButton.LeftButton and not self.isFullScreen():
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Move while dragging."""
        if self._drag is not None:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """End drag."""
        self._drag = None

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        """Right-click menu for sizes (the window has no title bar)."""
        menu = QMenu(self)
        menu.addAction("Fullscreen (F11)", self.toggle_fullscreen)
        for w, h in ((1280, 720), (1920, 1080)):
            menu.addAction(f"Window {w} x {h}", lambda w=w, h=h: (self.showNormal(), self.resize(w, h)))
        menu.addSeparator()
        menu.addAction("Hide output window", self.hide)
        menu.exec(event.globalPos())
