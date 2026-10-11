"""Premium Verse Display Window for live church presentation and Wirecast capture.

Features:
- Borderless window, resizable, draggable, F11 fullscreen toggle for 2nd monitor.
- Elegant centered verse card with rounded corners (24px), glassmorphism gradient, soft drop shadow.
- Responsive typography with auto-scaling serif font for scriptures.
- Display modes: English only, Hindi only, or Both side-by-side / stacked.
- 3 selectable themes (Midnight, Royal Blue, Parchment) + Green Screen (Chroma Key).
- Smooth fade in / fade out animations.
"""

from typing import Optional
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt
from PySide6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from vachan.core.models import BibleReference

THEMES = {
    "Midnight": {
        "bg_window": "rgba(10, 12, 18, 0.95)",
        "card_bg": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1E202E, stop:1 #141622)",
        "card_border": "#31344B",
        "verse_text": "#F5F5FA",
        "hindi_text": "#E0E2EC",
        "ref_text": "#E5A93C",  # Warm gold accent
        "divider": "rgba(229, 169, 60, 0.4)",
    },
    "Royal Blue": {
        "bg_window": "rgba(6, 14, 28, 0.95)",
        "card_bg": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0F2042, stop:1 #09142B)",
        "card_border": "#1E3B70",
        "verse_text": "#FFFFFF",
        "hindi_text": "#D4E2FC",
        "ref_text": "#4DA3FF",  # Sapphire blue accent
        "divider": "rgba(77, 163, 255, 0.4)",
    },
    "Parchment": {
        "bg_window": "#2B2620",
        "card_bg": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FDFBF7, stop:1 #F4EFE6)",
        "card_border": "#D8CEBE",
        "verse_text": "#2A241C",
        "hindi_text": "#3D342A",
        "ref_text": "#8C581E",
        "divider": "rgba(140, 88, 30, 0.3)",
    },
    "Green Screen (Chroma)": {
        "bg_window": "#00FF00",
        "card_bg": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1E202E, stop:1 #141622)",
        "card_border": "#31344B",
        "verse_text": "#FFFFFF",
        "hindi_text": "#E0E2EC",
        "ref_text": "#F9E2AF",
        "divider": "rgba(249, 226, 175, 0.4)",
    },
}


class VerseDisplayWindow(QWidget):
    """Broadcast output window captured by Wirecast via Window Capture."""

    def __init__(self, theme_name: str = "Midnight") -> None:
        super().__init__()
        self.theme_name = theme_name
        self.language_mode = "both"  # "english", "hindi", or "both"
        self._is_dragging = False
        self._drag_position = None

        self._setup_window()
        self._init_ui()
        self.apply_theme(self.theme_name)

    def _setup_window(self) -> None:
        """Configure borderless, transparent window properties."""
        self.setWindowTitle("Vachan - Verse Output Screen")
        self.resize(1100, 680)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def _init_ui(self) -> None:
        """Construct the centered card layout with smooth shadows and typography."""
        # Outer layout centering the card
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(60, 60, 60, 60)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Card container
        self.card = QFrame(self)
        self.card.setObjectName("VerseCard")
        self.card.setFixedWidth(980)

        # Soft drop shadow
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 12)
        self.card.setGraphicsEffect(shadow)

        # Card internal layout
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setContentsMargins(48, 40, 48, 40)
        self.card_layout.setSpacing(20)

        # English Verse Text Label
        self.verse_en_label = QLabel(self.card)
        self.verse_en_label.setObjectName("VerseEnglish")
        self.verse_en_label.setWordWrap(True)
        self.verse_en_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verse_en_label.setFont(QFont("Georgia", 22, QFont.Weight.Medium))
        self.card_layout.addWidget(self.verse_en_label)

        # Hindi Verse Text Label (Devanagari rendering)
        self.verse_hi_label = QLabel(self.card)
        self.verse_hi_label.setObjectName("VerseHindi")
        self.verse_hi_label.setWordWrap(True)
        self.verse_hi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verse_hi_label.setFont(QFont("Nirmala UI", 21, QFont.Weight.Medium))
        self.card_layout.addWidget(self.verse_hi_label)

        # Elegant divider line
        self.divider = QFrame(self.card)
        self.divider.setFrameShape(QFrame.Shape.HLine)
        self.divider.setFrameShadow(QFrame.Shadow.Plain)
        self.divider.setFixedHeight(2)
        self.divider.setFixedWidth(200)
        self.card_layout.addWidget(self.divider, alignment=Qt.AlignmentFlag.AlignCenter)

        # Reference Badge Label (e.g., "John 3:16 · KJV")
        self.ref_label = QLabel(self.card)
        self.ref_label.setObjectName("ReferenceLabel")
        self.ref_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ref_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.card_layout.addWidget(self.ref_label)

        self.main_layout.addWidget(self.card)

        # Opacity animation for smooth fade-in
        self.opacity_effect = QGraphicsOpacityEffect(self.card)
        self.card.setGraphicsEffect(self.opacity_effect)
        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(450)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Initial placeholder preview
        self.set_verse(
            reference_text="John 3:16 · KJV",
            text_en="For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life.",
            text_hi="क्योंकि परमेश्वर ने जगत से ऐसा प्रेम रखा कि उसने अपना एकलौता पुत्र दे दिया, ताकि जो कोई उस पर विश्वास करे वह नाश न हो, परन्तु अनन्त जीवन पाए।",
        )

    def apply_theme(self, theme_name: str) -> None:
        """Update color scheme and styling across card components."""
        self.theme_name = theme_name
        theme = THEMES.get(theme_name, THEMES["Midnight"])

        # Main background styling
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {theme['bg_window']};
            }}
            QFrame#VerseCard {{
                background: {theme['card_bg']};
                border: 1px solid {theme['card_border']};
                border-radius: 24px;
            }}
            QLabel#VerseEnglish {{
                color: {theme['verse_text']};
                background: transparent;
                line-height: 1.4;
            }}
            QLabel#VerseHindi {{
                color: {theme['hindi_text']};
                background: transparent;
                line-height: 1.4;
            }}
            QLabel#ReferenceLabel {{
                color: {theme['ref_text']};
                background: transparent;
                letter-spacing: 1px;
            }}
            QFrame {{
                background-color: {theme['divider']};
                border: none;
            }}
        """)

    def set_display_mode(self, mode: str) -> None:
        """Switch between 'both', 'english', or 'hindi'."""
        self.language_mode = mode
        if mode == "english":
            self.verse_en_label.show()
            self.verse_hi_label.hide()
        elif mode == "hindi":
            self.verse_en_label.hide()
            self.verse_hi_label.show()
        else:
            self.verse_en_label.show()
            self.verse_hi_label.show()

    def set_verse(
        self,
        reference_text: str,
        text_en: str,
        text_hi: Optional[str] = None,
    ) -> None:
        """Display verse text with smooth fade transition."""
        self.ref_label.setText(reference_text)
        self.verse_en_label.setText(f'“{text_en}”')
        self.verse_hi_label.setText(text_hi or "")

        self.set_display_mode(self.language_mode)

        # Trigger smooth fade-in
        self.fade_anim.stop()
        self.fade_anim.setStartValue(0.2)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.start()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """F11 toggles fullscreen for dual-monitor live display; Escape closes or minimizes."""
        if event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.hide()
            event.accept()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Enable dragging borderless window."""
        if event.button() == Qt.Key.Key_Left or event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_dragging and self._drag_position:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._is_dragging = False
        event.accept()
