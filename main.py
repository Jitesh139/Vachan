"""Vachan - Windows desktop app for live church streaming.

Main application entrypoint.
"""

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    """Main window placeholder for Vachan operator dashboard."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vachan")
        self.resize(800, 500)

        # Central widget container
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Temporary placeholder label
        label = QLabel("Vachan\nLive Bible Verse Detection System", self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 20px; font-weight: bold; color: #555;")
        layout.addWidget(label)

        self.setCentralWidget(central_widget)


def main() -> int:
    """Application main function."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
