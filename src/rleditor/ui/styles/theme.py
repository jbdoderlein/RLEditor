from __future__ import annotations

from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    """Apply a neutral, high-contrast theme suitable for long training sessions."""

    app.setStyleSheet(
        """
        QWidget {
            background-color: #f2f4f8;
            color: #1f2430;
            font-family: 'Noto Sans', 'DejaVu Sans', sans-serif;
            font-size: 13px;
        }

        QMainWindow, QFrame#MainSurface {
            background-color: #f6f8fb;
        }

        QGroupBox {
            border: 1px solid #ccd3df;
            border-radius: 8px;
            margin-top: 10px;
            padding: 10px;
            background: #ffffff;
            font-weight: 600;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }

        QPushButton {
            background-color: #e7edf7;
            border: 1px solid #b8c4d8;
            border-radius: 6px;
            padding: 6px 10px;
            min-height: 20px;
        }

        QPushButton:hover {
            background-color: #dbe6f5;
        }

        QPushButton:pressed {
            background-color: #cfdcf1;
        }

        QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox, QTextEdit {
            background-color: #ffffff;
            border: 1px solid #c3cada;
            border-radius: 6px;
            padding: 4px 6px;
        }

        QTabWidget::pane {
            border: 1px solid #cfd6e4;
            background: #ffffff;
            border-radius: 8px;
            top: -1px;
        }

        QTabBar::tab {
            background: #eaf0f9;
            border: 1px solid #cad4e5;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 6px 12px;
            margin-right: 2px;
        }

        QTabBar::tab:selected {
            background: #ffffff;
            border-bottom-color: #ffffff;
        }

        QLabel#TitleLabel {
            font-size: 20px;
            font-weight: 700;
            color: #13213a;
        }

        QLabel#SubtitleLabel {
            color: #4b5b78;
            font-size: 12px;
        }

        QFrame#MetricCardFrame {
            border: 1px solid #d5dceb;
            border-radius: 8px;
            background: #ffffff;
        }

        QLabel#MetricTitleLabel {
            font-size: 11px;
            color: #5a6a88;
            font-weight: 600;
        }

        QLabel#MetricValueLabel {
            font-size: 16px;
            color: #17243c;
            font-weight: 700;
        }
        """
    )
