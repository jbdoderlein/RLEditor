from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon


def application_icon() -> QIcon:
    for icon_path in (
        Path(__file__).resolve().parents[3] / "icon.png",
        Path.cwd() / "icon.png",
    ):
        if icon_path.exists():
            return QIcon(str(icon_path))
    return QIcon()
