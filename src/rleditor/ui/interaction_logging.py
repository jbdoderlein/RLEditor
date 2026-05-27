from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QAbstractButton, QApplication, QTabWidget, QWidget
from shiboken6 import isValid


class InteractionLogger(QObject):
    """JSON-lines logger for UI and high-level experiment interactions."""

    def __init__(self, path: Path | str) -> None:
        super().__init__()
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._session_id = uuid4().hex
        self._connected_buttons: set[int] = set()
        self._connected_tabs: set[int] = set()
        self.log("session_started", log_path=str(self.path))

    def close(self) -> None:
        if self._file.closed:
            return
        self.log("session_finished")
        self._file.close()

    def attach(
        self,
        app: QApplication,
        *,
        root_widget: QWidget,
        training_service: object | None = None,
    ) -> None:
        app.installEventFilter(self)
        self._safe_connect_widget_tree(root_widget)
        status_changed = getattr(training_service, "status_changed", None)
        if status_changed is not None:
            status_changed.connect(self._on_training_status_changed)

    def eventFilter(self, watched: QObject, event: object) -> bool:
        event_type = getattr(event, "type", lambda: None)()
        if event_type == QEvent.Type.ChildAdded:
            child = getattr(event, "child", lambda: None)()
            if isinstance(child, QWidget):
                QTimer.singleShot(0, lambda child=child: self._safe_connect_widget_tree(child))
        return super().eventFilter(watched, event)

    def log(self, event: str, **payload: object) -> None:
        if self._file.closed:
            return
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "session_id": self._session_id,
            "event": event,
            **{key: self._to_json_value(value) for key, value in payload.items()},
        }
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def _safe_connect_widget_tree(self, widget: QWidget) -> None:
        if not self._is_valid_qobject(widget):
            return
        try:
            self._connect_widget_tree(widget)
        except RuntimeError as exc:
            if self._is_deleted_qobject_error(exc):
                return
            raise

    def _connect_widget_tree(self, widget: QWidget) -> None:
        if not self._is_valid_qobject(widget):
            return
        widgets = [widget, *widget.findChildren(QWidget)]
        for child in widgets:
            if not self._is_valid_qobject(child):
                continue
            if isinstance(child, QAbstractButton):
                self._connect_button(child)
            if isinstance(child, QTabWidget):
                self._connect_tabs(child)

    def _connect_button(self, button: QAbstractButton) -> None:
        if not self._is_valid_qobject(button):
            return
        object_id = id(button)
        if object_id in self._connected_buttons:
            return
        self._connected_buttons.add(object_id)
        try:
            button.clicked.connect(
                lambda checked=False, button=button: self._log_button_clicked(button, checked=checked)
            )
        except RuntimeError as exc:
            if self._is_deleted_qobject_error(exc):
                return
            raise

    def _connect_tabs(self, tabs: QTabWidget) -> None:
        if not self._is_valid_qobject(tabs):
            return
        object_id = id(tabs)
        if object_id in self._connected_tabs:
            return
        self._connected_tabs.add(object_id)
        try:
            tabs.currentChanged.connect(lambda index, tabs=tabs: self._log_tab_changed(tabs, index))
        except RuntimeError as exc:
            if self._is_deleted_qobject_error(exc):
                return
            raise

    def _log_button_clicked(self, button: QAbstractButton, *, checked: bool) -> None:
        if not self._is_valid_qobject(button):
            return
        self.log(
            "button_clicked",
            widget=self._widget_payload(button),
            checked=checked,
        )

    def _log_tab_changed(self, tabs: QTabWidget, index: int) -> None:
        if not self._is_valid_qobject(tabs):
            return
        self.log(
            "view_changed",
            widget=self._widget_payload(tabs),
            index=index,
            title=tabs.tabText(index) if 0 <= index < tabs.count() else "",
        )

    def _on_training_status_changed(self, status: object) -> None:
        self.log("training_status_changed", status=getattr(status, "value", str(status)))

    def _widget_payload(self, widget: QWidget) -> dict[str, object]:
        if not self._is_valid_qobject(widget):
            return {"class": widget.__class__.__name__, "deleted": True}
        payload: dict[str, object] = {
            "class": widget.__class__.__name__,
            "object_name": widget.objectName(),
            "path": self._widget_path(widget),
        }
        text = getattr(widget, "text", None)
        if callable(text):
            payload["text"] = text()
        tool_tip = widget.toolTip()
        if tool_tip:
            payload["tooltip"] = tool_tip
        return payload

    def _widget_path(self, widget: QWidget) -> list[str]:
        path: list[str] = []
        current: QObject | None = widget
        while current is not None:
            if not self._is_valid_qobject(current):
                break
            label = current.__class__.__name__
            object_name = current.objectName()
            if object_name:
                label = f"{label}#{object_name}"
            elif isinstance(current, QAbstractButton) and current.text():
                label = f"{label}[{current.text()}]"
            path.append(label)
            current = current.parent()
        return list(reversed(path))

    def _is_valid_qobject(self, value: QObject) -> bool:
        try:
            return isValid(value)
        except RuntimeError:
            return False

    def _is_deleted_qobject_error(self, exc: RuntimeError) -> bool:
        message = str(exc)
        return "already deleted" in message or "Internal C++ object" in message

    def _to_json_value(self, value: object) -> object:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, dict):
            return {str(key): self._to_json_value(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self._to_json_value(item) for item in value]
        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            try:
                return self._to_json_value(value.to_dict())
            except Exception:
                return repr(value)
        return repr(value)
