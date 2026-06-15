from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


TAB_COLORS = {
    "Checkpoint History": "#2a9d8f",
    "Task History": "#577590",
    "Task Editor": "#e76f51",
    "Training": "#3a86ff",
    "Evaluation": "#8a5cf6",
    "Episode Inspector": "#f4a261",
    "Unknown": "#94a3b8",
}

VIEW_CLASS_TO_TAB = {
    "CheckpointHistoryView": "Checkpoint History",
    "TaskHistoryView": "Task History",
    "TaskEditorView": "Task Editor",
    "FrozenLakeTaskEditorWidget": "Task Editor",
    "BlackjackTaskEditorWidget": "Task Editor",
    "TrainingMonitorView": "Training",
    "EvaluationView": "Evaluation",
    "EpisodeInspectorView": "Episode Inspector",
}

EVENT_LANES = [
    "Navigation",
    "Buttons",
    "Training",
    "History/Eval",
    "Session/File",
    "Other",
]


@dataclass(slots=True)
class InteractionEvent:
    timestamp: datetime
    session_id: str
    event_type: str
    tab_title: str
    label: str
    lane: str
    record: dict[str, Any]


@dataclass(slots=True)
class TabSegment:
    tab_title: str
    start: datetime
    end: datetime


@dataclass(slots=True)
class SessionTimeline:
    session_id: str
    events: list[InteractionEvent]
    tab_segments: list[TabSegment]

    @property
    def start(self) -> datetime:
        return self.events[0].timestamp

    @property
    def end(self) -> datetime:
        return self.events[-1].timestamp

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds())


def load_timelines(path: Path | str) -> list[SessionTimeline]:
    records = _load_jsonl_records(Path(path))
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        timestamp = _parse_timestamp(record.get("timestamp"))
        if timestamp is None:
            continue
        session_id = str(record.get("session_id") or "unknown")
        normalized = dict(record)
        normalized["_parsed_timestamp"] = timestamp
        by_session[session_id].append(normalized)

    timelines = [
        _build_timeline(session_id, session_records)
        for session_id, session_records in by_session.items()
        if session_records
    ]
    return sorted(timelines, key=lambda timeline: timeline.start)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payload["_line_number"] = line_number
                records.append(payload)
    return records


def _build_timeline(session_id: str, records: list[dict[str, Any]]) -> SessionTimeline:
    records = sorted(records, key=lambda record: record["_parsed_timestamp"])
    active_tab = "Unknown"
    segment_start = records[0]["_parsed_timestamp"]
    tab_segments: list[TabSegment] = []
    events: list[InteractionEvent] = []

    for record in records:
        timestamp = record["_parsed_timestamp"]
        event_type = str(record.get("event", "unknown"))
        if event_type == "view_changed":
            title = str(record.get("title") or "Unknown")
            if timestamp > segment_start:
                tab_segments.append(TabSegment(active_tab, segment_start, timestamp))
            active_tab = title or "Unknown"
            segment_start = timestamp

        tab_title = _tab_title_for_record(record, current_tab=active_tab)
        events.append(
            InteractionEvent(
                timestamp=timestamp,
                session_id=session_id,
                event_type=event_type,
                tab_title=tab_title,
                label=_event_label(record),
                lane=_event_lane(event_type),
                record={key: value for key, value in record.items() if not key.startswith("_")},
            )
        )

    final_end = records[-1]["_parsed_timestamp"]
    if final_end >= segment_start:
        tab_segments.append(TabSegment(active_tab, segment_start, final_end))

    return SessionTimeline(session_id=session_id, events=events, tab_segments=tab_segments)


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _tab_title_for_record(record: dict[str, Any], *, current_tab: str) -> str:
    if record.get("event") == "view_changed":
        return str(record.get("title") or "Unknown")

    widget = record.get("widget")
    if isinstance(widget, dict):
        path = widget.get("path")
        if isinstance(path, list):
            for item in reversed(path):
                item_text = str(item)
                class_name = item_text.split("#", 1)[0].split("[", 1)[0]
                if class_name in VIEW_CLASS_TO_TAB:
                    return VIEW_CLASS_TO_TAB[class_name]
    return current_tab or "Unknown"


def _event_label(record: dict[str, Any]) -> str:
    event = str(record.get("event", "unknown"))
    if event == "view_changed":
        return f"View: {record.get('title', 'Unknown')}"
    if event == "button_clicked":
        widget = record.get("widget")
        text = ""
        if isinstance(widget, dict):
            text = str(widget.get("text") or "")
        return f"Button: {text or 'unknown'}"
    if event == "training_started":
        algorithm = record.get("algorithm", "unknown")
        tasks = record.get("tasks")
        task_text = ""
        if isinstance(tasks, list) and tasks:
            first_task = tasks[0]
            if isinstance(first_task, dict):
                task_text = f" on {first_task.get('name', 'task')}"
        return f"Training started: {algorithm}{task_text}"
    if event == "training_status_changed":
        return f"Training status: {record.get('status', 'unknown')}"
    if event == "breakpoint_triggered":
        message = str(record.get("message") or "breakpoint")
        return f"Breakpoint: {message}"
    if event == "checkpoint_evaluated":
        return f"Checkpoint evaluated: {record.get('checkpoint_id', 'unknown')}"
    if event == "training_config_loaded_from_history":
        return "Loaded config from checkpoint history"
    if event == "curriculum_import_started":
        return f"Curriculum import: {record.get('step_count', 'unknown')} step(s)"
    if event == "project_saved":
        return "Project saved"
    if event == "application_started":
        return f"Application started: {record.get('environment_id', 'unknown')}"
    if event == "session_started":
        return "Session started"
    if event == "session_finished":
        return "Session finished"
    return event.replace("_", " ").title()


def _event_lane(event_type: str) -> str:
    if event_type == "view_changed":
        return "Navigation"
    if event_type == "button_clicked":
        return "Buttons"
    if event_type in {
        "training_started",
        "training_status_changed",
        "breakpoint_triggered",
    }:
        return "Training"
    if event_type in {
        "checkpoint_evaluated",
        "training_config_loaded_from_history",
        "curriculum_import_started",
        "curriculum_import_completed",
    }:
        return "History/Eval"
    if event_type in {"session_started", "application_started", "project_saved", "session_finished"}:
        return "Session/File"
    return "Other"


def _format_elapsed(seconds: float) -> str:
    seconds_int = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


class TimelineWidget(QWidget):
    event_selected = Signal(object)

    def __init__(self, timeline: SessionTimeline, *, pixels_per_second: float = 2.0) -> None:
        super().__init__()
        self._timeline = timeline
        self._pixels_per_second = pixels_per_second
        self._selected_event_index: int | None = None
        self._marker_rects: list[tuple[QRectF, int]] = []
        self.setMouseTracking(True)
        self._update_size()

    @property
    def current_timeline(self) -> SessionTimeline:
        return self._timeline

    def set_timeline(self, timeline: SessionTimeline) -> None:
        self._timeline = timeline
        self._selected_event_index = None
        self._update_size()
        self.update()

    def set_pixels_per_second(self, value: float) -> None:
        self._pixels_per_second = max(0.1, min(40.0, value))
        self._update_size()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        left = 140
        right = 32
        top = 42
        segment_height = 32
        lane_top = top + 76
        lane_height = 42
        axis_y = top + segment_height + 22
        timeline_width = max(1, self.width() - left - right)

        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Sans Serif", 10, QFont.Weight.Bold))
        painter.drawText(QRect(16, 8, self.width() - 32, 24), Qt.AlignmentFlag.AlignLeft, self._title_text())

        self._draw_tab_segments(painter, left, top, timeline_width, segment_height)
        self._draw_axis(painter, left, axis_y, timeline_width)
        self._draw_lanes(painter, left, lane_top, timeline_width, lane_height)
        self._draw_events(painter, left, lane_top, lane_height)

    def mousePressEvent(self, event) -> None:
        position = event.position()
        for rect, index in reversed(self._marker_rects):
            if rect.contains(position):
                self._selected_event_index = index
                self.event_selected.emit(self._timeline.events[index])
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        position = event.position()
        for rect, index in reversed(self._marker_rects):
            if rect.contains(position):
                timeline_event = self._timeline.events[index]
                elapsed = (timeline_event.timestamp - self._timeline.start).total_seconds()
                self.setToolTip(f"{_format_elapsed(elapsed)} | {timeline_event.label}")
                return
        self.setToolTip("")
        super().mouseMoveEvent(event)

    def _update_size(self) -> None:
        duration = max(self._timeline.duration_seconds, 1.0)
        width = 172 + max(900, min(24000, int(duration * self._pixels_per_second)))
        height = 42 + 76 + len(EVENT_LANES) * 42 + 48
        self.setMinimumSize(width, height)
        self.resize(width, height)

    def _title_text(self) -> str:
        duration = _format_elapsed(self._timeline.duration_seconds)
        return f"Session {self._timeline.session_id} | {len(self._timeline.events)} events | duration {duration}"

    def _x_for_time(self, timestamp: datetime, *, left: int) -> float:
        elapsed = max(0.0, (timestamp - self._timeline.start).total_seconds())
        return left + elapsed * self._pixels_per_second

    def _draw_tab_segments(
        self,
        painter: QPainter,
        left: int,
        top: int,
        timeline_width: int,
        height: int,
    ) -> None:
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.setBrush(QColor("#f8fafc"))
        painter.drawRoundedRect(QRectF(left, top, timeline_width, height), 4, 4)
        painter.setPen(QColor("#334155"))
        painter.drawText(QRectF(16, top, left - 28, height), Qt.AlignmentFlag.AlignVCenter, "Active tab")

        for segment in self._timeline.tab_segments:
            start_x = self._x_for_time(segment.start, left=left)
            end_x = max(start_x + 2, self._x_for_time(segment.end, left=left))
            rect = QRectF(start_x, top, end_x - start_x, height)
            color = QColor(TAB_COLORS.get(segment.tab_title, TAB_COLORS["Unknown"]))
            painter.fillRect(rect, color)
            if rect.width() > 52:
                painter.setPen(QColor("#ffffff"))
                painter.drawText(rect.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter, segment.tab_title)

    def _draw_axis(self, painter: QPainter, left: int, y: int, timeline_width: int) -> None:
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.drawLine(left, y, left + timeline_width, y)
        duration = max(self._timeline.duration_seconds, 1.0)
        tick_count = min(10, max(2, int(timeline_width / 180)))
        for index in range(tick_count + 1):
            ratio = index / tick_count
            x = left + ratio * timeline_width
            seconds = duration * ratio
            painter.drawLine(QPointF(x, y - 5), QPointF(x, y + 5))
            painter.setPen(QColor("#475569"))
            painter.drawText(QRectF(x - 32, y + 8, 64, 18), Qt.AlignmentFlag.AlignCenter, _format_elapsed(seconds))
            painter.setPen(QPen(QColor("#94a3b8"), 1))

    def _draw_lanes(
        self,
        painter: QPainter,
        left: int,
        top: int,
        timeline_width: int,
        lane_height: int,
    ) -> None:
        for index, lane in enumerate(EVENT_LANES):
            y = top + index * lane_height
            fill = QColor("#f8fafc" if index % 2 == 0 else "#ffffff")
            painter.fillRect(QRectF(left, y, timeline_width, lane_height), fill)
            painter.setPen(QColor("#e2e8f0"))
            painter.drawLine(left, y + lane_height, left + timeline_width, y + lane_height)
            painter.setPen(QColor("#334155"))
            painter.drawText(QRectF(16, y, left - 28, lane_height), Qt.AlignmentFlag.AlignVCenter, lane)

    def _draw_events(self, painter: QPainter, left: int, lane_top: int, lane_height: int) -> None:
        self._marker_rects = []
        for index, event in enumerate(self._timeline.events):
            lane_index = EVENT_LANES.index(event.lane) if event.lane in EVENT_LANES else len(EVENT_LANES) - 1
            x = self._x_for_time(event.timestamp, left=left)
            y = lane_top + lane_index * lane_height + lane_height / 2.0
            color = QColor(TAB_COLORS.get(event.tab_title, TAB_COLORS["Unknown"]))
            selected = index == self._selected_event_index
            rect = QRectF(x - 7, y - 7, 14, 14)
            self._marker_rects.append((rect.adjusted(-4, -4, 4, 4), index))
            self._draw_event_marker(painter, event.event_type, QPointF(x, y), color, selected=selected)

            if selected:
                painter.setPen(QColor("#0f172a"))
                painter.drawText(QRectF(x + 10, y - 11, 240, 22), Qt.AlignmentFlag.AlignVCenter, event.label)

    def _draw_event_marker(
        self,
        painter: QPainter,
        event_type: str,
        center: QPointF,
        color: QColor,
        *,
        selected: bool,
    ) -> None:
        pen_color = QColor("#0f172a") if selected else QColor("#1e293b")
        painter.setPen(QPen(pen_color, 2 if selected else 1))
        painter.setBrush(color)
        x = center.x()
        y = center.y()
        if event_type == "button_clicked":
            painter.drawEllipse(center, 6, 6)
        elif event_type == "training_started":
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x, y - 8),
                        QPointF(x + 8, y + 7),
                        QPointF(x - 8, y + 7),
                    ]
                )
            )
        elif event_type == "training_status_changed":
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x, y - 8),
                        QPointF(x + 8, y),
                        QPointF(x, y + 8),
                        QPointF(x - 8, y),
                    ]
                )
            )
        elif event_type == "breakpoint_triggered":
            painter.drawLine(QPointF(x - 7, y - 7), QPointF(x + 7, y + 7))
            painter.drawLine(QPointF(x + 7, y - 7), QPointF(x - 7, y + 7))
            painter.drawEllipse(center, 5, 5)
        elif event_type in {"session_started", "session_finished", "application_started", "project_saved"}:
            painter.drawRect(QRectF(x - 6, y - 6, 12, 12))
        elif event_type == "view_changed":
            painter.drawLine(QPointF(x, y - 10), QPointF(x, y + 10))
        else:
            painter.drawRoundedRect(QRectF(x - 6, y - 6, 12, 12), 3, 3)


class InteractionTimelineWindow(QMainWindow):
    def __init__(
        self,
        *,
        log_path: Path,
        timelines: list[SessionTimeline],
        initial_session_id: str | None = None,
        pixels_per_second: float = 2.0,
    ) -> None:
        super().__init__()
        self._timelines = timelines
        self._pixels_per_second = pixels_per_second
        self.setWindowTitle("Interaction Timeline")

        selected_timeline = self._initial_timeline(initial_session_id)
        self.timeline_widget = TimelineWidget(selected_timeline, pixels_per_second=pixels_per_second)
        self.timeline_widget.event_selected.connect(self._show_event_details)

        self.session_combo = QComboBox(self)
        for timeline in timelines:
            self.session_combo.addItem(self._session_label(timeline), timeline.session_id)
        self.session_combo.setCurrentIndex(timelines.index(selected_timeline))
        self.session_combo.currentIndexChanged.connect(self._on_session_changed)

        zoom_in = QPushButton("+", self)
        zoom_out = QPushButton("-", self)
        zoom_in.setToolTip("Zoom in")
        zoom_out.setToolTip("Zoom out")
        zoom_in.clicked.connect(lambda: self._set_zoom(self._pixels_per_second * 1.35))
        zoom_out.clicked.connect(lambda: self._set_zoom(self._pixels_per_second / 1.35))

        header = QHBoxLayout()
        header.addWidget(QLabel(f"Log: {log_path}", self), 1)
        header.addWidget(QLabel("Session", self))
        header.addWidget(self.session_combo)
        header.addWidget(QLabel("Zoom", self))
        header.addWidget(zoom_out)
        header.addWidget(zoom_in)

        legend = QLabel(self._legend_html(), self)
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setWordWrap(True)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.timeline_widget)

        self.details = QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(160)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        timeline_panel = QWidget(splitter)
        timeline_layout = QVBoxLayout(timeline_panel)
        timeline_layout.addLayout(header)
        timeline_layout.addWidget(legend)
        timeline_layout.addWidget(scroll, 1)
        splitter.addWidget(timeline_panel)
        splitter.addWidget(self.details)
        splitter.setSizes([620, 220])

        self.setCentralWidget(splitter)
        self.resize(1280, 820)
        if selected_timeline.events:
            self._show_event_details(selected_timeline.events[0])

    def _initial_timeline(self, session_id: str | None) -> SessionTimeline:
        if session_id is not None:
            for timeline in self._timelines:
                if timeline.session_id == session_id:
                    return timeline
        return self._timelines[-1]

    def _on_session_changed(self, index: int) -> None:
        session_id = self.session_combo.itemData(index)
        for timeline in self._timelines:
            if timeline.session_id == session_id:
                self.timeline_widget.set_timeline(timeline)
                self._show_event_details(timeline.events[0])
                return

    def _set_zoom(self, value: float) -> None:
        self._pixels_per_second = max(0.1, min(40.0, value))
        self.timeline_widget.set_pixels_per_second(self._pixels_per_second)

    def _show_event_details(self, event: InteractionEvent) -> None:
        elapsed = (event.timestamp - self.timeline_widget.current_timeline.start).total_seconds()
        header = {
            "elapsed": _format_elapsed(elapsed),
            "event": event.event_type,
            "tab": event.tab_title,
            "lane": event.lane,
            "label": event.label,
        }
        text = json.dumps(header, indent=2, sort_keys=True)
        text += "\n\nraw record:\n"
        text += json.dumps(event.record, indent=2, sort_keys=True)
        self.details.setPlainText(text)

    def _session_label(self, timeline: SessionTimeline) -> str:
        started = timeline.start.strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"{started} | {timeline.session_id[:10]} | {len(timeline.events)} events"

    def _legend_html(self) -> str:
        tab_items = []
        for title, color in TAB_COLORS.items():
            if title == "Unknown":
                continue
            tab_items.append(
                f"<span style='background-color:{color};'>&nbsp;&nbsp;&nbsp;</span> {title}"
            )
        return (
            "<b>Tab colors:</b> "
            + " &nbsp; ".join(tab_items)
            + "<br><b>Event markers:</b> line=view, circle=button, triangle=training start, "
            "diamond=status, crossed circle=breakpoint, square=session/file"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a Qt timeline for an RL Editor interaction log.")
    parser.add_argument("interaction_log", type=Path, help="JSON Lines interaction log path.")
    parser.add_argument("--session", help="Session id to show initially. Defaults to the latest session.")
    parser.add_argument(
        "--zoom",
        type=float,
        default=2.0,
        help="Horizontal pixels per second. Defaults to 2.0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        timelines = load_timelines(args.interaction_log)
    except OSError as exc:
        parser.error(f"Could not read interaction log: {exc}")

    if not timelines:
        parser.error("No valid interaction events with timestamps were found.")

    if args.session is not None and args.session not in {timeline.session_id for timeline in timelines}:
        parser.error(f"Session id not found: {args.session}")

    app = QApplication([sys.argv[0], str(args.interaction_log)])
    window = InteractionTimelineWindow(
        log_path=args.interaction_log,
        timelines=timelines,
        initial_session_id=args.session,
        pixels_per_second=args.zoom,
    )
    window.show()
    try:
        return app.exec()
    except Exception as exc:
        QMessageBox.critical(window, "Interaction Timeline", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
