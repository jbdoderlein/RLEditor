from __future__ import annotations

import json

from rleditor.tools.interaction_timeline import load_timelines


def test_interaction_timeline_groups_events_and_infers_tabs(tmp_path) -> None:
    log_path = tmp_path / "interaction.jsonl"
    records = [
        {
            "timestamp": "2026-05-27T10:00:00.000+02:00",
            "session_id": "session_a",
            "event": "session_started",
        },
        {
            "timestamp": "2026-05-27T10:00:02.000+02:00",
            "session_id": "session_a",
            "event": "view_changed",
            "title": "Task History",
        },
        {
            "timestamp": "2026-05-27T10:00:05.000+02:00",
            "session_id": "session_a",
            "event": "button_clicked",
            "widget": {
                "text": "Copy Task",
                "path": ["MainWindow", "QTabWidget", "TaskHistoryView", "QPushButton[Copy Task]"],
            },
        },
        {
            "timestamp": "2026-05-27T10:00:10.000+02:00",
            "session_id": "session_a",
            "event": "view_changed",
            "title": "Training",
        },
        {
            "timestamp": "2026-05-27T10:00:12.000+02:00",
            "session_id": "session_a",
            "event": "training_started",
            "algorithm": "q_learning",
            "tasks": [{"name": "Frozen Lake"}],
        },
        {
            "timestamp": "2026-05-27T10:00:15.000+02:00",
            "session_id": "session_a",
            "event": "breakpoint_triggered",
            "message": "Breakpoint hit",
        },
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    timelines = load_timelines(log_path)

    assert len(timelines) == 1
    timeline = timelines[0]
    assert timeline.session_id == "session_a"
    assert timeline.duration_seconds == 15.0
    assert [segment.tab_title for segment in timeline.tab_segments] == [
        "Unknown",
        "Task History",
        "Training",
    ]
    assert timeline.events[2].tab_title == "Task History"
    assert timeline.events[2].lane == "Buttons"
    assert timeline.events[2].label == "Button: Copy Task"
    assert timeline.events[4].tab_title == "Training"
    assert timeline.events[4].lane == "Training"
    assert timeline.events[4].label == "Training started: q_learning on Frozen Lake"


def test_interaction_timeline_uses_latest_session_last(tmp_path) -> None:
    log_path = tmp_path / "interaction.jsonl"
    records = [
        {
            "timestamp": "2026-05-27T10:00:00.000+02:00",
            "session_id": "older",
            "event": "session_started",
        },
        {
            "timestamp": "2026-05-27T11:00:00.000+02:00",
            "session_id": "newer",
            "event": "session_started",
        },
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    timelines = load_timelines(log_path)

    assert [timeline.session_id for timeline in timelines] == ["older", "newer"]
