from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QTabWidget, QWidget
from shiboken6 import delete

from rleditor.app import _build_parser
from rleditor.ui.interaction_logging import InteractionLogger


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_cli_parser_accepts_interaction_log_path(tmp_path: Path) -> None:
    log_path = tmp_path / "interactions.jsonl"
    parser = _build_parser(["frozen_lake"])

    args = parser.parse_args(["--env", "frozen_lake", "--interaction-log", str(log_path)])

    assert args.interaction_log == log_path


def test_interaction_logger_records_button_clicks_view_changes_and_domain_events(tmp_path: Path) -> None:
    app = _app()
    log_path = tmp_path / "logs" / "interactions.jsonl"
    window = QMainWindow()
    tabs = QTabWidget(window)
    tabs.setObjectName("main_tabs")
    first_tab = QWidget(tabs)
    second_tab = QWidget(tabs)
    tabs.addTab(first_tab, "First")
    tabs.addTab(second_tab, "Second")
    button = QPushButton("Start Training", first_tab)
    button.setObjectName("start_training_button")
    window.setCentralWidget(tabs)
    logger = InteractionLogger(log_path)

    try:
        logger.attach(app, root_widget=window)
        button.click()
        tabs.setCurrentIndex(1)
        logger.log("training_started", algorithm="q_learning", max_steps=10)
    finally:
        logger.close()

    records = _records(log_path)
    assert {record["event"] for record in records} >= {
        "session_started",
        "button_clicked",
        "view_changed",
        "training_started",
        "session_finished",
    }
    assert any(
        record["event"] == "button_clicked"
        and isinstance(record.get("widget"), dict)
        and record["widget"]["text"] == "Start Training"
        and record["widget"]["object_name"] == "start_training_button"
        for record in records
    )
    assert any(
        record["event"] == "view_changed"
        and record.get("title") == "Second"
        for record in records
    )
    assert any(
        record["event"] == "training_started"
        and record.get("algorithm") == "q_learning"
        and record.get("max_steps") == 10
        for record in records
    )


def test_interaction_logger_ignores_deleted_widget_during_deferred_discovery(tmp_path: Path) -> None:
    _app()
    log_path = tmp_path / "interactions.jsonl"
    logger = InteractionLogger(log_path)
    transient_widget = QLabel("Transient")

    delete(transient_widget)

    try:
        logger._safe_connect_widget_tree(transient_widget)
    finally:
        logger.close()

    assert _records(log_path)[0]["event"] == "session_started"
