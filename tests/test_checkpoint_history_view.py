from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.core.models import Checkpoint, TaskSnapshot
from rleditor.ui.views.checkpoint_history_view import CheckpointHistoryView


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_checkpoint_history_view_renders_checkpoint_details_as_html_tables() -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_003",
        label="Checkpoint 003",
        created_at="2026-04-28 11:23:39",
        reason="run_finished",
        parent_checkpoint_id="checkpoint_001",
        run_id="run_8c5c4ab6",
        task_name="FL Main Task Easy goal",
        step=100000,
        episode=3624,
        task_snapshot=TaskSnapshot(
            environment_id="frozen_lake",
            task_name="FL Main Task Easy goal",
        ),
        metadata={
            "training_metrics": {
                "mean_reward": -0.57,
                "success_rate": 0.2,
                "episode_reward_mean": -0.57,
                "episode_length_mean": 30.14,
                "exploration_rate": 0.02,
                "value_loss": 0.006029579930842521,
                "policy_loss": None,
            }
        },
    )

    view._set_checkpoint_details(checkpoint, heading="Checkpoint produced by this run")
    html = view.checkpoint_details.toHtml()

    assert "Checkpoint produced by this run" in html
    assert "Checkpoint ID" in html
    assert "Recorded training metrics" in html
    assert "Success rate" in html
    assert "20.0%" in html
