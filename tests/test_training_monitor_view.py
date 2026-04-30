from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.core.models import TrainingMetrics
from rleditor.ui.views.training_monitor_view import TrainingMonitorView


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_training_monitor_breakpoints_default_to_pause_and_checkpoint() -> None:
    _app()
    view = TrainingMonitorView()

    view.breakpoint_kind_combo.setCurrentIndex(0)
    view.breakpoint_value_spin.setValue(3.0)
    view._add_breakpoint_rule()

    config = view.build_config()

    assert len(config.breakpoints) == 1
    assert config.breakpoints[0].actions == ["pause", "checkpoint"]
    assert "actions=pause+checkpoint" in view.breakpoint_list.item(0).text()


def test_training_monitor_can_build_unlimited_step_config() -> None:
    _app()
    view = TrainingMonitorView()

    view.total_steps_spin.setValue(-1)
    config = view.build_config()

    assert config.max_steps is None


def test_training_monitor_uses_selected_task_algorithm_hint() -> None:
    _app()
    view = TrainingMonitorView()

    view.set_algorithm_hint("random")
    config = view.build_config()

    assert config.algorithm == "random"
    assert view.algorithm_label.text() == "random"


def test_training_monitor_surfaces_classic_q_learning_metrics() -> None:
    _app()
    view = TrainingMonitorView()

    view.set_metrics(
        TrainingMetrics(
            step=128,
            episode=9,
            episode_reward_mean=0.625,
            success_rate=0.5,
            episode_length_mean=11.5,
            exploration_rate=0.08,
            value_loss=0.143,
            fps=91.2,
        )
    )

    assert "steps=128" in view.metrics_label.text()
    assert "episodes=9" in view.metrics_label.text()
    assert "return_mean=0.625" in view.metrics_label.text()
    assert "success=50.0%" in view.metrics_label.text()
    assert "epsilon=8.0%" in view.metrics_label.text()
    assert "td_error=0.143" in view.metrics_label.text()
    assert set(view._metric_cards) == {
        "episode_reward_mean",
        "success_rate",
        "episode_length_mean",
        "exploration_rate",
        "value_loss",
        "fps",
    }


def test_training_monitor_splits_live_scalars_by_run() -> None:
    _app()
    view = TrainingMonitorView()

    view.set_run_metrics(
        "run_a",
        "Task A",
        TrainingMetrics(step=10, episode_reward_mean=0.1, success_rate=0.2),
    )
    view.set_run_metrics(
        "run_b",
        "Task B",
        TrainingMetrics(step=20, episode_reward_mean=0.3, success_rate=0.4),
    )

    assert view.metrics_splitter.count() == 2
    assert set(view._run_metric_panels) == {"run_a", "run_b"}
    assert view._run_metric_panels["run_a"].title() == "Task A | run_a"
    assert view._run_metric_panels["run_b"].metric_cards["success_rate"].value_label.text() == "40.0%"
