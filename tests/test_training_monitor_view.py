from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from rleditor.core.models import Breakpoint, RunConfig, TrainingMetrics
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


def test_training_monitor_breakpoint_group_keeps_fixed_height() -> None:
    _app()
    view = TrainingMonitorView()

    assert view.breakpoint_group.maximumHeight() == view.breakpoint_group.minimumHeight()


def test_training_monitor_builds_episode_based_q_learning_config() -> None:
    _app()
    view = TrainingMonitorView()

    view.episode_count_spin.setValue(250)
    view.max_steps_per_episode_spin.setValue(75)
    view.learning_rate_spin.setValue(0.2)
    view.discount_factor_spin.setValue(0.95)

    config = view.build_config()

    assert config.max_steps is None
    assert config.max_episodes == 250
    assert config.max_steps_per_episode == 75
    assert config.learning_rate == pytest.approx(0.2)
    assert config.gamma == pytest.approx(0.95)
    assert config.hyperparameters["learning_rate"] == pytest.approx(0.2)
    assert config.hyperparameters["gamma"] == pytest.approx(0.95)


def test_training_monitor_can_build_unlimited_episode_config() -> None:
    _app()
    view = TrainingMonitorView()

    view.episode_count_spin.setValue(0)
    config = view.build_config()

    assert config.max_steps is None
    assert config.max_episodes is None


def test_training_monitor_defaults_to_100_steps_per_episode() -> None:
    _app()
    view = TrainingMonitorView()

    config = view.build_config()

    assert config.max_steps_per_episode == 100


def test_training_monitor_applies_run_config_to_controls() -> None:
    _app()
    view = TrainingMonitorView()

    view.set_config(
        RunConfig(
            algorithm="sb3_ppo",
            max_steps=None,
            max_episodes=42,
            max_steps_per_episode=125,
            episode_trace_sample_rate=0.25,
            learning_rate=0.2,
            gamma=0.95,
            breakpoints=[
                Breakpoint(
                    kind="success_rate_gte",
                    value=0.8,
                    window=20,
                    actions=["pause", "checkpoint"],
                )
            ],
        )
    )

    config = view.build_config()

    assert config.algorithm == "sb3_ppo"
    assert config.max_steps is None
    assert config.max_episodes == 42
    assert config.max_steps_per_episode == 125
    assert config.episode_trace_sample_rate == pytest.approx(0.25)
    assert config.learning_rate == pytest.approx(0.2)
    assert config.gamma == pytest.approx(0.95)
    assert len(config.breakpoints) == 1
    assert config.breakpoints[0].kind == "success_rate_gte"
    assert config.breakpoints[0].value == pytest.approx(0.8)
    assert config.breakpoints[0].window == 20


def test_training_monitor_can_select_stable_baselines3_dqn() -> None:
    _app()
    view = TrainingMonitorView()

    index = view.algorithm_combo.findData("sb3_dqn")
    assert index >= 0

    view.algorithm_combo.setCurrentIndex(index)
    config = view.build_config()

    assert config.algorithm == "sb3_dqn"


def test_training_monitor_can_select_stable_baselines3_ppo() -> None:
    _app()
    view = TrainingMonitorView()

    index = view.algorithm_combo.findData("sb3_ppo")
    assert index >= 0

    view.algorithm_combo.setCurrentIndex(index)
    config = view.build_config()

    assert config.algorithm == "sb3_ppo"


def test_training_monitor_uses_selected_task_algorithm_hint() -> None:
    _app()
    view = TrainingMonitorView()

    view.set_algorithm_hint("sb3_ppo")
    config = view.build_config()

    assert config.algorithm == "sb3_ppo"


def test_training_monitor_ignores_unknown_algorithm_hint() -> None:
    _app()
    view = TrainingMonitorView()

    view.set_algorithm_hint("random")
    config = view.build_config()

    assert config.algorithm == "q_learning"


def test_training_monitor_surfaces_classic_q_learning_metrics() -> None:
    _app()
    view = TrainingMonitorView()

    view.set_metrics(
        TrainingMetrics(
            step=128,
            episode=9,
            cumulative_reward=12.75,
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
    assert "cumulative_reward=12.750" in view.metrics_label.text()
    assert "epsilon=" not in view.metrics_label.text()
    assert "td_error=0.143" in view.metrics_label.text()
    assert set(view._metric_cards) == {
        "episode_reward_mean",
        "success_rate",
        "episode_length_mean",
        "cumulative_reward",
        "value_loss",
        "fps",
    }


def test_training_monitor_sparklines_keep_full_run_history() -> None:
    _app()
    view = TrainingMonitorView()

    for index in range(200):
        view.set_metrics(
            TrainingMetrics(
                step=index,
                cumulative_reward=float(index),
                episode_reward_mean=float(index),
                success_rate=0.0,
                episode_length_mean=1.0,
                fps=60.0,
            )
        )

    values = view._metric_cards["cumulative_reward"].sparkline._values
    assert len(values) == 200
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(199.0)


def test_training_monitor_adds_axes_to_episode_length_and_cumulative_reward() -> None:
    _app()
    view = TrainingMonitorView()

    assert view._metric_cards["episode_length_mean"].sparkline._show_axes is True
    assert view._metric_cards["cumulative_reward"].sparkline._show_axes is True
    assert view._metric_cards["episode_reward_mean"].sparkline._show_axes is False
    assert view._metric_cards["success_rate"].sparkline._show_axes is False


def test_training_monitor_axis_sparklines_use_episode_numbers() -> None:
    _app()
    view = TrainingMonitorView()

    for index in range(251):
        view.set_metrics(
            TrainingMetrics(
                step=index * 10,
                episode=index * 40,
                cumulative_reward=float(index),
                episode_reward_mean=0.0,
                success_rate=0.0,
                episode_length_mean=50.0,
                fps=60.0,
            )
        )

    cumulative_sparkline = view._metric_cards["cumulative_reward"].sparkline
    length_sparkline = view._metric_cards["episode_length_mean"].sparkline
    reward_sparkline = view._metric_cards["episode_reward_mean"].sparkline

    assert cumulative_sparkline._x_values[-1] == pytest.approx(10_000.0)
    assert length_sparkline._x_values[-1] == pytest.approx(10_000.0)
    assert reward_sparkline._x_values[-1] == pytest.approx(250.0)


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
