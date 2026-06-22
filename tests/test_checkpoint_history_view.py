from __future__ import annotations

import os
import json

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog

from rleditor.application.services import TrainingHistorySnapshot
from rleditor.core.models import (
    Checkpoint,
    EpisodeStep,
    EpisodeTrace,
    RunConfig,
    TaskSnapshot,
    TrainingRun,
    TrainingStatus,
)
from rleditor.ui.views.checkpoint_history_view import CheckpointHistoryView, _TrainingEdgeEditDialog


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
    assert "Checkpoint ID" not in html
    assert "Training results" in html
    assert "Episode" in html
    assert "Success rate" in html
    assert "20.0%" in html
    assert "Mean reward" in html
    assert "Episode length" in html


def test_checkpoint_history_view_prefers_evaluation_metrics_and_lists_eval_episodes() -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_004",
        label="Checkpoint 004",
        created_at="2026-04-28 11:30:00",
        reason="run_finished",
        run_id="run_train",
        task_name="Training Task",
        step=50,
        episode=3,
        task_snapshot=TaskSnapshot(
            environment_id="tiny_env",
            task_name="Training Task",
        ),
        metadata={
            "evaluation": {
                "run_id": "eval_checkpoint_004",
                "task_name": "Evaluation Task",
                "environment_id": "tiny_env",
                "episode_count": 2,
                "max_steps_per_episode": 5,
                "seed": 42,
            },
            "evaluation_metrics": {
                "mean_reward": 1.0,
                "success_rate": 1.0,
                "episode_reward_mean": 1.0,
                "episode_length_mean": 3.0,
                "exploration_rate": 0.0,
                "value_loss": None,
                "policy_loss": None,
            },
            "training_metrics": {
                "mean_reward": 0.0,
                "success_rate": 0.0,
            },
        },
    )
    snapshot = TrainingHistorySnapshot(
        runs=[],
        checkpoints=[checkpoint],
        episodes_by_run={
            "eval_checkpoint_004": [
                EpisodeTrace(episode_id=1, run_id="eval_checkpoint_004", total_reward=1.0, success=True),
                EpisodeTrace(episode_id=2, run_id="eval_checkpoint_004", total_reward=1.0, success=True),
            ]
        },
        run_task_snapshots={},
    )

    view.set_history(snapshot)

    assert "Evaluation results" in view.checkpoint_details.toHtml()
    assert "100.0%" in view.checkpoint_details.toHtml()
    assert "Evaluation Task" in view.segment_details.toPlainText()
    assert "Seed: 42" in view.segment_details.toPlainText()
    assert view.episode_list.count() == 2


def test_checkpoint_history_view_edge_selection_shows_training_metrics_not_node_evaluation() -> None:
    _app()
    view = CheckpointHistoryView()
    task_snapshot = TaskSnapshot(
        environment_id="tiny_env",
        task_name="Training Task",
        task_id="task_train",
    )
    run = TrainingRun(
        run_id="run_train",
        task_id="task_train",
        status=TrainingStatus.FINISHED,
        started_at="2026-05-17 09:00:00",
        ended_at="2026-05-17 09:10:00",
        metadata={
            "algorithm": "q_learning",
            "seed": 123,
            "run_config": {
                "algorithm": "q_learning",
                "seed": 123,
                "max_steps": 100,
                "max_episodes": 12,
                "max_steps_per_episode": 8,
            },
        },
    )
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_010",
        label="Checkpoint 010",
        created_at="2026-05-17 09:10:00",
        reason="run_finished",
        run_id="run_train",
        task_id="task_train",
        task_name="Training Task",
        step=100,
        episode=4,
        task_snapshot=task_snapshot,
        metadata={
            "algorithm": "q_learning",
            "evaluation": {
                "run_id": "eval_checkpoint_010",
                "task_name": "Evaluation Task",
                "environment_id": "tiny_env",
                "episode_count": 2,
            },
            "evaluation_metrics": {
                "mean_reward": 1.0,
                "success_rate": 1.0,
            },
            "training_metrics": {
                "step": 100,
                "episode": 4,
                "mean_reward": 0.25,
                "success_rate": 0.35,
                "cumulative_reward": 14.5,
                "episode_reward_mean": 0.25,
                "episode_length_mean": 6.5,
                "exploration_rate": 0.1,
            },
        },
    )
    snapshot = TrainingHistorySnapshot(
        runs=[run],
        checkpoints=[checkpoint],
        episodes_by_run={
            "run_train": [
                EpisodeTrace(episode_id=1, run_id="run_train", total_reward=0.0, success=False),
                EpisodeTrace(episode_id=4, run_id="run_train", total_reward=1.0, success=True),
            ],
            "eval_checkpoint_010": [
                EpisodeTrace(episode_id=1, run_id="eval_checkpoint_010", total_reward=1.0, success=True),
            ],
        },
        run_task_snapshots={"run_train": task_snapshot},
    )

    view.set_history(snapshot)
    assert "Evaluation results" in view.checkpoint_details.toHtml()
    assert "100.0%" in view.checkpoint_details.toHtml()

    edge = view.graph_widget.edge_for_id("edge:checkpoint_010")
    assert edge is not None
    view.graph_widget.select_edge(edge.edge_id)
    view._show_edge_details(edge)

    html = view.checkpoint_details.toHtml()
    assert view.details_group.title() == "Training Details"
    assert "Selected training run" in html
    assert "Training setup" in html
    assert "Training results" in html
    assert "Task name" in html
    assert "Training Task" in html
    assert "Recorded episodes" in html
    assert "Max steps / episode" in html
    assert "Success rate" in html
    assert "35.0%" in html
    assert "Cumulative reward" in html
    assert "14.500" in html
    assert "100.0%" not in html
    assert "Episodes" in html
    assert "12" in html
    assert ">Max steps</td>" not in html
    assert "Run ID" not in html
    assert view.segment_group.title() == "Training Run"
    assert view.episode_list.count() == 2


def test_checkpoint_history_view_emits_training_config_when_edge_is_selected() -> None:
    _app()
    view = CheckpointHistoryView()
    config = RunConfig(
        algorithm="sb3_dqn",
        max_steps=None,
        max_episodes=17,
        max_steps_per_episode=80,
        episode_trace_sample_rate=0.4,
        learning_rate=0.15,
        gamma=0.93,
    )
    run = TrainingRun(
        run_id="run_config",
        task_id="task_config",
        status=TrainingStatus.FINISHED,
        metadata={"run_config": config.to_dict()},
    )
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_config",
        label="Config checkpoint",
        created_at="2026-05-17 09:10:00",
        reason="run_finished",
        run_id="run_config",
        task_id="task_config",
        task_name="Config Task",
        step=100,
        episode=17,
        task_snapshot=TaskSnapshot(environment_id="tiny_env", task_name="Config Task"),
    )
    view.set_history(
        TrainingHistorySnapshot(
            runs=[run],
            checkpoints=[checkpoint],
            episodes_by_run={},
            run_task_snapshots={},
        )
    )
    emitted: list[RunConfig] = []
    view.training_run_config_selected.connect(emitted.append)

    edge = view.graph_widget.edge_for_id("edge:checkpoint_config")
    assert edge is not None
    view._on_edge_selected(edge)

    assert len(emitted) == 1
    assert emitted[0].algorithm == "sb3_dqn"
    assert emitted[0].max_steps is None
    assert emitted[0].max_episodes == 17
    assert emitted[0].max_steps_per_episode == 80
    assert emitted[0].learning_rate == 0.15
    assert emitted[0].gamma == 0.93


def test_checkpoint_history_view_live_edit_dialog_edits_edge_segment_config() -> None:
    _app()
    view = CheckpointHistoryView()
    config = RunConfig(
        algorithm="q_learning",
        max_steps=100,
        max_episodes=10,
        max_steps_per_episode=80,
        episode_trace_sample_rate=0.4,
        learning_rate=0.15,
        gamma=0.93,
        epsilon=0.8,
        seed=12,
    )
    run = TrainingRun(
        run_id="run_segment",
        task_id="task_segment",
        status=TrainingStatus.FINISHED,
        metadata={"run_config": config.to_dict()},
    )
    source_checkpoint = Checkpoint(
        checkpoint_id="checkpoint_source",
        label="Source checkpoint",
        created_at="2026-05-17 09:00:00",
        reason="breakpoint",
        run_id="run_segment",
        task_id="task_segment",
        task_name="Segment Task",
        step=40,
        episode=4,
        task_snapshot=TaskSnapshot(environment_id="tiny_env", task_name="Segment Task"),
    )
    target_checkpoint = Checkpoint(
        checkpoint_id="checkpoint_target",
        label="Target checkpoint",
        created_at="2026-05-17 09:10:00",
        reason="run_finished",
        parent_checkpoint_id="checkpoint_source",
        run_id="run_segment",
        task_id="task_segment",
        task_name="Segment Task",
        step=70,
        episode=7,
        task_snapshot=TaskSnapshot(environment_id="tiny_env", task_name="Segment Task"),
    )
    view.set_history(
        TrainingHistorySnapshot(
            runs=[run],
            checkpoints=[source_checkpoint, target_checkpoint],
            episodes_by_run={},
            run_task_snapshots={},
        )
    )

    edge = view.graph_widget.edge_for_id("edge:checkpoint_target")
    assert edge is not None
    view._on_edge_selected(edge)

    live_config = view._live_edit_config_for_edge(edge)
    assert live_config is not None
    assert view.live_edit_button.isEnabled()
    assert live_config.max_episodes == 3
    assert live_config.max_steps == 30

    dialog = _TrainingEdgeEditDialog(live_config)
    dialog.episode_count_spin.setValue(5)
    dialog.max_steps_spin.setValue(55)
    dialog.learning_rate_spin.setValue(0.25)
    dialog.discount_factor_spin.setValue(0.9)
    dialog.epsilon_spin.setValue(0.6)
    dialog.seed_spin.setValue(-1)
    edited = dialog.edited_config()

    assert edited.algorithm == "q_learning"
    assert edited.max_episodes == 5
    assert edited.max_steps == 55
    assert edited.learning_rate == 0.25
    assert edited.gamma == 0.9
    assert edited.epsilon == 0.6
    assert edited.seed is None
    assert edited.hyperparameters["learning_rate"] == 0.25

    default_dialog = _TrainingEdgeEditDialog(RunConfig(max_steps_per_episode=None))
    assert default_dialog.max_steps_per_episode_spin.value() == 100


def test_checkpoint_history_view_live_edit_request_emits_selected_edge(monkeypatch) -> None:
    _app()
    view = CheckpointHistoryView()
    config = RunConfig(max_episodes=3, max_steps_per_episode=20)
    emitted_config = RunConfig(max_episodes=4, max_steps_per_episode=30)
    run = TrainingRun(
        run_id="run_live_edit",
        task_id="task_live_edit",
        status=TrainingStatus.FINISHED,
        metadata={"run_config": config.to_dict()},
    )
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_live_edit",
        label="Live edit checkpoint",
        created_at="2026-05-17 09:10:00",
        reason="run_finished",
        run_id="run_live_edit",
        task_id="task_live_edit",
        task_name="Live Edit Task",
        step=60,
        episode=3,
        task_snapshot=TaskSnapshot(environment_id="tiny_env", task_name="Live Edit Task"),
    )
    view.set_history(
        TrainingHistorySnapshot(
            runs=[run],
            checkpoints=[checkpoint],
            episodes_by_run={},
            run_task_snapshots={},
        )
    )
    edge = view.graph_widget.edge_for_id("edge:checkpoint_live_edit")
    assert edge is not None
    view.graph_widget.select_edge(edge.edge_id)
    captured_dialog_config: list[RunConfig] = []

    class _FakeDialog:
        def __init__(self, config: RunConfig, parent=None) -> None:
            _ = parent
            captured_dialog_config.append(config)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def edited_config(self) -> RunConfig:
            return emitted_config

    monkeypatch.setattr(
        "rleditor.ui.views.checkpoint_history_view._TrainingEdgeEditDialog",
        _FakeDialog,
    )
    emitted: list[tuple[object, RunConfig]] = []
    view.training_edge_live_edit_requested.connect(lambda edge, config: emitted.append((edge, config)))

    view._request_live_edit_for_selected_edge()

    assert captured_dialog_config[0].max_episodes == 3
    assert emitted == [(edge, emitted_config)]


def test_checkpoint_history_view_state_visit_heatmap_counts_selected_edge_episodes() -> None:
    _app()
    view = CheckpointHistoryView()
    task_snapshot = TaskSnapshot(
        environment_id="frozen_lake",
        task_name="Heatmap Task",
        task_config={"map_desc": ["SF", "FG"]},
    )
    run = TrainingRun(
        run_id="run_heatmap",
        task_id="task_heatmap",
        status=TrainingStatus.FINISHED,
        metadata={"run_config": RunConfig(max_episodes=2).to_dict()},
    )
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_heatmap",
        label="Heatmap checkpoint",
        created_at="2026-05-17 09:10:00",
        reason="run_finished",
        run_id="run_heatmap",
        task_id="task_heatmap",
        task_name="Heatmap Task",
        step=3,
        episode=2,
        task_snapshot=task_snapshot,
    )
    view.set_history(
        TrainingHistorySnapshot(
            runs=[run],
            checkpoints=[checkpoint],
            episodes_by_run={
                "run_heatmap": [
                    EpisodeTrace(
                        episode_id=1,
                        run_id="run_heatmap",
                        total_reward=1.0,
                        success=True,
                        initial_observation=0,
                        steps=[
                            EpisodeStep(
                                t=0,
                                observation=0,
                                action=2,
                                next_observation=1,
                                reward=0.0,
                                terminated=False,
                            ),
                            EpisodeStep(
                                t=1,
                                observation=1,
                                action=1,
                                next_observation=3,
                                reward=1.0,
                                terminated=True,
                            ),
                        ],
                    ),
                    EpisodeTrace(
                        episode_id=2,
                        run_id="run_heatmap",
                        total_reward=0.0,
                        success=False,
                        initial_observation=1,
                        steps=[
                            EpisodeStep(
                                t=0,
                                observation=1,
                                action=2,
                                next_observation=1,
                                reward=0.0,
                                terminated=False,
                            ),
                        ],
                    ),
                ]
            },
            run_task_snapshots={"run_heatmap": task_snapshot},
        )
    )
    edge = view.graph_widget.edge_for_id("edge:checkpoint_heatmap")
    assert edge is not None
    view._show_edge_details(edge)

    action_row = view.inspect_episode_button.parentWidget().layout().itemAt(2).layout()
    assert action_row.stretch(0) == 4
    assert action_row.stretch(1) == 1
    assert view.state_visit_heatmap_button.isEnabled()

    context = view._state_visit_heatmap_context()
    assert context is not None
    map_rows, visit_counts, episodes = context
    assert map_rows == ["SF", "FG"]
    assert len(episodes) == 2
    assert visit_counts == {0: 1, 1: 3, 3: 1}

    dialog = view._build_state_visit_heatmap_dialog(
        map_rows=map_rows,
        visit_counts=visit_counts,
        episodes=episodes,
    )
    try:
        assert dialog.visit_cells[(0, 0)].text() == "S\n0\n1"
        assert dialog.visit_cells[(0, 1)].text() == "F\n1\n3"
        assert dialog.visit_cells[(1, 0)].text() == "F\n2\n0"
        assert dialog.visit_cells[(1, 1)].text() == "G\n3\n1"
    finally:
        dialog.close()


def test_checkpoint_history_view_double_click_edge_requests_live_edit(monkeypatch) -> None:
    _app()
    view = CheckpointHistoryView()
    config = RunConfig(max_episodes=3, max_steps_per_episode=20)
    emitted_config = RunConfig(max_episodes=4, max_steps_per_episode=30)
    run = TrainingRun(
        run_id="run_live_edit",
        task_id="task_live_edit",
        status=TrainingStatus.FINISHED,
        metadata={"run_config": config.to_dict()},
    )
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_live_edit",
        label="Live edit checkpoint",
        created_at="2026-05-17 09:10:00",
        reason="run_finished",
        run_id="run_live_edit",
        task_id="task_live_edit",
        task_name="Live Edit Task",
        step=60,
        episode=3,
        task_snapshot=TaskSnapshot(environment_id="tiny_env", task_name="Live Edit Task"),
    )
    view.set_history(
        TrainingHistorySnapshot(
            runs=[run],
            checkpoints=[checkpoint],
            episodes_by_run={},
            run_task_snapshots={},
        )
    )
    edge = view.graph_widget.edge_for_id("edge:checkpoint_live_edit")
    assert edge is not None

    class _FakeDialog:
        def __init__(self, config: RunConfig, parent=None) -> None:
            _ = config, parent

        def exec(self):
            return QDialog.DialogCode.Accepted

        def edited_config(self) -> RunConfig:
            return emitted_config

    monkeypatch.setattr(
        "rleditor.ui.views.checkpoint_history_view._TrainingEdgeEditDialog",
        _FakeDialog,
    )
    emitted: list[tuple[object, RunConfig]] = []
    view.training_edge_live_edit_requested.connect(lambda edge, config: emitted.append((edge, config)))

    midpoint_x = (edge.source_point.x() + edge.target_point.x()) / 2
    midpoint_y = (edge.source_point.y() + edge.target_point.y()) / 2
    QTest.mouseDClick(
        view.graph_widget,
        Qt.MouseButton.LeftButton,
        pos=QPoint(int(midpoint_x), int(midpoint_y)),
    )

    assert emitted == [(edge, emitted_config)]


def test_checkpoint_history_view_multiple_node_selection_compares_metric_columns() -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint_a = Checkpoint(
        checkpoint_id="checkpoint_a",
        label="Checkpoint A",
        created_at="2026-05-17 09:00:00",
        reason="run_finished",
        run_id="run_a",
        task_name="Task A",
        step=50,
        episode=5,
        task_snapshot=TaskSnapshot(environment_id="tiny_env", task_name="Task A"),
        metadata={
            "evaluation": {"run_id": "eval_a", "episode_count": 3},
            "evaluation_metrics": {
                "episode": 3,
                "success_rate": 1.0,
                "mean_reward": 0.9,
                "episode_length_mean": 4.0,
            },
        },
    )
    checkpoint_b = Checkpoint(
        checkpoint_id="checkpoint_b",
        label="Checkpoint B",
        created_at="2026-05-17 09:05:00",
        reason="run_finished",
        run_id="run_b",
        task_name="Task B",
        step=80,
        episode=8,
        task_snapshot=TaskSnapshot(environment_id="tiny_env", task_name="Task B"),
        metadata={
            "training_metrics": {
                "episode": 8,
                "success_rate": 0.25,
                "mean_reward": -0.1,
                "episode_length_mean": 7.5,
            },
        },
    )

    view.set_history(
        TrainingHistorySnapshot(
            runs=[],
            checkpoints=[checkpoint_a, checkpoint_b],
            episodes_by_run={},
            run_task_snapshots={},
        )
    )
    view.graph_widget.select_node("checkpoint_a")
    view.graph_widget.select_node("checkpoint_b", additive=True)
    node_b = view.graph_widget.node_for_id("checkpoint_b")
    assert node_b is not None

    view._show_node_details(node_b)

    html = view.checkpoint_details.toHtml()
    assert view.graph_widget.selected_node_ids == ("checkpoint_a", "checkpoint_b")
    assert view.details_group.title() == "Node Details"
    assert "Selected checkpoints" in html
    assert "Metric" in html
    assert "Checkpoint A" in html
    assert "Checkpoint B" in html
    assert "Success rate" in html
    assert "100.0%" in html
    assert "25.0%" in html
    assert "Mean reward" in html
    assert "-0.100" in html
    assert "Episode length" in html
    assert "Checkpoint ID" not in html


def test_checkpoint_history_view_emits_manual_evaluation_for_selected_checkpoint() -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_004",
        label="Checkpoint 004",
        created_at="2026-04-28 11:30:00",
        reason="run_finished",
        run_id="run_train",
        task_name="Training Task",
        step=50,
        episode=3,
        task_snapshot=TaskSnapshot(
            environment_id="tiny_env",
            task_name="Training Task",
        ),
        metadata={"learner_state": {"q_values": []}},
    )
    captured: list[Checkpoint] = []
    view.checkpoint_evaluation_requested.connect(captured.append)
    view.set_history(TrainingHistorySnapshot([], [checkpoint], {}, {}))

    assert view.evaluate_checkpoint_button.isHidden()
    node = view.graph_widget.node_for_id("checkpoint_004")
    assert node is not None

    class _FakeDialog:
        selected_action = "evaluate"

        def exec(self):
            return QDialog.DialogCode.Accepted

    view._build_checkpoint_node_action_dialog = lambda _checkpoint: _FakeDialog()  # type: ignore[method-assign]
    view._open_checkpoint_node_actions(node)

    assert captured
    assert captured[0].checkpoint_id == "checkpoint_004"


def test_checkpoint_history_view_emits_delete_for_selected_checkpoints() -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint_a = Checkpoint(
        checkpoint_id="checkpoint_a",
        label="Checkpoint A",
        created_at="2026-04-28 11:30:00",
        reason="run_finished",
        run_id="run_a",
        task_name="Task A",
        step=10,
        episode=1,
        task_snapshot=TaskSnapshot(environment_id="tiny_env", task_name="Task A"),
    )
    checkpoint_b = Checkpoint(
        checkpoint_id="checkpoint_b",
        label="Checkpoint B",
        created_at="2026-04-28 11:31:00",
        reason="run_finished",
        parent_checkpoint_id="checkpoint_a",
        run_id="run_b",
        task_name="Task B",
        step=20,
        episode=2,
        task_snapshot=TaskSnapshot(environment_id="tiny_env", task_name="Task B"),
    )
    captured: list[list[str]] = []
    view.checkpoint_delete_requested.connect(captured.append)
    view.set_history(TrainingHistorySnapshot([], [checkpoint_a, checkpoint_b], {}, {}))
    view.graph_widget.select_node("checkpoint_a")
    view.graph_widget.select_node("checkpoint_b", additive=True)
    node_b = view.graph_widget.node_for_id("checkpoint_b")
    assert node_b is not None
    view._show_node_details(node_b)

    assert view.delete_checkpoint_button.isHidden()
    QTest.keyClick(view.graph_widget, Qt.Key.Key_Delete)

    assert captured == [["checkpoint_a", "checkpoint_b"]]


def test_checkpoint_history_view_shows_q_table_for_q_learning_node() -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_q",
        label="Q checkpoint",
        created_at="2026-04-28 11:30:00",
        reason="run_finished",
        run_id="run_q",
        task_name="Frozen Lake",
        step=12,
        episode=2,
        task_snapshot=TaskSnapshot(
            environment_id="frozen_lake",
            task_name="Frozen Lake",
            task_config={"map_desc": ["SF", "FG"]},
        ),
        metadata={
            "algorithm": "q_learning",
            "learner_state": {
                "algorithm": "q_learning",
                "q_values": [
                    {"state_key": "0", "action": 1, "value": 0.5},
                    {"state_key": "0", "action": 2, "value": 0.25},
                    {"state_key": "1", "action": 2, "value": 1.0},
                ],
            },
        },
    )

    view.set_history(TrainingHistorySnapshot([], [checkpoint], {}, {}))

    assert view.show_q_table_button.isEnabled()
    assert view.show_q_table_button.isHidden()
    actions_dialog = view._build_checkpoint_node_action_dialog(checkpoint)
    try:
        assert actions_dialog.show_q_table_button.isEnabled()
        assert not actions_dialog.show_q_table_button.isHidden()
    finally:
        actions_dialog.close()
    dialog = view._build_q_table_dialog(checkpoint)
    try:
        assert not hasattr(dialog, "q_table")
        assert dialog.policy_cells[(0, 0)].text() == "S\n↓\n0.500"
        assert dialog.policy_cells[(0, 1)].text() == "F\n→\n1.000"
    finally:
        dialog.close()


def test_checkpoint_history_view_double_click_checkpoint_opens_actions_and_requests_rename() -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_rename",
        label="Old checkpoint name",
        created_at="2026-06-17 10:00:00",
        reason="run_finished",
    )
    emitted: list[Checkpoint] = []
    view.checkpoint_rename_requested.connect(emitted.append)

    view.set_history(TrainingHistorySnapshot([], [checkpoint], {}, {}))
    node = view.graph_widget.node_for_id("checkpoint_rename")
    assert node is not None
    assert node.label == "Old checkpoint name"

    class _FakeDialog:
        selected_action = "rename"

        def exec(self):
            return QDialog.DialogCode.Accepted

    view._build_checkpoint_node_action_dialog = lambda _checkpoint: _FakeDialog()  # type: ignore[method-assign]

    QTest.mouseDClick(
        view.graph_widget,
        Qt.MouseButton.LeftButton,
        pos=QPoint(int(node.center.x()), int(node.center.y())),
    )

    assert len(emitted) == 1
    assert emitted[0].checkpoint_id == "checkpoint_rename"


def test_checkpoint_history_view_shows_sb3_policy_map_for_frozen_lake_checkpoint(monkeypatch) -> None:
    _app()

    class _FakeModel:
        def predict(self, observation, *, deterministic: bool):
            assert deterministic is True
            return int(observation) % 4, None

    def _load_model(*, algorithm, env, learner_state):
        assert algorithm == "sb3_dqn"
        assert env is None
        assert learner_state["backend"] == "stable_baselines3"
        return _FakeModel()

    monkeypatch.setattr(
        "rleditor.ui.views.checkpoint_history_view.load_stable_baselines3_model",
        _load_model,
    )
    view = CheckpointHistoryView()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_sb3",
        label="SB3 checkpoint",
        created_at="2026-04-28 11:30:00",
        reason="run_finished",
        run_id="run_sb3",
        task_name="Frozen Lake",
        step=12,
        episode=2,
        task_snapshot=TaskSnapshot(
            environment_id="frozen_lake",
            task_name="Frozen Lake",
            task_config={"map_desc": ["SF", "FG"]},
        ),
        metadata={
            "algorithm": "sb3_dqn",
            "learner_state": {
                "algorithm": "sb3_dqn",
                "backend": "stable_baselines3",
                "model_zip_base64": "abc",
            },
        },
    )

    view.set_history(TrainingHistorySnapshot([], [checkpoint], {}, {}))

    assert view.show_policy_button.isEnabled()
    assert view.show_policy_button.isHidden()
    actions_dialog = view._build_checkpoint_node_action_dialog(checkpoint)
    try:
        assert actions_dialog.show_policy_button.isEnabled()
        assert not actions_dialog.show_policy_button.isHidden()
    finally:
        actions_dialog.close()
    dialog = view._build_policy_dialog(checkpoint)
    try:
        assert dialog.policy_cells[(0, 0)].text() == "S\n←"
        assert dialog.policy_cells[(0, 1)].text() == "F\n↓"
    finally:
        dialog.close()


def test_checkpoint_history_view_builds_selected_checkpoint_export_payload() -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_005",
        label="Checkpoint 005",
        created_at="2026-04-28 11:45:00",
        reason="run_finished",
        run_id="run_selected",
        task_name="Selected Task",
        step=25,
        episode=4,
        task_snapshot=TaskSnapshot(
            environment_id="frozen_lake",
            task_name="Selected Task",
            task_id="task_selected",
        ),
        metadata={
            "algorithm": "sb3_dqn",
            "learner_state": {
                "backend": "stable_baselines3",
                "model_zip_base64": "abc",
            },
            "evaluation_metrics": {
                "mean_reward": 1.0,
                "success_rate": 1.0,
            },
        },
    )

    payload = view._checkpoint_export_payload(checkpoint)

    assert payload["checkpoint_id"] == "checkpoint_005"
    assert payload["metadata"]["algorithm"] == "sb3_dqn"
    assert payload["metadata"]["learner_state"]["backend"] == "stable_baselines3"
    assert payload["task_snapshot"]["task_id"] == "task_selected"
    assert "training_runs" not in payload
    assert "tasks" not in payload


def test_checkpoint_history_view_parses_imported_checkpoint_payload() -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_006",
        label="Checkpoint 006",
        created_at="2026-04-28 12:00:00",
        reason="import_test",
        task_snapshot=TaskSnapshot(
            environment_id="frozen_lake",
            task_name="Imported Task",
        ),
        metadata={"algorithm": "sb3_dqn"},
    )

    parsed = view._checkpoint_from_import_payload(checkpoint.to_dict())

    assert parsed.checkpoint_id == "checkpoint_006"
    assert parsed.task_snapshot is not None
    assert parsed.task_snapshot.environment_id == "frozen_lake"
    assert parsed.metadata["algorithm"] == "sb3_dqn"


def test_checkpoint_history_view_merged_import_routes_by_file_content(monkeypatch, tmp_path) -> None:
    _app()
    view = CheckpointHistoryView()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_imported",
        label="Imported checkpoint",
        created_at="2026-04-28 12:00:00",
        reason="import_test",
        task_snapshot=TaskSnapshot(environment_id="frozen_lake", task_name="Imported Task"),
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint.to_dict()), encoding="utf-8")
    curriculum_payload = {
        "curriculum": {"steps": [{"step_id": 1, "env_id": 0, "steps": 10}]},
        "environments": [{"environment_id": "tiny_env", "task_name": "Task"}],
    }
    curriculum_path = tmp_path / "curriculum.json"
    curriculum_path.write_text(json.dumps(curriculum_payload), encoding="utf-8")

    imported_checkpoints: list[Checkpoint] = []
    imported_curricula: list[object] = []
    view.checkpoint_import_requested.connect(imported_checkpoints.append)
    view.curriculum_import_requested.connect(imported_curricula.append)

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(checkpoint_path), "JSON Files (*.json)"),
    )
    view._import_from_file()

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(curriculum_path), "JSON Files (*.json)"),
    )
    view._import_from_file()

    assert [item.checkpoint_id for item in imported_checkpoints] == ["checkpoint_imported"]
    assert imported_curricula == [curriculum_payload]


def test_checkpoint_history_view_builds_curriculum_export_for_selected_lineage() -> None:
    _app()
    view = CheckpointHistoryView()
    assert view.export_curriculum_plan_button.text() == "Export Curriculum"
    assert view.show_training_report_button.text() == "Show Training Report"
    assert view.import_checkpoint_button.text() == "Import"
    assert view.import_curriculum_button.isHidden()
    assert view.export_curriculum_button.text() == "Export Trace"
    assert view.export_curriculum_button.isHidden()
    main_task = TaskSnapshot(
        environment_id="frozen_lake",
        task_name="Main Task",
        task_id="task_main",
        task_config={"size": 4},
    )
    subtask = TaskSnapshot(
        environment_id="frozen_lake",
        task_name="Sub Task",
        task_id="task_sub",
        task_config={"start_state": 5},
    )
    run_1 = TrainingRun(
        run_id="run_main",
        task_id="task_main",
        status=TrainingStatus.FINISHED,
        parent_checkpoint_id=None,
        metadata={
            "algorithm": "q_learning",
            "run_config": {"algorithm": "q_learning", "max_steps": 100},
        },
    )
    run_2 = TrainingRun(
        run_id="run_sub",
        task_id="task_sub",
        status=TrainingStatus.STOPPED,
        parent_checkpoint_id="checkpoint_001",
        metadata={
            "algorithm": "q_learning",
            "run_config": {"algorithm": "q_learning", "max_steps": 50},
        },
    )
    checkpoint_1 = Checkpoint(
        checkpoint_id="checkpoint_001",
        label="Checkpoint 001",
        created_at="2026-04-30 10:00:00",
        reason="run_finished",
        run_id="run_main",
        task_id="task_main",
        task_name="Main Task",
        step=100,
        episode=2,
        task_snapshot=main_task,
        metadata={"algorithm": "q_learning", "learner_state": {"q": [1, 2, 3]}},
    )
    checkpoint_2 = Checkpoint(
        checkpoint_id="checkpoint_002",
        label="Checkpoint 002",
        created_at="2026-04-30 10:05:00",
        reason="breakpoint_success_rate_gte",
        parent_checkpoint_id="checkpoint_001",
        run_id="run_sub",
        task_id="task_sub",
        task_name="Sub Task",
        step=50,
        episode=3,
        task_snapshot=subtask,
        metadata={"algorithm": "q_learning"},
    )
    snapshot = TrainingHistorySnapshot(
        runs=[run_1, run_2],
        checkpoints=[checkpoint_1, checkpoint_2],
        episodes_by_run={
            "run_main": [
                EpisodeTrace(episode_id=1, run_id="run_main", total_reward=0.0, success=False),
                EpisodeTrace(episode_id=2, run_id="run_main", total_reward=1.0, success=True),
                EpisodeTrace(episode_id=3, run_id="run_main", total_reward=1.0, success=True),
            ],
            "run_sub": [
                EpisodeTrace(episode_id=1, run_id="run_sub", total_reward=0.0, success=False),
                EpisodeTrace(episode_id=3, run_id="run_sub", total_reward=1.0, success=True),
                EpisodeTrace(episode_id=4, run_id="run_sub", total_reward=1.0, success=True),
            ],
        },
        run_task_snapshots={
            "run_main": main_task,
            "run_sub": subtask,
        },
    )

    view.set_history(snapshot)
    payload = view._curriculum_export_payload(checkpoint_2)

    assert "export_type" not in payload
    assert "schema_version" not in payload
    assert "lineage_checkpoint_ids" not in payload
    assert "origin" not in payload
    assert "segments" not in payload
    assert "checkpoints" not in payload
    assert payload["meta"] == {
        "target_checkpoint_id": "checkpoint_002",
        "includes_episode_traces": True,
        "training_run_count": 2,
        "task_count": 2,
    }

    tasks = payload["tasks"]
    assert [task["task_ref_id"] for task in tasks] == ["task_001", "task_002"]
    assert [task["task_name"] for task in tasks] == ["Main Task", "Sub Task"]
    assert tasks[0]["task_config"] == {"size": 4}
    assert tasks[1]["task_config"] == {"start_state": 5}

    training_runs = payload["training_runs"]
    assert len(training_runs) == 2
    assert training_runs[0]["source_checkpoint_id"] == "checkpoint_000_untrained"
    assert training_runs[0]["target_checkpoint_id"] == "checkpoint_001"
    assert training_runs[0]["task_ref_id"] == "task_001"
    assert training_runs[0]["parameters"]["max_steps"] == 100
    assert [trace["episode_id"] for trace in training_runs[0]["recorded_episode_traces"]] == [1, 2]

    assert training_runs[1]["source_checkpoint_id"] == "checkpoint_001"
    assert training_runs[1]["target_checkpoint_id"] == "checkpoint_002"
    assert training_runs[1]["task_ref_id"] == "task_002"
    assert training_runs[1]["parameters"]["max_steps"] == 50
    assert [trace["episode_id"] for trace in training_runs[1]["recorded_episode_traces"]] == [1, 3]

    compact_payload = view._curriculum_export_payload(
        checkpoint_2,
        include_episode_traces=False,
    )
    compact_training_runs = compact_payload["training_runs"]

    assert compact_payload["meta"]["includes_episode_traces"] is False
    assert compact_training_runs[0]["recorded_episode_trace_count"] == 2
    assert "recorded_episode_traces" not in compact_training_runs[0]
    assert [trace["episode_id"] for trace in compact_training_runs[0]["recorded_episode_summaries"]] == [1, 2]
    assert compact_training_runs[1]["recorded_episode_trace_count"] == 2
    assert "recorded_episode_traces" not in compact_training_runs[1]
    assert [trace["episode_id"] for trace in compact_training_runs[1]["recorded_episode_summaries"]] == [1, 3]

    plan_payload = view._curriculum_plan_export_payload(checkpoint_2)
    assert plan_payload["curriculum"]["size"] == 2
    assert [environment["task_name"] for environment in plan_payload["environments"]] == [
        "Main Task",
        "Sub Task",
    ]

    plan_steps = plan_payload["curriculum"]["steps"]
    assert plan_steps[0]["step_id"] == 1
    assert plan_steps[0]["env_id"] == 0
    assert plan_steps[0]["steps"] == 100
    assert plan_steps[0]["algorithm"] == "q_learning"
    assert plan_steps[1]["step_id"] == 2
    assert plan_steps[1]["env_id"] == 1
    assert plan_steps[1]["steps"] == 50

    serialized_plan = json.dumps(plan_payload)
    assert "recorded_episode" not in serialized_plan
    assert "training_runs" not in plan_payload


def test_checkpoint_history_view_builds_training_report_for_selected_lineage() -> None:
    _app()
    view = CheckpointHistoryView()
    main_task = TaskSnapshot(
        environment_id="tiny_env",
        task_name="Main Task",
        task_id="task_main",
        task_config={"difficulty": 1},
    )
    subtask = TaskSnapshot(
        environment_id="tiny_env",
        task_name="Sub Task",
        task_id="task_sub",
        task_config={"difficulty": 2},
    )
    run_1 = TrainingRun(
        run_id="run_main",
        task_id="task_main",
        status=TrainingStatus.FINISHED,
        metadata={"run_config": RunConfig(max_episodes=2).to_dict()},
    )
    run_2 = TrainingRun(
        run_id="run_sub",
        task_id="task_sub",
        status=TrainingStatus.FINISHED,
        parent_checkpoint_id="checkpoint_001",
        metadata={"run_config": RunConfig(max_episodes=1).to_dict()},
    )
    checkpoint_1 = Checkpoint(
        checkpoint_id="checkpoint_001",
        label="Checkpoint 001",
        created_at="2026-05-17 10:00:00",
        reason="run_finished",
        run_id="run_main",
        task_id="task_main",
        task_name="Main Task",
        step=20,
        episode=2,
        task_snapshot=main_task,
    )
    checkpoint_2 = Checkpoint(
        checkpoint_id="checkpoint_002",
        label="Checkpoint 002",
        created_at="2026-05-17 10:05:00",
        reason="run_finished",
        parent_checkpoint_id="checkpoint_001",
        run_id="run_sub",
        task_id="task_sub",
        task_name="Sub Task",
        step=10,
        episode=1,
        task_snapshot=subtask,
    )
    view.set_history(
        TrainingHistorySnapshot(
            runs=[run_1, run_2],
            checkpoints=[checkpoint_1, checkpoint_2],
            episodes_by_run={
                "run_main": [
                    EpisodeTrace(episode_id=1, run_id="run_main", total_reward=1.0, success=True),
                    EpisodeTrace(episode_id=2, run_id="run_main", total_reward=-0.5, success=False),
                ],
                "run_sub": [
                    EpisodeTrace(episode_id=1, run_id="run_sub", total_reward=2.0, success=True),
                ],
            },
            run_task_snapshots={
                "run_main": main_task,
                "run_sub": subtask,
            },
        )
    )

    assert view.show_training_report_button.isEnabled()
    report = view._training_report_payload(checkpoint_2)

    assert report["total_episodes"] == 3
    assert report["recorded_episode_count"] == 3
    assert report["cumulative_reward_points"] == [
        (1, 1.0),
        (2, 0.5),
        (3, 2.5),
    ]
    assert report["task_switches"] == [(2, "Sub Task")]

    dialog = view._build_training_report_dialog(checkpoint_2)
    try:
        assert dialog.report["target_checkpoint_id"] == "checkpoint_002"
        assert dialog.windowTitle() == "Training report"
    finally:
        dialog.close()

    view.graph_widget.select_node("checkpoint_001")
    view.graph_widget.select_node("checkpoint_002", additive=True)
    selected_checkpoints = view._selected_training_report_checkpoints()
    assert [checkpoint.checkpoint_id for checkpoint in selected_checkpoints] == [
        "checkpoint_001",
        "checkpoint_002",
    ]

    multi_dialog = view._build_training_report_dialog(selected_checkpoints)
    try:
        assert [report["target_checkpoint_id"] for report in multi_dialog.reports] == [
            "checkpoint_001",
            "checkpoint_002",
        ]
        assert multi_dialog.report["reports"][1]["target_checkpoint_id"] == "checkpoint_002"
    finally:
        multi_dialog.close()
