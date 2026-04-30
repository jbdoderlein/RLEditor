from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.application.services import TrainingHistorySnapshot
from rleditor.core.models import Checkpoint, EpisodeTrace, TaskSnapshot, TrainingRun, TrainingStatus
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


def test_checkpoint_history_view_builds_curriculum_export_for_selected_lineage() -> None:
    _app()
    view = CheckpointHistoryView()
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
