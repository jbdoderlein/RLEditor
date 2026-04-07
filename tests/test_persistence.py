from __future__ import annotations

import json

from rleditor.application.persistence import ProjectState, ProjectStore
from rleditor.application.services import TrainingHistorySnapshot
from rleditor.core.models import (
    Checkpoint,
    DerivedTaskDefinition,
    EpisodeTrace,
    TaskDefinition,
    TaskSnapshot,
    TrainingRun,
    TrainingStatus,
)


def test_project_store_persists_task_workspace_history_and_checkpoint_state(tmp_path) -> None:
    store = ProjectStore(tmp_path / "project.json")
    learner_state = {
        "algorithm": "q_learning",
        "q_values": [{"state_key": "0", "action": 1, "value": 0.5}],
    }

    task = TaskDefinition(
        environment_id="tiny_env",
        name="Main Task",
        task_id="task_main",
        config={"difficulty": 1},
    )
    derived_task = DerivedTaskDefinition(
        environment_id="tiny_env",
        name="Derived Task",
        task_id="task_derived",
        parent_task_id="task_main",
        config={"difficulty": 2},
        source_episode_id=1,
        source_moment_index=2,
    )
    task_snapshot = TaskSnapshot(
        environment_id="tiny_env",
        task_name="Main Task",
        task_id="task_main",
        task_config={"difficulty": 1},
    )
    run = TrainingRun(
        run_id="run_001",
        task_id="task_main",
        status=TrainingStatus.FINISHED,
        parent_checkpoint_id=None,
    )
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_001",
        label="Checkpoint 001",
        created_at="2026-04-28 12:00:00",
        reason="run_finished",
        run_id="run_001",
        task_id="task_main",
        task_name="Main Task",
        step=10,
        episode=1,
        task_snapshot=task_snapshot,
        metadata={
            "algorithm": "q_learning",
            "learner_state": learner_state,
        },
    )
    trace = EpisodeTrace(
        episode_id=1,
        run_id="run_001",
        total_reward=1.0,
        success=True,
        task_snapshot=task_snapshot,
    )

    store.save(
        ProjectState(
            environment_id="tiny_env",
            task_workspace=[task, derived_task],
            history=TrainingHistorySnapshot(
                runs=[run],
                checkpoints=[checkpoint],
                episodes_by_run={"run_001": [trace]},
                run_task_snapshots={"run_001": task_snapshot},
            ),
        )
    )

    raw_payload = json.loads((tmp_path / "project.json").read_text(encoding="utf-8"))
    checkpoint_payload = raw_payload["history"]["checkpoints"][0]
    assert "learner_state" not in checkpoint_payload["metadata"]
    assert checkpoint_payload["storage_uri"] is not None
    assert (tmp_path / checkpoint_payload["storage_uri"]).exists()

    restored = store.load()

    assert restored is not None
    assert restored.environment_id == "tiny_env"
    assert isinstance(restored.task_workspace[1], DerivedTaskDefinition)
    assert restored.task_workspace[1].parent_task_id == "task_main"
    assert restored.history.runs[0].run_id == "run_001"
    assert restored.history.episodes_by_run["run_001"][0].success is True
    assert restored.history.run_task_snapshots["run_001"].task_name == "Main Task"
    assert restored.history.checkpoints[0].metadata["learner_state"] == learner_state
