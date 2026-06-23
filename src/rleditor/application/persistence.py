from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from rleditor.application.services import TrainingHistorySnapshot
from rleditor.core.models import (
    Checkpoint,
    DerivedTaskDefinition,
    EpisodeTrace,
    TaskDefinition,
    TaskSnapshot,
    TrainingRun,
)


_SCHEMA_VERSION = 1


@dataclass(slots=True)
class ProjectState:
    environment_id: str
    task_workspace: list[TaskDefinition]
    history: TrainingHistorySnapshot


class ProjectStore:
    """JSON-backed project store for task workspace and training artifacts."""

    def __init__(self, project_path: Path | str) -> None:
        self.project_path = Path(project_path).expanduser()

    @classmethod
    def default_for_environment(cls, environment_id: str) -> ProjectStore:
        data_home = os.environ.get("XDG_DATA_HOME")
        root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
        return cls(root / "rleditor" / f"{environment_id}.json")

    def load(self) -> ProjectState | None:
        if not self.project_path.exists():
            return None

        payload = json.loads(self.project_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"Project file is not a JSON object: {self.project_path}"
            raise ValueError(msg)

        environment_id = str(payload.get("environment_id", ""))
        tasks_payload = payload.get("task_workspace", [])
        history_payload = payload.get("history", {})

        tasks = [
            _task_from_dict(item)
            for item in tasks_payload
            if isinstance(item, dict)
        ]
        history = self._history_from_dict(history_payload if isinstance(history_payload, dict) else {})

        return ProjectState(
            environment_id=environment_id,
            task_workspace=tasks,
            history=history,
        )

    def save(self, state: ProjectState, progress_callback: Callable[[int, str], None] | None = None) -> None:
        def progress(percent: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(percent, message)

        progress(5, "Preparing project save")
        self.project_path.parent.mkdir(parents=True, exist_ok=True)
        progress(10, "Serializing task workspace")
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "environment_id": state.environment_id,
            "task_workspace": [task.to_dict() for task in state.task_workspace],
            "history": self._history_to_dict(state.history, progress_callback=progress_callback),
        }
        progress(90, "Writing project file")
        tmp_path = self.project_path.with_suffix(self.project_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.project_path)
        progress(100, "Project save complete")

    def save_checkpoint_state(self, checkpoint_id: str, learner_state: dict[str, Any]) -> str | None:
        if not learner_state:
            return None

        checkpoint_dir = self.project_path.with_suffix("").parent / f"{self.project_path.stem}_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_dir / f"{checkpoint_id}.json"
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "learner_state": deepcopy(learner_state),
        }
        tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(state_path)
        return state_path.relative_to(self.project_path.parent).as_posix()

    def _load_checkpoint_state(self, storage_uri: str | None) -> dict[str, Any] | None:
        if not storage_uri:
            return None

        state_path = Path(storage_uri)
        if not state_path.is_absolute():
            state_path = self.project_path.parent / state_path
        if not state_path.exists():
            return None

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        learner_state = payload.get("learner_state")
        return deepcopy(learner_state) if isinstance(learner_state, dict) else None

    def _history_from_dict(self, payload: dict[str, Any]) -> TrainingHistorySnapshot:
        runs_payload = payload.get("runs", [])
        checkpoints_payload = payload.get("checkpoints", [])
        episodes_payload = payload.get("episodes_by_run", {})
        snapshots_payload = payload.get("run_task_snapshots", {})

        checkpoints: list[Checkpoint] = []
        for item in checkpoints_payload if isinstance(checkpoints_payload, list) else []:
            if not isinstance(item, dict):
                continue
            checkpoint = Checkpoint.from_dict(item)
            metadata = dict(checkpoint.metadata)
            if "learner_state" not in metadata:
                learner_state = self._load_checkpoint_state(checkpoint.storage_uri)
                if learner_state is not None:
                    metadata["learner_state"] = learner_state
                    checkpoint.metadata = metadata
            checkpoints.append(checkpoint)

        episodes_by_run: dict[str, list[EpisodeTrace]] = {}
        if isinstance(episodes_payload, dict):
            for run_id, trace_payloads in episodes_payload.items():
                if not isinstance(trace_payloads, list):
                    continue
                episodes_by_run[str(run_id)] = [
                    EpisodeTrace.from_dict(trace_payload)
                    for trace_payload in trace_payloads
                    if isinstance(trace_payload, dict)
                ]

        run_task_snapshots: dict[str, TaskSnapshot] = {}
        if isinstance(snapshots_payload, dict):
            for run_id, snapshot_payload in snapshots_payload.items():
                if isinstance(snapshot_payload, dict):
                    run_task_snapshots[str(run_id)] = TaskSnapshot.from_dict(snapshot_payload)

        return TrainingHistorySnapshot(
            runs=[
                TrainingRun.from_dict(item)
                for item in runs_payload
                if isinstance(item, dict)
            ] if isinstance(runs_payload, list) else [],
            checkpoints=checkpoints,
            episodes_by_run=episodes_by_run,
            run_task_snapshots=run_task_snapshots,
        )

    def _history_to_dict(
        self,
        history: TrainingHistorySnapshot,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        def progress(percent: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(percent, message)

        progress(20, "Serializing training runs")
        checkpoints = []
        total_checkpoints = max(1, len(history.checkpoints))
        for index, checkpoint in enumerate(history.checkpoints, start=1):
            checkpoints.append(self._checkpoint_to_project_dict(checkpoint))
            progress(
                20 + int(index / total_checkpoints * 25),
                f"Saving checkpoint state {index}/{len(history.checkpoints)}",
            )

        progress(50, "Serializing recorded episodes")
        episodes_by_run: dict[str, list[dict[str, Any]]] = {}
        total_runs = max(1, len(history.episodes_by_run))
        for index, (run_id, traces) in enumerate(history.episodes_by_run.items(), start=1):
            episodes_by_run[run_id] = [trace.to_dict() for trace in traces]
            progress(
                50 + int(index / total_runs * 30),
                f"Serializing episode traces {index}/{len(history.episodes_by_run)}",
            )

        progress(85, "Serializing task snapshots")
        return {
            "runs": [run.to_dict() for run in history.runs],
            "checkpoints": checkpoints,
            "episodes_by_run": episodes_by_run,
            "run_task_snapshots": {
                run_id: snapshot.to_dict()
                for run_id, snapshot in history.run_task_snapshots.items()
            },
        }

    def _checkpoint_to_project_dict(self, checkpoint: Checkpoint) -> dict[str, Any]:
        payload = checkpoint.to_dict()
        metadata = dict(payload.get("metadata") or {})
        storage_uri = checkpoint.storage_uri
        learner_state = metadata.get("learner_state")
        if storage_uri is None and isinstance(learner_state, dict):
            storage_uri = self.save_checkpoint_state(checkpoint.checkpoint_id, learner_state)
            payload["storage_uri"] = storage_uri
        if storage_uri:
            metadata.pop("learner_state", None)
            payload["metadata"] = metadata
        return payload


def _task_from_dict(payload: dict[str, Any]) -> TaskDefinition:
    derived_keys = {
        "derived_task_id",
        "parent_task_id",
        "derivation_reason",
        "source_episode_id",
        "source_moment_index",
        "source_run_id",
        "start_state",
        "goal_state",
    }
    if any(key in payload for key in derived_keys):
        return DerivedTaskDefinition.from_dict(payload)
    return TaskDefinition.from_dict(payload)
