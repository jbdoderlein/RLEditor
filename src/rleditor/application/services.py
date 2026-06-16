from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QObject, Signal

from rleditor.core.models import (
    BreakpointEvent,
    DerivedTaskDefinition,
    EpisodeTrace,
    Checkpoint,
    RunConfig,
    TaskDefinition,
    TaskDerivationOptions,
    TaskSnapshot,
    TrainingMetrics,
    TrainingRun,
    TrainingStatus,
)
from rleditor.infra.evaluation_runner import EvaluationResult, evaluate_policy
from rleditor.infra.training_runner import TrainingRunner
from rleditor.plugins.registry import PluginRegistry


@dataclass(slots=True)
class TrainingHistorySnapshot:
    runs: list[TrainingRun]
    checkpoints: list[Checkpoint]
    episodes_by_run: dict[str, list[EpisodeTrace]]
    run_task_snapshots: dict[str, TaskSnapshot]


@dataclass(slots=True)
class _RunContext:
    run_id: str
    runner: TrainingRunner
    task: TaskDefinition
    config: RunConfig
    task_snapshot: TaskSnapshot
    latest_metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    status: TrainingStatus = TrainingStatus.IDLE


class TaskService:
    """Task and derived-task orchestration independent from GUI widgets."""

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    def create_default_task(self, plugin_id: str) -> TaskDefinition:
        plugin = self._registry.get_environment_plugin(plugin_id)
        return plugin.backend.default_task()

    def task_from_snapshot(self, snapshot: TaskSnapshot) -> TaskDefinition:
        return TaskDefinition(
            environment_id=snapshot.environment_id,
            name=snapshot.task_name,
            task_id=snapshot.task_id,
            config=deepcopy(snapshot.task_config),
            reward_config=deepcopy(snapshot.reward_config),
            termination_config=deepcopy(snapshot.termination_config),
            metadata=deepcopy(snapshot.metadata),
        )

    def derive_task(
        self,
        source_task: TaskDefinition,
        *,
        name: str,
        options: TaskDerivationOptions | None = None,
        parent_task_id: str | None = None,
    ) -> DerivedTaskDefinition:
        options = options or TaskDerivationOptions()
        resolved_parent_task_id = parent_task_id or source_task.task_id or source_task.name
        merged_config = deepcopy(source_task.config)
        merged_config.update(deepcopy(options.config_updates))

        merged_rewards = deepcopy(source_task.reward_config)
        merged_rewards.update(deepcopy(options.reward_config_updates))

        merged_termination = deepcopy(source_task.termination_config)
        merged_termination.update(deepcopy(options.termination_config_updates))

        merged_metadata = deepcopy(source_task.metadata)
        merged_metadata.update(
            {
                "derived_from": source_task.task_id or source_task.name,
                "derivation_reason": options.derivation_reason,
                "source_episode_id": options.source_episode_id,
                "source_moment_index": options.source_moment_index,
                "source_run_id": options.source_run_id,
            }
        )

        return DerivedTaskDefinition(
            environment_id=source_task.environment_id,
            name=name,
            task_id=None,
            config=merged_config,
            reward_config=merged_rewards,
            termination_config=merged_termination,
            metadata=merged_metadata,
            parent_task_id=resolved_parent_task_id,
            derivation_reason=options.derivation_reason,
            source_episode_id=options.source_episode_id,
            source_moment_index=options.source_moment_index,
            source_run_id=options.source_run_id,
            start_state=options.start_state,
            goal_state=options.goal_state,
        )


class TrainingService(QObject):
    """High-level API consumed by the GUI and backed by a runner adapter."""

    status_changed = Signal(object)
    metrics_updated = Signal(object)
    run_metrics_updated = Signal(str, object, str)
    episode_captured = Signal(object)
    breakpoint_triggered = Signal(object)
    history_changed = Signal()

    def __init__(
        self,
        registry: PluginRegistry | None = None,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._runner: TrainingRunner | None = None
        self._status = TrainingStatus.IDLE
        self._latest_metrics = TrainingMetrics()
        self._run_contexts: dict[str, _RunContext] = {}
        self._current_session_run_ids: list[str] = []
        self._primary_live_run_id: str | None = None
        self._coordinated_breakpoint_pause_pending = False
        self._checkpoint_counter = 0
        self._checkpoints: list[Checkpoint] = []
        self._runs: list[TrainingRun] = []
        self._runs_by_id: dict[str, TrainingRun] = {}
        self._episodes_by_run: dict[str, list[EpisodeTrace]] = {}
        self._pending_episodes_by_run: dict[str, list[EpisodeTrace]] = {}
        self._run_task_snapshots: dict[str, TaskSnapshot] = {}

    @property
    def status(self) -> TrainingStatus:
        return self._status

    def history_snapshot(self, *, deep: bool = True) -> TrainingHistorySnapshot:
        if not deep:
            return TrainingHistorySnapshot(
                runs=list(self._runs),
                checkpoints=list(self._checkpoints),
                episodes_by_run={
                    run_id: list(episodes)
                    for run_id, episodes in self._episodes_by_run.items()
                },
                run_task_snapshots=dict(self._run_task_snapshots),
            )

        return TrainingHistorySnapshot(
            runs=deepcopy(self._runs),
            checkpoints=deepcopy(self._checkpoints),
            episodes_by_run={run_id: deepcopy(episodes) for run_id, episodes in self._episodes_by_run.items()},
            run_task_snapshots=deepcopy(self._run_task_snapshots),
        )

    def load_history(self, snapshot: TrainingHistorySnapshot) -> None:
        if self._has_live_runs():
            msg = "Cannot load persisted training history while training is active."
            raise RuntimeError(msg)

        self._runner = None
        self._status = TrainingStatus.IDLE
        self._latest_metrics = TrainingMetrics()
        self._run_contexts.clear()
        self._current_session_run_ids = []
        self._primary_live_run_id = None
        self._coordinated_breakpoint_pause_pending = False
        self._runs = deepcopy(snapshot.runs)
        self._runs_by_id = {run.run_id: run for run in self._runs}
        self._checkpoints = deepcopy(snapshot.checkpoints)
        self._episodes_by_run = {
            run_id: deepcopy(episodes)
            for run_id, episodes in snapshot.episodes_by_run.items()
        }
        self._pending_episodes_by_run = {
            run.run_id: []
            for run in self._runs
        }
        self._run_task_snapshots = deepcopy(snapshot.run_task_snapshots)
        self._checkpoint_counter = self._next_checkpoint_counter_floor()
        self.history_changed.emit()

    def import_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        if self._has_live_runs():
            msg = "Cannot import a checkpoint while training is active."
            raise RuntimeError(msg)

        imported_checkpoint = deepcopy(checkpoint)
        original_checkpoint_id = imported_checkpoint.checkpoint_id
        if not original_checkpoint_id or self._checkpoint_id_exists(original_checkpoint_id):
            imported_checkpoint.checkpoint_id = self._next_available_checkpoint_id()
            imported_checkpoint.label = self._renamed_import_label(
                imported_checkpoint.label,
                imported_checkpoint.checkpoint_id,
            )

        self._checkpoints.append(imported_checkpoint)
        if imported_checkpoint.run_id is not None and imported_checkpoint.task_snapshot is not None:
            self._run_task_snapshots.setdefault(
                imported_checkpoint.run_id,
                deepcopy(imported_checkpoint.task_snapshot),
            )
        self._checkpoint_counter = self._next_checkpoint_counter_floor()
        self.history_changed.emit()
        return deepcopy(imported_checkpoint)

    def delete_checkpoint_tree(self, checkpoint_ids: list[str] | tuple[str, ...] | set[str]) -> list[str]:
        if self._has_live_runs():
            msg = "Cannot delete checkpoints while training is active."
            raise RuntimeError(msg)

        root_ids = {str(checkpoint_id) for checkpoint_id in checkpoint_ids if str(checkpoint_id)}
        known_ids = {checkpoint.checkpoint_id for checkpoint in self._checkpoints}
        root_ids &= known_ids
        if not root_ids:
            return []

        deleted_ids = self._descendant_checkpoint_ids(root_ids)
        deleted_checkpoints = [
            checkpoint
            for checkpoint in self._checkpoints
            if checkpoint.checkpoint_id in deleted_ids
        ]
        self._checkpoints = [
            checkpoint
            for checkpoint in self._checkpoints
            if checkpoint.checkpoint_id not in deleted_ids
        ]

        removed_run_ids = self._orphaned_run_ids_after_checkpoint_delete(
            deleted_checkpoints=deleted_checkpoints,
            deleted_checkpoint_ids=deleted_ids,
        )
        self._purge_run_records(removed_run_ids)

        self._checkpoint_counter = self._next_checkpoint_counter_floor()
        self.history_changed.emit()
        return sorted(deleted_ids)

    def _descendant_checkpoint_ids(self, root_ids: set[str]) -> set[str]:
        deleted_ids = set(root_ids)
        changed = True
        while changed:
            changed = False
            for checkpoint in self._checkpoints:
                if checkpoint.checkpoint_id in deleted_ids:
                    continue
                if checkpoint.parent_checkpoint_id in deleted_ids:
                    deleted_ids.add(checkpoint.checkpoint_id)
                    changed = True
        return deleted_ids

    def _orphaned_run_ids_after_checkpoint_delete(
        self,
        *,
        deleted_checkpoints: list[Checkpoint],
        deleted_checkpoint_ids: set[str],
    ) -> set[str]:
        surviving_run_ids = {
            checkpoint.run_id
            for checkpoint in self._checkpoints
            if checkpoint.run_id is not None
        }
        candidate_run_ids = {
            checkpoint.run_id
            for checkpoint in deleted_checkpoints
            if checkpoint.run_id is not None
        }
        candidate_run_ids.update(
            run.run_id
            for run in self._runs
            if run.parent_checkpoint_id in deleted_checkpoint_ids
        )
        for checkpoint in deleted_checkpoints:
            evaluation = checkpoint.metadata.get("evaluation")
            if isinstance(evaluation, dict):
                run_id = evaluation.get("run_id")
                if isinstance(run_id, str) and run_id:
                    candidate_run_ids.add(run_id)
        return candidate_run_ids - surviving_run_ids

    def _purge_run_records(self, run_ids: set[str]) -> None:
        if not run_ids:
            return

        for run_id in run_ids:
            context = self._run_contexts.pop(run_id, None)
            if context is not None:
                try:
                    context.runner.stop()
                except Exception:
                    pass
            self._runs_by_id.pop(run_id, None)
            self._episodes_by_run.pop(run_id, None)
            self._pending_episodes_by_run.pop(run_id, None)
            self._run_task_snapshots.pop(run_id, None)

        self._runs = [run for run in self._runs if run.run_id not in run_ids]
        self._current_session_run_ids = [
            run_id for run_id in self._current_session_run_ids if run_id not in run_ids
        ]
        if self._primary_live_run_id in run_ids:
            self._primary_live_run_id = None
        if self._primary_live_run_id is None:
            self._runner = None

    def evaluate_checkpoint(self, checkpoint_id: str, evaluation_policy: dict[str, Any]) -> Checkpoint:
        if self._has_live_runs():
            msg = "Cannot evaluate a checkpoint while training is active."
            raise RuntimeError(msg)

        checkpoint_index = self._checkpoint_index(checkpoint_id)
        if checkpoint_index is None:
            msg = f"Unknown checkpoint: {checkpoint_id}"
            raise RuntimeError(msg)

        checkpoint = deepcopy(self._checkpoints[checkpoint_index])
        metadata = dict(checkpoint.metadata)
        learner_state = metadata.get("learner_state")
        if not isinstance(learner_state, dict):
            msg = f"Checkpoint {checkpoint_id} does not contain a learner state."
            raise RuntimeError(msg)

        config = self._config_for_checkpoint(checkpoint)
        config.evaluation_policy = deepcopy(evaluation_policy)
        evaluation_result, evaluation_error = self._run_checkpoint_evaluation(
            checkpoint_id=checkpoint.checkpoint_id,
            config=config,
            learner_state=deepcopy(learner_state),
            policy=config.evaluation_policy,
        )

        evaluation_run_id = f"eval_{checkpoint.checkpoint_id}"
        metadata.pop("evaluation", None)
        metadata.pop("evaluation_metrics", None)
        metadata.pop("evaluation_error", None)
        if evaluation_result is not None:
            metadata["evaluation_metrics"] = self._metrics_payload(evaluation_result.metrics)
            metadata["evaluation"] = self._evaluation_metadata_payload(
                evaluation_result,
                config=config,
            )
        elif evaluation_error is not None:
            self._episodes_by_run.pop(evaluation_run_id, None)
            self._pending_episodes_by_run.pop(evaluation_run_id, None)
            self._run_task_snapshots.pop(evaluation_run_id, None)
            metadata["evaluation_error"] = evaluation_error

        checkpoint.metadata = metadata
        self._checkpoints[checkpoint_index] = checkpoint
        self.history_changed.emit()
        if evaluation_error is not None:
            msg = f"Evaluation failed for {checkpoint_id}: {evaluation_error}"
            raise RuntimeError(msg)
        return deepcopy(checkpoint)

    def evaluate_checkpoint_multiple(
        self,
        checkpoint_id: str,
        evaluation_policies: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._has_live_runs():
            msg = "Cannot evaluate a checkpoint while training is active."
            raise RuntimeError(msg)

        checkpoint_index = self._checkpoint_index(checkpoint_id)
        if checkpoint_index is None:
            msg = f"Unknown checkpoint: {checkpoint_id}"
            raise RuntimeError(msg)

        checkpoint = deepcopy(self._checkpoints[checkpoint_index])
        learner_state = checkpoint.metadata.get("learner_state")
        if not isinstance(learner_state, dict):
            msg = f"Checkpoint {checkpoint_id} does not contain a learner state."
            raise RuntimeError(msg)

        base_config = self._config_for_checkpoint(checkpoint)
        rows: list[dict[str, Any]] = []
        for index, policy in enumerate(evaluation_policies, start=1):
            config = RunConfig.from_dict(base_config.to_dict())
            config.evaluation_policy = deepcopy(policy)
            row = self._evaluation_result_base_row(
                checkpoint_id=checkpoint.checkpoint_id,
                policy=policy,
            )
            result, error = self._run_checkpoint_evaluation(
                checkpoint_id=checkpoint.checkpoint_id,
                config=config,
                learner_state=deepcopy(learner_state),
                policy=config.evaluation_policy,
                evaluation_run_id=f"eval_{checkpoint.checkpoint_id}_{index}",
                store_result=False,
            )
            if result is not None:
                row.update(
                    {
                        "task_name": result.task.name,
                        "task_id": result.task.task_id,
                        "environment_id": result.task.environment_id,
                        **self._metrics_payload(result.metrics),
                        "error": "",
                    }
                )
            else:
                row["error"] = error or "Evaluation did not run."
            rows.append(row)
        return rows

    def start(
        self,
        task: TaskDefinition,
        config: RunConfig,
        *,
        initial_checkpoint: Checkpoint | None = None,
        start_from_scratch: bool = False,
        run_in_background: bool = False,
    ) -> None:
        self.start_many(
            [task],
            config,
            initial_checkpoint=initial_checkpoint,
            start_from_scratch=start_from_scratch,
            run_in_background=run_in_background,
        )

    def start_many(
        self,
        tasks: list[TaskDefinition],
        config: RunConfig,
        *,
        initial_checkpoint: Checkpoint | None = None,
        start_from_scratch: bool = False,
        run_in_background: bool = False,
    ) -> None:
        self.start_many_with_configs(
            tasks,
            [config for _task in tasks],
            initial_checkpoint=initial_checkpoint,
            start_from_scratch=start_from_scratch,
            run_in_background=run_in_background,
        )

    def start_many_with_configs(
        self,
        tasks: list[TaskDefinition],
        configs: list[RunConfig],
        *,
        initial_checkpoint: Checkpoint | None = None,
        start_from_scratch: bool = False,
        run_in_background: bool = False,
    ) -> None:
        if not tasks:
            return
        if len(tasks) != len(configs):
            msg = "Each training task must have a matching run config."
            raise RuntimeError(msg)
        if self._has_live_runs():
            msg = "Training is already active; stop or finish the current run group before starting another."
            raise RuntimeError(msg)

        resolved_env_factories = [
            self._resolve_env_factory(task)
            for task in tasks
        ]
        self._latest_metrics = TrainingMetrics()
        self._current_session_run_ids = []
        self._primary_live_run_id = None
        self._coordinated_breakpoint_pause_pending = False
        pending_starts: list[tuple[_RunContext, Checkpoint | None, Callable[[TaskDefinition], object]]] = []
        created_run_ids: list[str] = []

        for index, task in enumerate(tasks):
            config = configs[index]
            run_id = f"run_{uuid4().hex[:8]}"
            parent_checkpoint = self._resolve_initial_checkpoint(
                task,
                config,
                preferred_checkpoint=initial_checkpoint,
                start_from_scratch=start_from_scratch,
            )
            task_snapshot = self._build_task_snapshot(task)
            runner = TrainingRunner()
            self._connect_runner(run_id, runner)

            context = _RunContext(
                run_id=run_id,
                runner=runner,
                task=task,
                config=config,
                task_snapshot=task_snapshot,
            )
            self._run_contexts[run_id] = context
            self._current_session_run_ids.append(run_id)
            created_run_ids.append(run_id)
            if index == 0:
                self._primary_live_run_id = run_id
                self._runner = runner

            training_run = TrainingRun(
                run_id=run_id,
                task_id=task.task_id,
                run_config_id=config.run_config_id,
                status=TrainingStatus.RUNNING,
                started_at=self._timestamp_now(),
                parent_checkpoint_id=None if parent_checkpoint is None else parent_checkpoint.checkpoint_id,
                metadata={
                    "algorithm": config.algorithm,
                    "seed": config.seed,
                    "task_name": task.name,
                    "started_from_checkpoint_id": None if parent_checkpoint is None else parent_checkpoint.checkpoint_id,
                    "started_from_scratch": start_from_scratch or parent_checkpoint is None,
                    "run_config": config.to_dict(),
                },
            )
            self._runs.append(training_run)
            self._runs_by_id[run_id] = training_run
            self._episodes_by_run[run_id] = []
            self._pending_episodes_by_run[run_id] = []
            self._run_task_snapshots[run_id] = task_snapshot

            pending_starts.append((context, parent_checkpoint, resolved_env_factories[index]))

        try:
            for context, parent_checkpoint, env_factory in pending_starts:
                context.runner.start(
                    context.task,
                    context.config,
                    run_id=context.run_id,
                    env_factory=env_factory,
                    initial_checkpoint=parent_checkpoint,
                    auto_run=False,
                )
        except RuntimeError:
            self._discard_created_runs(created_run_ids)
            self._emit_status_if_changed()
            self.history_changed.emit()
            raise

        self.history_changed.emit()
        if run_in_background:
            for context, _parent_checkpoint, _env_factory in pending_starts:
                context.runner.start_background()

    def pause(self) -> None:
        self._coordinated_breakpoint_pause_pending = False
        for context in self._session_contexts(live_only=True):
            if context.status == TrainingStatus.RUNNING:
                context.runner.pause()

    def resume(self) -> None:
        self._coordinated_breakpoint_pause_pending = False
        for context in self._session_contexts(live_only=True):
            if context.status == TrainingStatus.PAUSED:
                context.runner.resume()

    def stop(self) -> None:
        self._coordinated_breakpoint_pause_pending = False
        for context in self._session_contexts(live_only=True):
            context.runner.stop()

    def _connect_runner(self, run_id: str, runner: TrainingRunner) -> None:
        runner.status_changed.connect(
            lambda status, run_id=run_id: self._on_run_status_changed(run_id, status)
        )
        runner.metrics_updated.connect(
            lambda metrics, run_id=run_id: self._on_run_metrics_updated(run_id, metrics)
        )
        runner.episode_captured.connect(
            lambda trace, run_id=run_id: self._on_run_episode_captured(run_id, trace)
        )
        runner.breakpoint_triggered.connect(
            lambda event, run_id=run_id: self._on_run_breakpoint_triggered(run_id, event)
        )

    def _resolve_env_factory(self, task: TaskDefinition) -> Callable[[TaskDefinition], object]:
        if self._registry is None:
            msg = (
                f"Cannot start training for task '{task.name}': no environment registry is available."
            )
            raise RuntimeError(msg)

        try:
            plugin = self._registry.get_environment_plugin(task.environment_id)
        except KeyError as exc:
            msg = (
                f"Cannot start training for task '{task.name}': "
                f"unknown environment '{task.environment_id}'."
            )
            raise RuntimeError(msg) from exc

        env_factory = getattr(plugin.backend, "create_env", None)
        if not callable(env_factory):
            msg = (
                f"Cannot start training for task '{task.name}': "
                f"environment '{task.environment_id}' does not expose create_env."
            )
            raise RuntimeError(msg)
        return env_factory

    def _discard_created_runs(self, run_ids: list[str]) -> None:
        run_id_set = set(run_ids)
        for run_id in run_ids:
            context = self._run_contexts.pop(run_id, None)
            if context is not None:
                try:
                    context.runner.stop()
                except Exception:
                    pass
            self._runs_by_id.pop(run_id, None)
            self._episodes_by_run.pop(run_id, None)
            self._pending_episodes_by_run.pop(run_id, None)
            self._run_task_snapshots.pop(run_id, None)

        self._runs = [run for run in self._runs if run.run_id not in run_id_set]
        self._current_session_run_ids = [
            run_id for run_id in self._current_session_run_ids if run_id not in run_id_set
        ]
        if self._primary_live_run_id in run_id_set:
            self._primary_live_run_id = None
            self._runner = None

    def _on_run_status_changed(self, run_id: str, status: TrainingStatus) -> None:
        context = self._run_contexts.get(run_id)
        if context is None:
            return

        context.status = status
        run = self._runs_by_id.get(run_id)
        if run is not None:
            run.status = status
            run.metadata["latest_metrics"] = self._metrics_payload(context.latest_metrics)
            if status in {TrainingStatus.FINISHED, TrainingStatus.STOPPED}:
                run.ended_at = self._timestamp_now()

        flushed_episodes = False
        if status in {TrainingStatus.PAUSED, TrainingStatus.FINISHED, TrainingStatus.STOPPED}:
            flushed_episodes = self._flush_pending_episodes(run_id=run_id, emit_latest_episode=True)

        checkpoint_created = False
        if status in {TrainingStatus.FINISHED, TrainingStatus.STOPPED}:
            checkpoint_created = self._capture_checkpoint_for_run(run_id, reason=f"run_{status.value}")

        self._emit_status_if_changed()
        if status in {TrainingStatus.FINISHED, TrainingStatus.STOPPED}:
            self._coordinated_breakpoint_pause_pending = False
            if not checkpoint_created:
                self.history_changed.emit()
        elif flushed_episodes or status == TrainingStatus.PAUSED:
            self.history_changed.emit()

    def _on_run_metrics_updated(self, run_id: str, metrics: TrainingMetrics) -> None:
        context = self._run_contexts.get(run_id)
        if context is None:
            return

        context.latest_metrics = metrics
        self.run_metrics_updated.emit(run_id, metrics, context.task.name)
        self._latest_metrics = self._aggregate_session_metrics()
        self.metrics_updated.emit(self._latest_metrics)

    def _on_run_episode_captured(self, run_id: str, trace: EpisodeTrace) -> None:
        if trace.run_id is None:
            trace.run_id = run_id
        if trace.run_id is not None:
            self._pending_episodes_by_run.setdefault(trace.run_id, []).append(trace)
            context = self._run_contexts.get(trace.run_id)
            if context is not None and context.status != TrainingStatus.RUNNING:
                flushed_episodes = self._flush_pending_episodes(
                    run_id=trace.run_id,
                    emit_latest_episode=True,
                )
                if flushed_episodes:
                    self.history_changed.emit()

    def _on_run_breakpoint_triggered(self, run_id: str, event: BreakpointEvent) -> None:
        context = self._run_contexts.get(run_id)
        if context is None:
            return

        actions = set(event.breakpoint.actions)
        flushed_episodes = self._flush_pending_episodes(
            run_id=run_id,
            emit_latest_episode=True,
        )

        if not self._coordinated_breakpoint_pause_pending and len(self._session_contexts(live_only=True)) > 1:
            self._coordinated_breakpoint_pause_pending = True
            for other_context in self._session_contexts(live_only=True):
                if other_context.run_id == run_id or other_context.status != TrainingStatus.RUNNING:
                    continue
                other_context.runner.pause()
            if "pause" not in actions and "stop" not in actions and context.status == TrainingStatus.RUNNING:
                context.runner.pause()

        event_with_context = BreakpointEvent(
            breakpoint=event.breakpoint,
            step=event.step,
            episode=event.episode,
            message=f"{context.task.name}: {event.message}",
        )
        self.breakpoint_triggered.emit(event_with_context)
        if "checkpoint" in actions:
            self._capture_checkpoint_for_run(run_id, reason=f"breakpoint_{event.breakpoint.kind}")
        elif flushed_episodes:
            self.history_changed.emit()

    def _capture_checkpoint_for_run(self, run_id: str, *, reason: str) -> bool:
        context = self._run_contexts.get(run_id)
        if context is None:
            return False
        if context.latest_metrics.step <= 0:
            return False

        run = self._runs_by_id.get(run_id)
        if run is None:
            return False

        run_checkpoints = [
            checkpoint for checkpoint in self._checkpoints if checkpoint.run_id == run_id
        ]
        if reason in {"run_finished", "run_stopped"} and run_checkpoints:
            latest_checkpoint = run_checkpoints[-1]
            if (
                latest_checkpoint.step == context.latest_metrics.step
                and latest_checkpoint.episode == context.latest_metrics.episode
            ):
                return False

        self._checkpoint_counter += 1
        checkpoint_id = f"checkpoint_{self._checkpoint_counter:03d}"
        created_at = self._timestamp_now()
        label = (
            f"Checkpoint {self._checkpoint_counter:03d} | "
            f"{context.task.name} | "
            f"ep {context.latest_metrics.episode} step {context.latest_metrics.step}"
        )
        parent_checkpoint_id = None
        run_checkpoints = [checkpoint for checkpoint in self._checkpoints if checkpoint.run_id == run_id]
        if run_checkpoints:
            parent_checkpoint_id = run_checkpoints[-1].checkpoint_id
        else:
            parent_checkpoint_id = run.parent_checkpoint_id

        task_snapshot = self._run_task_snapshots.get(run_id) or context.task_snapshot
        learner_state = context.runner.export_learner_state()
        evaluation_result, evaluation_error = self._evaluate_checkpoint_policy(
            checkpoint_id=checkpoint_id,
            context=context,
            learner_state=learner_state,
        )

        metadata: dict[str, Any] = {
            "algorithm": context.config.algorithm,
            "seed": context.config.seed,
            "run_config_id": context.config.run_config_id,
            "training_metrics": self._metrics_payload(context.latest_metrics),
            "learner_state": learner_state,
        }
        if evaluation_result is not None:
            metadata["evaluation_metrics"] = self._metrics_payload(evaluation_result.metrics)
            metadata["evaluation"] = self._evaluation_metadata_payload(
                evaluation_result,
                config=context.config,
            )
        elif evaluation_error is not None:
            metadata["evaluation_error"] = evaluation_error

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            label=label,
            created_at=created_at,
            reason=reason,
            parent_checkpoint_id=parent_checkpoint_id,
            run_id=run_id,
            task_id=context.task.task_id,
            task_name=context.task.name,
            step=context.latest_metrics.step,
            episode=context.latest_metrics.episode,
            task_snapshot=deepcopy(task_snapshot),
            metadata=metadata,
        )

        self._checkpoints.append(checkpoint)
        self.history_changed.emit()
        return True

    def _evaluate_checkpoint_policy(
        self,
        *,
        checkpoint_id: str,
        context: _RunContext,
        learner_state: dict[str, Any],
    ) -> tuple[EvaluationResult | None, str | None]:
        policy = context.config.evaluation_policy
        return self._run_checkpoint_evaluation(
            checkpoint_id=checkpoint_id,
            config=context.config,
            learner_state=learner_state,
            policy=policy,
        )

    def _evaluation_result_base_row(
        self,
        *,
        checkpoint_id: str,
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        task_payload = policy.get("task") if isinstance(policy, dict) else None
        if isinstance(task_payload, dict):
            task_name = str(
                task_payload.get("name", task_payload.get("task_name", "Evaluation Task"))
            )
            task_id = task_payload.get("task_id")
            environment_id = str(task_payload.get("environment_id", ""))
        else:
            task_name = "Evaluation Task"
            task_id = None
            environment_id = ""
        return {
            "checkpoint_id": checkpoint_id,
            "task_name": task_name,
            "task_id": task_id,
            "environment_id": environment_id,
            "episode_count": policy.get("episode_count") if isinstance(policy, dict) else None,
            "max_steps_per_episode": (
                policy.get("max_steps_per_episode") if isinstance(policy, dict) else None
            ),
            "seed": policy.get("seed") if isinstance(policy, dict) else None,
        }

    def _run_checkpoint_evaluation(
        self,
        *,
        checkpoint_id: str,
        config: RunConfig,
        learner_state: dict[str, Any],
        policy: dict[str, Any],
        evaluation_run_id: str | None = None,
        store_result: bool = True,
    ) -> tuple[EvaluationResult | None, str | None]:
        if not isinstance(policy, dict) or not policy:
            return None, None

        task_payload = policy.get("task")
        if not isinstance(task_payload, dict):
            return None, "Evaluation policy does not contain a task snapshot."

        try:
            episode_count = int(policy.get("episode_count", 0))
        except (TypeError, ValueError):
            return None, "Evaluation episode count is invalid."
        if episode_count <= 0:
            return None, None

        try:
            evaluation_seed = self._evaluation_seed(config)
        except ValueError as exc:
            return None, str(exc)

        max_steps_per_episode = self._evaluation_max_steps_per_episode(config)
        evaluation_task = self._task_from_policy_payload(task_payload)
        evaluation_run_id = evaluation_run_id or f"eval_{checkpoint_id}"

        try:
            result = evaluate_policy(
                task=evaluation_task,
                config=config,
                learner_state=learner_state,
                env_factory=self._resolve_env_factory(evaluation_task),
                run_id=evaluation_run_id,
                episode_count=episode_count,
                max_steps_per_episode=max_steps_per_episode,
                seed=evaluation_seed,
            )
        except Exception as exc:
            return None, str(exc)

        if store_result:
            self._episodes_by_run[evaluation_run_id] = deepcopy(result.episodes)
            self._pending_episodes_by_run[evaluation_run_id] = []
            self._run_task_snapshots[evaluation_run_id] = deepcopy(result.task_snapshot)
        return result, None

    def _evaluation_metadata_payload(
        self,
        result: EvaluationResult,
        *,
        config: RunConfig,
    ) -> dict[str, Any]:
        return {
            "run_id": result.run_id,
            "task_id": result.task.task_id,
            "task_name": result.task.name,
            "environment_id": result.task.environment_id,
            "episode_count": result.metrics.episode,
            "max_steps_per_episode": self._evaluation_max_steps_per_episode(config),
            "seed": self._evaluation_seed(config),
            "trace_sample_rate": 1.0,
        }

    def _evaluation_max_steps_per_episode(self, config: RunConfig) -> int | None:
        policy = config.evaluation_policy
        if not isinstance(policy, dict):
            return None
        raw_value = policy.get("max_steps_per_episode")
        if raw_value is None:
            return None
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _evaluation_seed(self, config: RunConfig) -> int | None:
        policy = config.evaluation_policy
        if not isinstance(policy, dict):
            return config.seed
        raw_value = policy.get("seed")
        if raw_value is None or raw_value == "":
            return config.seed
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            msg = "Evaluation seed is invalid."
            raise ValueError(msg) from exc

    def _task_from_policy_payload(self, payload: dict[str, Any]) -> TaskDefinition:
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

    def _flush_pending_episodes(
        self,
        *,
        run_id: str | None,
        emit_latest_episode: bool,
    ) -> bool:
        if run_id is None:
            return False

        pending = self._pending_episodes_by_run.get(run_id)
        if not pending:
            return False

        stored_episodes = self._episodes_by_run.setdefault(run_id, [])
        stored_episodes.extend(pending)
        latest_trace = pending[-1]
        self._pending_episodes_by_run[run_id] = []

        if emit_latest_episode:
            self.episode_captured.emit(latest_trace)
        return True

    def _build_task_snapshot(self, task: TaskDefinition) -> TaskSnapshot:
        return TaskSnapshot(
            environment_id=task.environment_id,
            task_name=task.name,
            task_id=task.task_id,
            task_config=deepcopy(task.config),
            reward_config=deepcopy(task.reward_config),
            termination_config=deepcopy(task.termination_config),
            metadata=deepcopy(task.metadata),
        )

    def _aggregate_session_metrics(self) -> TrainingMetrics:
        contexts = self._session_contexts(live_only=True)
        if not contexts:
            contexts = self._session_contexts(live_only=False)
        if not contexts:
            return TrainingMetrics()
        if len(contexts) == 1:
            return contexts[0].latest_metrics

        def mean(values: list[float | None]) -> float | None:
            numeric = [float(value) for value in values if value is not None]
            if not numeric:
                return None
            return sum(numeric) / len(numeric)

        reward_step_mean = mean([context.latest_metrics.reward_step for context in contexts])
        episode_reward_mean = mean([context.latest_metrics.episode_reward_mean for context in contexts])
        success_rate = mean([context.latest_metrics.success_rate for context in contexts])
        episode_length_mean = mean([context.latest_metrics.episode_length_mean for context in contexts])
        exploration_rate = mean([context.latest_metrics.exploration_rate for context in contexts])
        value_loss = mean([context.latest_metrics.value_loss for context in contexts])
        policy_loss = mean([context.latest_metrics.policy_loss for context in contexts])

        return TrainingMetrics(
            step=sum(context.latest_metrics.step for context in contexts),
            episode=sum(context.latest_metrics.episode for context in contexts),
            reward_step=0.0 if reward_step_mean is None else reward_step_mean,
            cumulative_reward=sum(context.latest_metrics.cumulative_reward for context in contexts),
            mean_reward=0.0 if episode_reward_mean is None else episode_reward_mean,
            episode_reward_mean=0.0 if episode_reward_mean is None else episode_reward_mean,
            success_rate=0.0 if success_rate is None else success_rate,
            episode_length_mean=0.0 if episode_length_mean is None else episode_length_mean,
            fps=sum(context.latest_metrics.fps for context in contexts),
            exploration_rate=0.0 if exploration_rate is None else exploration_rate,
            value_loss=value_loss,
            policy_loss=policy_loss,
        )

    def _emit_status_if_changed(self) -> None:
        new_status = self._aggregate_status()
        if new_status == self._status:
            return
        self._status = new_status
        self.status_changed.emit(new_status)

    def _aggregate_status(self) -> TrainingStatus:
        contexts = self._session_contexts(live_only=False)
        if not contexts:
            return TrainingStatus.IDLE
        statuses = {context.status for context in contexts}
        if TrainingStatus.RUNNING in statuses:
            return TrainingStatus.RUNNING
        if TrainingStatus.PAUSED in statuses:
            return TrainingStatus.PAUSED
        if statuses == {TrainingStatus.FINISHED}:
            return TrainingStatus.FINISHED
        if statuses <= {TrainingStatus.FINISHED, TrainingStatus.STOPPED}:
            return TrainingStatus.STOPPED if TrainingStatus.STOPPED in statuses else TrainingStatus.FINISHED
        return TrainingStatus.IDLE

    def _session_contexts(self, *, live_only: bool) -> list[_RunContext]:
        contexts = [
            self._run_contexts[run_id]
            for run_id in self._current_session_run_ids
            if run_id in self._run_contexts
        ]
        if not live_only:
            return contexts
        return [
            context
            for context in contexts
            if context.status in {TrainingStatus.RUNNING, TrainingStatus.PAUSED}
        ]

    def _has_live_runs(self) -> bool:
        return bool(self._session_contexts(live_only=True))

    def _timestamp_now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _metrics_payload(self, metrics: TrainingMetrics) -> dict[str, float | int | None]:
        return {
            "step": metrics.step,
            "episode": metrics.episode,
            "reward_step": metrics.reward_step,
            "cumulative_reward": metrics.cumulative_reward,
            "mean_reward": metrics.mean_reward,
            "episode_reward_mean": metrics.episode_reward_mean,
            "success_rate": metrics.success_rate,
            "episode_length_mean": metrics.episode_length_mean,
            "fps": metrics.fps,
            "exploration_rate": metrics.exploration_rate,
            "value_loss": metrics.value_loss,
            "policy_loss": metrics.policy_loss,
        }

    def _resolve_initial_checkpoint(
        self,
        task: TaskDefinition,
        config: RunConfig,
        *,
        preferred_checkpoint: Checkpoint | None = None,
        start_from_scratch: bool = False,
    ) -> Checkpoint | None:
        if start_from_scratch or config.metadata.get("start_from_latest_checkpoint", True) is False:
            return None

        if self._checkpoint_is_compatible(preferred_checkpoint, task, config):
            return preferred_checkpoint

        for checkpoint in reversed(self._checkpoints):
            if self._checkpoint_is_compatible(checkpoint, task, config):
                return checkpoint
        return None

    def _checkpoint_is_compatible(
        self,
        checkpoint: Checkpoint | None,
        task: TaskDefinition,
        config: RunConfig,
    ) -> bool:
        if checkpoint is None:
            return False
        task_snapshot = checkpoint.task_snapshot
        if task_snapshot is None or task_snapshot.environment_id != task.environment_id:
            return False
        return checkpoint.metadata.get("algorithm") == config.algorithm

    def _checkpoint_index(self, checkpoint_id: str) -> int | None:
        for index, checkpoint in enumerate(self._checkpoints):
            if checkpoint.checkpoint_id == checkpoint_id:
                return index
        return None

    def _config_for_checkpoint(self, checkpoint: Checkpoint) -> RunConfig:
        metadata = checkpoint.metadata if isinstance(checkpoint.metadata, dict) else {}
        run_config_payload = metadata.get("run_config")
        if isinstance(run_config_payload, dict):
            return RunConfig.from_dict(run_config_payload)

        raw_seed = metadata.get("seed")
        try:
            seed = int(raw_seed) if raw_seed is not None else None
        except (TypeError, ValueError):
            seed = None
        return RunConfig(
            algorithm=str(metadata.get("algorithm", "q_learning")),
            seed=seed,
        )

    def _checkpoint_id_exists(self, checkpoint_id: str) -> bool:
        return any(checkpoint.checkpoint_id == checkpoint_id for checkpoint in self._checkpoints)

    def _next_available_checkpoint_id(self) -> str:
        counter = self._next_checkpoint_counter_floor() + 1
        while True:
            checkpoint_id = f"checkpoint_{counter:03d}"
            if not self._checkpoint_id_exists(checkpoint_id):
                return checkpoint_id
            counter += 1

    def _renamed_import_label(self, label: str, checkpoint_id: str) -> str:
        base_label = label.strip() if label else "Imported Checkpoint"
        suffix = f"imported as {checkpoint_id}"
        if suffix in base_label:
            return base_label
        return f"{base_label} | {suffix}"

    def _next_checkpoint_counter_floor(self) -> int:
        pattern = re.compile(r"^checkpoint_(\d+)$")
        max_counter = 0
        for checkpoint in self._checkpoints:
            match = pattern.match(checkpoint.checkpoint_id)
            if match is None:
                continue
            max_counter = max(max_counter, int(match.group(1)))
        return max_counter
