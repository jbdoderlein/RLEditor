from __future__ import annotations

import os
import time

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.application.services import TaskService, TrainingService
from rleditor.core.models import (
    Breakpoint,
    Checkpoint,
    EpisodeTrace,
    RunConfig,
    TaskDefinition,
    TaskDerivationOptions,
    TaskSnapshot,
    TrainingMetrics,
    TrainingRun,
    TrainingStatus,
)
from rleditor.infra.training_runner import TrainingRunner
from rleditor.plugins.base import EnvironmentPlugin
from rleditor.plugins.builtin.frozen_lake import build_frozen_lake_plugin
from rleditor.plugins.registry import PluginRegistry


class _DummyBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="dummy_env",
            name="Dummy Task",
            task_id="task_default",
            config={"difficulty": 1},
        )

    def create_env(self, task: TaskDefinition):
        return object()


class _TinyBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="tiny_env",
            name="Tiny Task",
            task_id="task_tiny",
            config={"difficulty": 1},
        )

    def create_env(self, task: TaskDefinition):
        _ = task
        return _TinyEnv()


class _TinyEnv:
    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(4)
        self._state = 0

    def reset(self, *, seed: int | None = None):
        _ = seed
        self._state = 0
        return self._state, {}

    def step(self, action: int):
        _ = action
        self._state = min(3, self._state + 1)
        terminated = self._state >= 3
        reward = 1.0 if terminated else 0.0
        return self._state, reward, terminated, False, {}

    def close(self) -> None:
        return


class _TinyGymEnv(gym.Env):
    action_space = Discrete(2)
    observation_space = Discrete(4)

    def __init__(self) -> None:
        super().__init__()
        self._state = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        _ = options
        self._state = 0
        return self._state, {}

    def step(self, action: int):
        _ = action
        self._state = min(3, self._state + 1)
        terminated = self._state >= 3
        reward = 1.0 if terminated else 0.0
        return self._state, reward, terminated, False, {}


class _TinyGymBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="tiny_gym_env",
            name="Tiny Gym Task",
            task_id="task_tiny_gym",
        )

    def create_env(self, task: TaskDefinition):
        _ = task
        return _TinyGymEnv()


class _TinyContinuousGymEnv(gym.Env):
    action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    observation_space = Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)

    def __init__(self) -> None:
        super().__init__()
        self._state = 0.0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        _ = options
        self._state = 0.0
        return np.array([self._state, 0.0], dtype=np.float32), {}

    def step(self, action):
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self._state += max(-1.0, min(1.0, action_value)) + 0.5
        terminated = self._state >= 1.5
        reward = -abs(1.5 - self._state)
        observation = np.array([self._state, action_value], dtype=np.float32)
        return observation, reward, terminated, False, {}


class _TinyContinuousGymBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="tiny_continuous_gym_env",
            name="Tiny Continuous Gym Task",
            task_id="task_tiny_continuous_gym",
        )

    def create_env(self, task: TaskDefinition):
        _ = task
        return _TinyContinuousGymEnv()


class _NeverDoneEnv:
    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(4)
        self._state = 0

    def reset(self, *, seed: int | None = None):
        _ = seed
        self._state = 0
        return self._state, {}

    def step(self, action: int):
        _ = action
        self._state = (self._state + 1) % 4
        return self._state, 0.25, False, False, {}

    def close(self) -> None:
        return


class _RestorableTinyEnv(_TinyEnv):
    def __init__(self) -> None:
        super().__init__()
        self._last_action: int | None = None

    def step(self, action: int):
        self._last_action = int(action)
        return super().step(action)

    def export_state(self):
        return {
            "state_index": self._state,
            "last_action": self._last_action,
        }


class _VariableTinyBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="variable_tiny_env",
            name="Variable Tiny Task",
            task_id="task_variable",
            config={"target_state": 3},
        )

    def create_env(self, task: TaskDefinition):
        target_state = int(task.config.get("target_state", 3))
        return _VariableTinyEnv(target_state=target_state)


class _VariableTinyEnv:
    def __init__(self, *, target_state: int) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(max(2, target_state + 1))
        self._target_state = max(1, target_state)
        self._state = 0

    def reset(self, *, seed: int | None = None):
        _ = seed
        self._state = 0
        return self._state, {}

    def step(self, action: int):
        _ = action
        self._state = min(self._target_state, self._state + 1)
        terminated = self._state >= self._target_state
        reward = 1.0 if terminated else 0.0
        return self._state, reward, terminated, False, {}

    def close(self) -> None:
        return


def _history_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="tiny_env",
            display_name="Tiny Env",
            description="History test plugin",
            backend=_TinyBackend(),
            gui_extension=None,
        )
    )
    return registry


def _sb3_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="tiny_gym_env",
            display_name="Tiny Gym Env",
            description="SB3 training test plugin",
            backend=_TinyGymBackend(),
            gui_extension=None,
        )
    )
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="tiny_continuous_gym_env",
            display_name="Tiny Continuous Gym Env",
            description="SB3 continuous training test plugin",
            backend=_TinyContinuousGymBackend(),
            gui_extension=None,
        )
    )
    return registry


def _frozen_lake_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_environment(build_frozen_lake_plugin())
    return registry


def _parallel_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="variable_tiny_env",
            display_name="Variable Tiny Env",
            description="Parallel training test plugin",
            backend=_VariableTinyBackend(),
            gui_extension=None,
        )
    )
    return registry


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_for(
    predicate,
    *,
    timeout_seconds: float = 1.0,
) -> None:
    app = _app()
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.001)
    assert predicate()


def test_task_service_create_default_task_uses_registry_plugin_backend() -> None:
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="dummy",
            display_name="Dummy",
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )

    service = TaskService(registry)
    task = service.create_default_task("dummy")

    assert task.environment_id == "dummy_env"
    assert task.task_id == "task_default"


def test_task_service_derive_task_propagates_lineage_and_metadata() -> None:
    service = TaskService(PluginRegistry())
    source = TaskDefinition(
        environment_id="frozen_lake",
        name="Main",
        task_id="task_main",
        config={"size": 4},
        reward_config={"tile:G": 1.0},
        termination_config={"max_steps": 64},
        metadata={"owner": "research"},
    )

    options = TaskDerivationOptions(
        config_updates={"size": 6},
        reward_config_updates={"tile:H": -2.0},
        termination_config_updates={"max_steps": 24},
        derivation_reason="focus_failure",
        source_episode_id=12,
        source_moment_index=11,
        source_run_id="run_alpha",
    )

    derived_task = service.derive_task(source, name="Derived Task A", options=options)

    assert derived_task.parent_task_id == "task_main"
    assert derived_task.derivation_reason == "focus_failure"
    assert derived_task.source_episode_id == 12
    assert derived_task.source_moment_index == 11
    assert derived_task.source_run_id == "run_alpha"
    assert derived_task.config["size"] == 6
    assert derived_task.reward_config["tile:H"] == -2.0
    assert derived_task.termination_config["max_steps"] == 24
    assert derived_task.metadata["owner"] == "research"
    assert derived_task.metadata["derived_from"] == "task_main"


def test_training_runner_adds_run_id_and_seed_to_episode_trace_task_snapshot() -> None:
    runner = TrainingRunner()
    task = TaskDefinition(environment_id="dummy", name="Task", task_id="task_1")
    config = RunConfig(max_steps=20, seed=123)

    captured: list = []
    runner.episode_captured.connect(captured.append)

    runner.start(
        task,
        config,
        run_id="run_abc",
        env_factory=lambda _task: _TinyEnv(),
    )

    for _ in range(10):
        runner._on_tick()
        if captured:
            break

    assert captured, "Expected at least one emitted episode trace"
    trace = captured[0]

    runner.stop()

    assert trace.run_id == "run_abc"
    assert trace.task_snapshot is not None
    assert trace.task_snapshot.task_id == "task_1"
    assert trace.task_snapshot.metadata["seed"] == 123
    assert trace.task_snapshot.metadata["run_id"] == "run_abc"


def test_training_runner_requires_explicit_environment_factory() -> None:
    runner = TrainingRunner()
    task = TaskDefinition(environment_id="dummy", name="Task", task_id="task_1")
    config = RunConfig(max_steps=20, seed=123)

    with pytest.raises(RuntimeError, match="no environment factory"):
        runner.start(task, config, run_id="run_missing_env")

    assert runner.status == TrainingStatus.IDLE


def test_training_runner_records_restorable_episode_moments_when_supported() -> None:
    runner = TrainingRunner()
    task = TaskDefinition(environment_id="dummy", name="Task", task_id="task_1")
    config = RunConfig(max_steps=20, max_episodes=1, seed=11, episode_trace_sample_rate=1.0)

    captured: list = []
    runner.episode_captured.connect(captured.append)

    runner.start(
        task,
        config,
        run_id="run_restorable",
        env_factory=lambda _task: _RestorableTinyEnv(),
    )

    for _ in range(10):
        runner._on_tick()
        if captured:
            break

    assert captured, "Expected at least one emitted episode trace"
    trace = captured[0]

    runner.stop()

    assert len(trace.moments) == len(trace.steps) + 1
    assert trace.moments[0].moment_index == 0
    assert trace.moments[0].restorable_env_state is not None
    assert trace.moments[-1].restorable_env_state is not None
    assert trace.metadata["restorable_state_captured"] is True


def test_training_runner_skips_episode_trace_capture_when_sample_rate_is_zero() -> None:
    runner = TrainingRunner()
    task = TaskDefinition(environment_id="dummy", name="Task", task_id="task_1")
    config = RunConfig(max_steps=20, max_episodes=1, seed=5, episode_trace_sample_rate=0.0)

    captured: list = []
    metrics: list = []
    runner.episode_captured.connect(captured.append)
    runner.metrics_updated.connect(metrics.append)

    runner.start(
        task,
        config,
        run_id="run_no_trace",
        env_factory=lambda _task: _RestorableTinyEnv(),
    )

    for _ in range(10):
        runner._on_tick()
        if runner.status == TrainingStatus.FINISHED:
            break

    assert runner.status == TrainingStatus.FINISHED
    assert captured == []
    assert metrics
    assert metrics[-1].episode == 1


def test_training_runner_emits_terminal_episode_before_breakpoint_pause() -> None:
    runner = TrainingRunner()
    task = TaskDefinition(environment_id="dummy", name="Task", task_id="task_1")
    config = RunConfig(
        max_steps=20,
        seed=123,
        breakpoints=[Breakpoint(kind="max_step", value=3)],
    )

    captured: list = []
    runner.episode_captured.connect(captured.append)

    runner.start(
        task,
        config,
        run_id="run_bp_terminal",
        env_factory=lambda _task: _TinyEnv(),
    )

    for _ in range(3):
        runner._on_tick()

    assert runner.status == TrainingStatus.PAUSED
    assert len(captured) == 1

    trace = captured[0]
    assert trace.steps, "Expected at least one step in captured episode"
    assert trace.steps[-1].terminated is True
    assert sum(1 for step in trace.steps if step.terminated or step.truncated) == 1

    runner.stop()


def test_training_runner_forces_episode_failure_at_max_steps_per_episode() -> None:
    runner = TrainingRunner()
    task = TaskDefinition(environment_id="dummy", name="Task", task_id="task_1")
    config = RunConfig(
        max_steps=20,
        max_episodes=1,
        max_steps_per_episode=2,
        seed=7,
    )

    captured: list = []
    metrics: list = []
    runner.episode_captured.connect(captured.append)
    runner.metrics_updated.connect(metrics.append)

    runner.start(
        task,
        config,
        run_id="run_max_step_episode",
        env_factory=lambda _task: _NeverDoneEnv(),
    )

    for _ in range(5):
        runner._on_tick()
        if captured:
            break

    assert captured, "Expected at least one emitted episode trace"
    trace = captured[0]
    assert len(trace.steps) == 2
    assert trace.success is False
    assert trace.steps[-1].truncated is True

    assert metrics, "Expected metrics updates to be emitted"
    latest_metrics = metrics[-1]
    assert latest_metrics.episode_reward_mean == trace.total_reward
    assert latest_metrics.mean_reward == trace.total_reward
    assert runner.status == TrainingStatus.FINISHED


def test_training_runner_allows_unlimited_total_steps() -> None:
    runner = TrainingRunner()
    task = TaskDefinition(environment_id="dummy", name="Task", task_id="task_1")
    config = RunConfig(max_steps=-1, seed=123)

    runner.start(
        task,
        config,
        run_id="run_unlimited_steps",
        env_factory=lambda _task: _NeverDoneEnv(),
    )

    for _ in range(5):
        runner._on_tick()

    assert config.max_steps is None
    assert runner.status == TrainingStatus.RUNNING

    runner.stop()


def test_training_runner_stop_breakpoint_stops_run_without_pause() -> None:
    runner = TrainingRunner()
    task = TaskDefinition(environment_id="dummy", name="Task", task_id="task_1")
    config = RunConfig(
        max_steps=20,
        seed=123,
        breakpoints=[Breakpoint(kind="max_step", value=2, actions=["stop"])],
    )

    runner.start(
        task,
        config,
        run_id="run_bp_stop",
        env_factory=lambda _task: _NeverDoneEnv(),
    )

    for _ in range(3):
        runner._on_tick()
        if runner.status == TrainingStatus.STOPPED:
            break

    assert runner.status == TrainingStatus.STOPPED
    runner.stop()


def test_training_service_exposes_checkpoint_history_snapshot() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="History Task", task_id="task_history")
    config = RunConfig(max_steps=20, max_episodes=1, seed=19)

    history_notifications: list[None] = []
    service.history_changed.connect(lambda: history_notifications.append(None))

    service.start(task, config)

    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    snapshot = service.history_snapshot()

    assert history_notifications
    assert len(snapshot.runs) == 1
    run = snapshot.runs[0]
    assert run.status == TrainingStatus.FINISHED
    assert run.run_id in snapshot.episodes_by_run
    assert len(snapshot.episodes_by_run[run.run_id]) == 1
    assert run.run_id in snapshot.run_task_snapshots
    assert snapshot.run_task_snapshots[run.run_id].task_name == "History Task"
    assert len(snapshot.checkpoints) == 1
    checkpoint = snapshot.checkpoints[0]
    assert checkpoint.run_id == run.run_id
    assert checkpoint.metadata["training_metrics"]["episode"] == 1


def test_training_service_evaluates_checkpoint_with_recorded_traces() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="Training Task", task_id="task_train")
    evaluation_task = TaskDefinition(environment_id="tiny_env", name="Evaluation Task", task_id="task_eval")
    config = RunConfig(
        max_steps=20,
        max_episodes=1,
        seed=23,
        evaluation_policy={
            "task": evaluation_task.to_dict(),
            "episode_count": 2,
            "max_steps_per_episode": 5,
            "trace_sample_rate": 1.0,
        },
    )

    service.start(task, config)
    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    snapshot = service.history_snapshot()
    checkpoint = snapshot.checkpoints[-1]
    evaluation = checkpoint.metadata["evaluation"]
    evaluation_run_id = evaluation["run_id"]
    evaluation_episodes = snapshot.episodes_by_run[evaluation_run_id]

    assert checkpoint.metadata["evaluation_metrics"]["episode"] == 2
    assert checkpoint.metadata["evaluation_metrics"]["success_rate"] == 1.0
    assert evaluation["task_name"] == "Evaluation Task"
    assert evaluation["max_steps_per_episode"] == 5
    assert len(evaluation_episodes) == 2
    assert all(trace.metadata["runner"] == "evaluation" for trace in evaluation_episodes)
    assert all(len(trace.moments) == len(trace.steps) + 1 for trace in evaluation_episodes)


def test_training_service_manually_evaluates_selected_checkpoint_with_fixed_seed() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="Training Task", task_id="task_train")
    evaluation_task = TaskDefinition(environment_id="tiny_env", name="Manual Evaluation Task", task_id="task_eval")
    config = RunConfig(max_steps=20, max_episodes=1, seed=23)

    service.start(task, config)
    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    checkpoint_id = service.history_snapshot().checkpoints[-1].checkpoint_id
    evaluated_checkpoint = service.evaluate_checkpoint(
        checkpoint_id,
        {
            "task": evaluation_task.to_dict(),
            "episode_count": 2,
            "max_steps_per_episode": 5,
            "seed": 101,
            "trace_sample_rate": 1.0,
        },
    )
    snapshot = service.history_snapshot()
    evaluation = evaluated_checkpoint.metadata["evaluation"]
    evaluation_run_id = evaluation["run_id"]
    evaluation_episodes = snapshot.episodes_by_run[evaluation_run_id]

    assert evaluation["task_name"] == "Manual Evaluation Task"
    assert evaluation["seed"] == 101
    assert evaluated_checkpoint.metadata["evaluation_metrics"]["episode"] == 2
    assert [trace.metadata["seed"] for trace in evaluation_episodes] == [101, 102]


def test_training_service_raises_without_simulated_fallback_for_unknown_env() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="missing_env", name="Missing Env Task", task_id="task_missing")
    config = RunConfig(max_steps=20, max_episodes=1, seed=19)

    with pytest.raises(RuntimeError, match="unknown environment"):
        service.start(task, config)

    snapshot = service.history_snapshot()
    assert snapshot.runs == []
    assert snapshot.checkpoints == []


def test_training_service_raises_without_simulated_fallback_for_invalid_env() -> None:
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="invalid_env",
            display_name="Invalid Env",
            description="Invalid env test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )
    service = TrainingService(registry)
    task = TaskDefinition(environment_id="invalid_env", name="Invalid Env Task", task_id="task_invalid")
    config = RunConfig(max_steps=20, max_episodes=1, seed=19)

    with pytest.raises(RuntimeError, match="Gymnasium-compatible"):
        service.start(task, config)

    snapshot = service.history_snapshot()
    assert snapshot.runs == []
    assert snapshot.checkpoints == []


def test_training_service_streams_metrics_but_flushes_episodes_only_when_run_reaches_boundary() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="Buffered Task", task_id="task_buffered")
    config = RunConfig(max_steps=20, max_episodes=1, seed=21)

    streamed_metrics: list[TrainingMetrics] = []
    flushed_episodes: list[EpisodeTrace] = []
    service.metrics_updated.connect(streamed_metrics.append)
    service.episode_captured.connect(flushed_episodes.append)

    service.start(task, config)
    run_id = service.history_snapshot().runs[0].run_id

    service._runner._on_tick()
    service._runner._on_tick()

    mid_snapshot = service.history_snapshot()
    assert streamed_metrics, "Expected live metrics before the run finishes"
    assert flushed_episodes == []
    assert mid_snapshot.episodes_by_run[run_id] == []

    service._runner._on_tick()

    final_snapshot = service.history_snapshot()
    assert len(flushed_episodes) == 1
    assert len(final_snapshot.episodes_by_run[run_id]) == 1


def test_training_service_history_snapshot_can_skip_deepcopy_for_ui_refresh() -> None:
    service = TrainingService(_history_registry())
    task_snapshot = TaskSnapshot(
        environment_id="tiny_env",
        task_name="History Task",
        task_id="task_history",
    )
    run = TrainingRun(run_id="run_history", task_id="task_history")
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_001",
        label="Checkpoint 001",
        created_at="now",
        reason="run_stopped",
        run_id=run.run_id,
        task_id="task_history",
        task_snapshot=task_snapshot,
    )
    trace = EpisodeTrace(
        episode_id=1,
        run_id=run.run_id,
        total_reward=1.0,
        success=True,
        task_snapshot=task_snapshot,
    )

    service._runs = [run]
    service._checkpoints = [checkpoint]
    service._episodes_by_run = {run.run_id: [trace]}
    service._run_task_snapshots = {run.run_id: task_snapshot}

    shallow_snapshot = service.history_snapshot(deep=False)
    deep_snapshot = service.history_snapshot()

    assert shallow_snapshot.runs[0] is run
    assert shallow_snapshot.checkpoints[0] is checkpoint
    assert shallow_snapshot.episodes_by_run[run.run_id][0] is trace
    assert shallow_snapshot.run_task_snapshots[run.run_id] is task_snapshot
    assert deep_snapshot.runs[0] is not run
    assert deep_snapshot.checkpoints[0] is not checkpoint
    assert deep_snapshot.episodes_by_run[run.run_id][0] is not trace
    assert deep_snapshot.run_task_snapshots[run.run_id] is not task_snapshot


def test_training_service_chains_checkpoints_within_one_run() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="History Task", task_id="task_history")
    config = RunConfig(
        max_steps=20,
        max_episodes=1,
        seed=23,
        breakpoints=[Breakpoint(kind="max_step", value=1, actions=["checkpoint"])],
    )

    service.start(task, config)

    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    snapshot = service.history_snapshot()
    run_checkpoints = [checkpoint for checkpoint in snapshot.checkpoints if checkpoint.run_id == snapshot.runs[0].run_id]

    assert len(run_checkpoints) == 2
    assert run_checkpoints[0].parent_checkpoint_id is None
    assert run_checkpoints[1].parent_checkpoint_id == run_checkpoints[0].checkpoint_id


def test_training_service_pause_breakpoint_can_checkpoint_before_run_stops() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="History Task", task_id="task_history")
    config = RunConfig(
        max_steps=20,
        seed=29,
        breakpoints=[Breakpoint(kind="max_step", value=3, actions=["pause", "checkpoint"])],
    )

    service.start(task, config)

    for _ in range(3):
        service._runner._on_tick()

    snapshot = service.history_snapshot()

    assert service.status == TrainingStatus.PAUSED
    assert len(snapshot.runs) == 1
    assert len(snapshot.checkpoints) == 1
    assert len(snapshot.episodes_by_run[snapshot.runs[0].run_id]) == 1
    assert snapshot.checkpoints[0].reason == "breakpoint_max_step"
    assert snapshot.checkpoints[0].run_id == snapshot.runs[0].run_id


def test_training_service_reuses_latest_checkpoint_for_next_run_on_same_environment() -> None:
    service = TrainingService(_history_registry())
    base_task = TaskDefinition(environment_id="tiny_env", name="Main Task", task_id="task_main")
    first_config = RunConfig(max_steps=20, max_episodes=1, seed=31)

    service.start(base_task, first_config)
    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    first_snapshot = service.history_snapshot()
    first_checkpoint = first_snapshot.checkpoints[-1]
    learner_state = first_checkpoint.metadata.get("learner_state")
    assert isinstance(learner_state, dict)
    assert learner_state.get("q_values")

    derived_task = TaskDefinition(environment_id="tiny_env", name="Derived Task", task_id="task_derived")
    second_config = RunConfig(max_steps=20, max_episodes=1, seed=37)

    service.start(derived_task, second_config)

    second_snapshot = service.history_snapshot()
    second_run = second_snapshot.runs[-1]
    assert second_run.parent_checkpoint_id == first_checkpoint.checkpoint_id
    assert service._runner._q_values


def test_training_service_keeps_checkpoint_learner_state_in_memory_for_project_save() -> None:
    service = TrainingService(_history_registry())
    base_task = TaskDefinition(environment_id="tiny_env", name="Main Task", task_id="task_main")
    config = RunConfig(max_steps=20, max_episodes=1, seed=31)

    service.start(base_task, config)
    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    checkpoint = service.history_snapshot().checkpoints[-1]

    assert checkpoint.storage_uri is None
    assert checkpoint.metadata["learner_state"]["q_values"]


def test_training_service_imports_checkpoint_and_renames_duplicate_ids() -> None:
    service = TrainingService(_history_registry())
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_009",
        label="Imported Checkpoint",
        created_at="2026-05-04 10:00:00",
        reason="imported",
        run_id="run_imported",
        task_snapshot=TaskSnapshot(
            environment_id="tiny_env",
            task_name="Imported Tiny Task",
            task_id="task_imported",
        ),
        metadata={
            "algorithm": "q_learning",
            "learner_state": {
                "algorithm": "q_learning",
                "q_values": [],
            },
        },
    )

    first_import = service.import_checkpoint(checkpoint)
    snapshot = service.history_snapshot()

    assert first_import.checkpoint_id == "checkpoint_009"
    assert snapshot.checkpoints[-1].checkpoint_id == "checkpoint_009"
    assert service._checkpoint_counter == 9
    assert snapshot.run_task_snapshots["run_imported"].task_name == "Imported Tiny Task"

    second_import = service.import_checkpoint(checkpoint)
    snapshot = service.history_snapshot()

    assert second_import.checkpoint_id == "checkpoint_010"
    assert second_import.label == "Imported Checkpoint | imported as checkpoint_010"
    assert snapshot.checkpoints[-1].checkpoint_id == "checkpoint_010"
    assert service._checkpoint_counter == 10


def test_training_service_can_start_from_selected_checkpoint_instead_of_latest() -> None:
    service = TrainingService(_history_registry())
    config = RunConfig(max_steps=20, max_episodes=1, seed=41)

    service.start(TaskDefinition(environment_id="tiny_env", name="Task A", task_id="task_a"), config)
    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    first_checkpoint = service.history_snapshot().checkpoints[-1]

    service.start(TaskDefinition(environment_id="tiny_env", name="Task B", task_id="task_b"), config)
    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    latest_checkpoint = service.history_snapshot().checkpoints[-1]
    assert latest_checkpoint.checkpoint_id != first_checkpoint.checkpoint_id

    service.start(
        TaskDefinition(environment_id="tiny_env", name="Task C", task_id="task_c"),
        config,
        initial_checkpoint=first_checkpoint,
    )

    third_run = service.history_snapshot().runs[-1]
    assert third_run.parent_checkpoint_id == first_checkpoint.checkpoint_id


def test_training_service_can_explicitly_start_from_scratch() -> None:
    service = TrainingService(_history_registry())
    config = RunConfig(max_steps=20, max_episodes=1, seed=43)

    service.start(TaskDefinition(environment_id="tiny_env", name="Task A", task_id="task_a"), config)
    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    first_checkpoint = service.history_snapshot().checkpoints[-1]
    assert first_checkpoint.checkpoint_id

    service.start(
        TaskDefinition(environment_id="tiny_env", name="Task Scratch", task_id="task_scratch"),
        config,
        start_from_scratch=True,
    )

    latest_run = service.history_snapshot().runs[-1]
    assert latest_run.parent_checkpoint_id is None
    assert latest_run.metadata["started_from_scratch"] is True


def test_training_service_background_mode_completes_without_manual_ticks() -> None:
    app = _app()
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="Background Task", task_id="task_background")
    config = RunConfig(max_steps=20, max_episodes=1, seed=47)

    service.start(task, config, run_in_background=True)

    deadline = time.perf_counter() + 1.0
    snapshot = service.history_snapshot()
    while time.perf_counter() < deadline:
        app.processEvents()
        snapshot = service.history_snapshot()
        if (
            service.status == TrainingStatus.FINISHED
            and snapshot.runs
            and snapshot.runs[-1].status == TrainingStatus.FINISHED
        ):
            break
        time.sleep(0.001)

    assert service.status == TrainingStatus.FINISHED
    assert snapshot.runs[-1].status == TrainingStatus.FINISHED
    assert snapshot.checkpoints


def test_training_service_can_run_stable_baselines3_dqn_backend() -> None:
    service = TrainingService(_sb3_registry())
    task = TaskDefinition(
        environment_id="tiny_gym_env",
        name="SB3 Tiny Task",
        task_id="task_sb3_tiny",
    )
    config = RunConfig(
        algorithm="sb3_dqn",
        max_steps=6,
        seed=71,
        evaluation_policy={
            "task": task.to_dict(),
            "episode_count": 1,
            "max_steps_per_episode": 4,
            "trace_sample_rate": 1.0,
        },
        episode_trace_sample_rate=1.0,
        hyperparameters={
            "learning_starts": 0,
            "buffer_size": 32,
            "batch_size": 1,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 4,
            "exploration_fraction": 0.2,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
        },
    )

    service.start(task, config, run_in_background=True)

    _wait_for(lambda: service.status == TrainingStatus.FINISHED, timeout_seconds=10.0)
    snapshot = service.history_snapshot()

    assert snapshot.runs[-1].metadata["algorithm"] == "sb3_dqn"
    assert snapshot.runs[-1].metadata["run_config"]["algorithm"] == "sb3_dqn"
    assert snapshot.checkpoints[-1].metadata["algorithm"] == "sb3_dqn"
    assert snapshot.checkpoints[-1].metadata["learner_state"]["backend"] == "stable_baselines3"
    assert snapshot.checkpoints[-1].metadata["evaluation_metrics"]["episode"] == 1
    assert snapshot.episodes_by_run[snapshot.checkpoints[-1].metadata["evaluation"]["run_id"]]
    assert snapshot.episodes_by_run[snapshot.runs[-1].run_id]


def test_training_service_evaluates_sb3_dqn_on_frozen_lake_discrete_actions() -> None:
    service = TrainingService(_frozen_lake_registry())
    task = TaskDefinition(
        environment_id="frozen_lake",
        name="Frozen Lake Eval Regression",
        task_id="task_frozen_lake_eval_regression",
        config={
            "map_desc": [
                "SFG",
                "FFF",
                "FFF",
            ],
            "is_slippery": False,
            "success_rate": 1.0,
        },
    )
    config = RunConfig(
        algorithm="sb3_dqn",
        max_steps=4,
        seed=89,
        episode_trace_sample_rate=0.0,
        evaluation_policy={
            "task": task.to_dict(),
            "episode_count": 1,
            "max_steps_per_episode": 3,
            "trace_sample_rate": 1.0,
        },
        hyperparameters={
            "learning_starts": 0,
            "buffer_size": 32,
            "batch_size": 1,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 4,
            "exploration_fraction": 0.2,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
        },
    )

    service.start(task, config, run_in_background=True)

    _wait_for(lambda: service.status == TrainingStatus.FINISHED, timeout_seconds=10.0)
    checkpoint = service.history_snapshot().checkpoints[-1]

    assert "evaluation_error" not in checkpoint.metadata
    assert checkpoint.metadata["evaluation_metrics"]["episode"] == 1


def test_training_service_can_run_stable_baselines3_ppo_backend() -> None:
    service = TrainingService(_sb3_registry())
    task = TaskDefinition(
        environment_id="tiny_continuous_gym_env",
        name="SB3 Tiny Continuous Task",
        task_id="task_sb3_tiny_continuous",
    )
    config = RunConfig(
        algorithm="sb3_ppo",
        max_steps=8,
        seed=83,
        evaluation_policy={
            "task": task.to_dict(),
            "episode_count": 1,
            "max_steps_per_episode": 4,
            "trace_sample_rate": 1.0,
        },
        episode_trace_sample_rate=1.0,
        hyperparameters={
            "n_steps": 4,
            "batch_size": 4,
            "n_epochs": 1,
        },
    )

    service.start(task, config, run_in_background=True)

    _wait_for(lambda: service.status == TrainingStatus.FINISHED, timeout_seconds=10.0)
    snapshot = service.history_snapshot()
    run = snapshot.runs[-1]
    episode = snapshot.episodes_by_run[run.run_id][-1]

    assert run.metadata["algorithm"] == "sb3_ppo"
    assert snapshot.checkpoints[-1].metadata["algorithm"] == "sb3_ppo"
    assert snapshot.checkpoints[-1].metadata["learner_state"]["backend"] == "stable_baselines3"
    assert snapshot.checkpoints[-1].metadata["evaluation_metrics"]["episode"] == 1
    assert isinstance(episode.steps[-1].action, list)


def test_training_service_can_run_multiple_tasks_in_parallel() -> None:
    service = TrainingService(_parallel_registry())
    config = RunConfig(max_steps=20, max_episodes=1, seed=53)
    task_a = TaskDefinition(
        environment_id="variable_tiny_env",
        name="Task A",
        task_id="task_a",
        config={"target_state": 2},
    )
    task_b = TaskDefinition(
        environment_id="variable_tiny_env",
        name="Task B",
        task_id="task_b",
        config={"target_state": 4},
    )

    service.start_many([task_a, task_b], config, run_in_background=True)

    _wait_for(lambda: service.status == TrainingStatus.FINISHED)
    snapshot = service.history_snapshot()

    assert len(snapshot.runs) == 2
    assert {run.status for run in snapshot.runs} == {TrainingStatus.FINISHED}
    assert len(snapshot.checkpoints) == 2
    assert {checkpoint.task_name for checkpoint in snapshot.checkpoints} == {"Task A", "Task B"}
    assert all(len(snapshot.episodes_by_run[run.run_id]) == 1 for run in snapshot.runs)


def test_training_service_parallel_group_pauses_when_first_run_hits_breakpoint() -> None:
    service = TrainingService(_parallel_registry())
    config = RunConfig(
        max_steps=40,
        seed=59,
        breakpoints=[Breakpoint(kind="episode_count_gte", value=1, actions=["pause", "checkpoint"])],
    )
    fast_task = TaskDefinition(
        environment_id="variable_tiny_env",
        name="Fast Task",
        task_id="task_fast",
        config={"target_state": 1},
    )
    slow_task = TaskDefinition(
        environment_id="variable_tiny_env",
        name="Slow Task",
        task_id="task_slow",
        config={"target_state": 6},
    )

    breakpoint_messages: list[str] = []
    service.breakpoint_triggered.connect(lambda event: breakpoint_messages.append(event.message))

    service.start_many([fast_task, slow_task], config, run_in_background=True)

    _wait_for(lambda: service.status == TrainingStatus.PAUSED)
    snapshot = service.history_snapshot()
    runs_by_task = {run.metadata["task_name"]: run for run in snapshot.runs}

    assert breakpoint_messages
    triggered_task_name = breakpoint_messages[0].split(":", 1)[0]
    assert breakpoint_messages[0].endswith("Breakpoint hit: episode_count >= 1")
    assert triggered_task_name in {"Fast Task", "Slow Task"}
    assert runs_by_task["Fast Task"].status == TrainingStatus.PAUSED
    assert runs_by_task["Slow Task"].status == TrainingStatus.PAUSED
    assert any(checkpoint.task_name == triggered_task_name for checkpoint in snapshot.checkpoints)
    assert snapshot.episodes_by_run[runs_by_task[triggered_task_name].run_id]
