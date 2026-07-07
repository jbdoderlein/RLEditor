from __future__ import annotations

import os
import time

from gymnasium.spaces import Box, Discrete
import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.application.services import TaskService, TrainingHistorySnapshot, TrainingService
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


class _NoCreateBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="no_create_env",
            name="No Create Task",
            task_id="task_no_create",
        )


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


class _ContinuousEnv:
    def __init__(self) -> None:
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self._state = np.array([0.0, 0.0], dtype=np.float32)
        self._step_count = 0
        self.last_action = None

    def reset(self, *, seed: int | None = None):
        _ = seed
        self._state = np.array([0.0, 0.0], dtype=np.float32)
        self._step_count = 0
        return self._state.copy(), {}

    def step(self, action):
        self.last_action = action
        self._step_count += 1
        self._state = self._state + np.asarray(action, dtype=np.float32)
        terminated = self._step_count >= 2
        reward = float(np.sum(action))
        return self._state.copy(), reward, terminated, False, {}

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


def test_task_service_rebuilds_task_from_snapshot_with_copied_payloads() -> None:
    service = TaskService(PluginRegistry())
    snapshot = TaskSnapshot(
        environment_id="tiny_env",
        task_name="Snapshot Task",
        task_id="task_snapshot",
        task_config={"layout": ["S", "G"]},
        reward_config={"goal": 1.0},
        termination_config={"max_steps": 12},
        metadata={"tags": ["saved"]},
    )

    task = service.task_from_snapshot(snapshot)
    task.config["layout"].append("H")
    task.metadata["tags"].append("mutated")

    assert task.environment_id == "tiny_env"
    assert task.name == "Snapshot Task"
    assert task.task_id == "task_snapshot"
    assert snapshot.task_config["layout"] == ["S", "G"]
    assert snapshot.metadata["tags"] == ["saved"]


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


def test_training_runner_records_continuous_actions_for_random_policy() -> None:
    runner = TrainingRunner()
    task = TaskDefinition(environment_id="continuous", name="Continuous Task", task_id="task_continuous")
    config = RunConfig(
        algorithm="random",
        max_steps=10,
        max_episodes=1,
        seed=13,
        episode_trace_sample_rate=1.0,
    )

    captured: list[EpisodeTrace] = []
    runner.episode_captured.connect(captured.append)
    runner.start(
        task,
        config,
        run_id="run_continuous",
        env_factory=lambda _task: _ContinuousEnv(),
    )

    for _ in range(5):
        runner._on_tick()
        if runner.status == TrainingStatus.FINISHED:
            break

    assert runner.status == TrainingStatus.FINISHED
    assert captured
    assert isinstance(captured[0].steps[0].action, list)
    assert len(captured[0].steps[0].action) == 2
    assert captured[0].moments[1].action_taken == captured[0].steps[0].action


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


def test_training_service_evaluates_checkpoint_multiple_without_storing_results() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="Training Task", task_id="task_train")
    first_eval_task = TaskDefinition(environment_id="tiny_env", name="Eval A", task_id="task_eval_a")
    second_eval_task = TaskDefinition(environment_id="tiny_env", name="Eval B", task_id="task_eval_b")
    config = RunConfig(max_steps=20, max_episodes=1, seed=23)

    service.start(task, config)
    for _ in range(10):
        service._runner._on_tick()
        if service.status == TrainingStatus.FINISHED:
            break

    checkpoint_id = service.history_snapshot().checkpoints[-1].checkpoint_id
    rows = service.evaluate_checkpoint_multiple(
        checkpoint_id,
        [
            {
                "task": first_eval_task.to_dict(),
                "episode_count": 2,
                "max_steps_per_episode": 5,
                "seed": 101,
                "trace_sample_rate": 1.0,
            },
            {
                "task": second_eval_task.to_dict(),
                "episode_count": 1,
                "max_steps_per_episode": 5,
                "seed": 201,
                "trace_sample_rate": 1.0,
            },
        ],
    )
    snapshot = service.history_snapshot()
    checkpoint = snapshot.checkpoints[-1]

    assert [row["task_name"] for row in rows] == ["Eval A", "Eval B"]
    assert rows[0]["episode"] == 2
    assert rows[1]["episode"] == 1
    assert rows[0]["success_rate"] == 1.0
    assert "evaluation_metrics" not in checkpoint.metadata
    assert not any(run_id.startswith(f"eval_{checkpoint_id}") for run_id in snapshot.episodes_by_run)


def test_training_service_evaluate_checkpoint_rejects_unknown_or_missing_learner_state() -> None:
    service = TrainingService(_history_registry())

    with pytest.raises(RuntimeError, match="Unknown checkpoint"):
        service.evaluate_checkpoint("checkpoint_missing", {})

    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_no_learner",
        label="No Learner",
        created_at="now",
        reason="imported",
    )
    service.import_checkpoint(checkpoint)

    with pytest.raises(RuntimeError, match="does not contain a learner state"):
        service.evaluate_checkpoint("checkpoint_no_learner", {})


def test_training_service_delete_checkpoint_tree_purges_descendant_runs_and_traces() -> None:
    service = TrainingService(_history_registry())
    task_snapshot = TaskSnapshot(environment_id="tiny_env", task_name="Tiny Task", task_id="task_tiny")
    runs = [
        TrainingRun(run_id="run_001", task_id="task_tiny", status=TrainingStatus.FINISHED),
        TrainingRun(
            run_id="run_002",
            task_id="task_tiny",
            status=TrainingStatus.FINISHED,
            parent_checkpoint_id="checkpoint_001",
        ),
        TrainingRun(
            run_id="run_003",
            task_id="task_tiny",
            status=TrainingStatus.FINISHED,
            parent_checkpoint_id="checkpoint_002",
        ),
    ]
    checkpoints = [
        Checkpoint(
            checkpoint_id="checkpoint_001",
            label="Checkpoint 001",
            created_at="2026-06-11 10:00:00",
            reason="run_finished",
            run_id="run_001",
            task_snapshot=task_snapshot,
            metadata={"learner_state": {"q_values": [{"state_key": "0", "action": 0, "value": 1.0}]}},
        ),
        Checkpoint(
            checkpoint_id="checkpoint_002",
            label="Checkpoint 002",
            created_at="2026-06-11 10:01:00",
            reason="run_finished",
            parent_checkpoint_id="checkpoint_001",
            run_id="run_002",
            task_snapshot=task_snapshot,
            metadata={
                "learner_state": {"q_values": [{"state_key": "1", "action": 0, "value": 2.0}]},
                "evaluation": {"run_id": "eval_checkpoint_002"},
            },
        ),
        Checkpoint(
            checkpoint_id="checkpoint_003",
            label="Checkpoint 003",
            created_at="2026-06-11 10:02:00",
            reason="run_finished",
            parent_checkpoint_id="checkpoint_002",
            run_id="run_003",
            task_snapshot=task_snapshot,
            metadata={"learner_state": {"q_values": [{"state_key": "2", "action": 0, "value": 3.0}]}},
        ),
    ]
    service.load_history(
        TrainingHistorySnapshot(
            runs=runs,
            checkpoints=checkpoints,
            episodes_by_run={
                "run_001": [EpisodeTrace(episode_id=1, run_id="run_001", total_reward=1.0, success=True)],
                "run_002": [EpisodeTrace(episode_id=1, run_id="run_002", total_reward=2.0, success=True)],
                "run_003": [EpisodeTrace(episode_id=1, run_id="run_003", total_reward=3.0, success=True)],
                "eval_checkpoint_002": [
                    EpisodeTrace(
                        episode_id=1,
                        run_id="eval_checkpoint_002",
                        total_reward=4.0,
                        success=True,
                    )
                ],
            },
            run_task_snapshots={
                "run_001": task_snapshot,
                "run_002": task_snapshot,
                "run_003": task_snapshot,
                "eval_checkpoint_002": task_snapshot,
            },
        )
    )
    service._pending_episodes_by_run["run_002"] = [
        EpisodeTrace(episode_id=2, run_id="run_002", total_reward=2.0, success=True)
    ]
    service._pending_episodes_by_run["eval_checkpoint_002"] = [
        EpisodeTrace(
            episode_id=2,
            run_id="eval_checkpoint_002",
            total_reward=4.0,
            success=True,
        )
    ]

    deleted_ids = service.delete_checkpoint_tree(["checkpoint_002"])
    snapshot = service.history_snapshot()

    assert deleted_ids == ["checkpoint_002", "checkpoint_003"]
    assert [checkpoint.checkpoint_id for checkpoint in snapshot.checkpoints] == ["checkpoint_001"]
    assert [run.run_id for run in snapshot.runs] == ["run_001"]
    assert set(snapshot.episodes_by_run) == {"run_001"}
    assert set(service._pending_episodes_by_run) == {"run_001"}
    assert set(snapshot.run_task_snapshots) == {"run_001"}
    assert "run_002" not in service._runs_by_id
    assert "run_003" not in service._runs_by_id


def test_training_service_records_evaluation_error_and_clears_stale_eval_traces() -> None:
    service = TrainingService(_history_registry())
    evaluation_task = TaskDefinition(environment_id="tiny_env", name="Bad Evaluation Task", task_id="task_eval_bad")
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_bad_eval",
        label="Bad Eval",
        created_at="now",
        reason="imported",
        metadata={
            "algorithm": "q_learning",
            "learner_state": {
                "algorithm": "q_learning",
                "q_values": [],
            },
        },
    )
    service.import_checkpoint(checkpoint)
    service._episodes_by_run["eval_checkpoint_bad_eval"] = [
        EpisodeTrace(episode_id=1, run_id="stale", total_reward=0.0, success=False)
    ]
    service._pending_episodes_by_run["eval_checkpoint_bad_eval"] = [
        EpisodeTrace(episode_id=2, run_id="stale", total_reward=0.0, success=False)
    ]
    service._run_task_snapshots["eval_checkpoint_bad_eval"] = TaskSnapshot(
        environment_id="tiny_env",
        task_name="Stale Eval",
    )

    with pytest.raises(RuntimeError, match="Evaluation episode count is invalid"):
        service.evaluate_checkpoint(
            "checkpoint_bad_eval",
            {
                "task": evaluation_task.to_dict(),
                "episode_count": "bad",
            },
        )

    snapshot = service.history_snapshot()
    evaluated_checkpoint = snapshot.checkpoints[-1]
    assert evaluated_checkpoint.metadata["evaluation_error"] == "Evaluation episode count is invalid."
    assert "eval_checkpoint_bad_eval" not in snapshot.episodes_by_run
    assert "eval_checkpoint_bad_eval" not in service._pending_episodes_by_run
    assert "eval_checkpoint_bad_eval" not in snapshot.run_task_snapshots


def test_training_service_reports_invalid_evaluation_seed_before_running_env() -> None:
    service = TrainingService(_history_registry())
    evaluation_task = TaskDefinition(environment_id="tiny_env", name="Seed Evaluation Task", task_id="task_eval_seed")
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_bad_seed",
        label="Bad Seed",
        created_at="now",
        reason="imported",
        metadata={
            "algorithm": "q_learning",
            "learner_state": {
                "algorithm": "q_learning",
                "q_values": [],
            },
        },
    )
    service.import_checkpoint(checkpoint)

    with pytest.raises(RuntimeError, match="Evaluation seed is invalid"):
        service.evaluate_checkpoint(
            "checkpoint_bad_seed",
            {
                "task": evaluation_task.to_dict(),
                "episode_count": 1,
                "seed": "bad",
            },
        )


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


def test_training_service_load_history_resets_live_state_and_copies_snapshot() -> None:
    service = TrainingService(_history_registry())
    task_snapshot = TaskSnapshot(
        environment_id="tiny_env",
        task_name="Loaded Task",
        task_id="task_loaded",
    )
    run = TrainingRun(
        run_id="run_loaded",
        task_id="task_loaded",
        status=TrainingStatus.FINISHED,
    )
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_007",
        label="Loaded Checkpoint",
        created_at="now",
        reason="loaded",
        run_id=run.run_id,
        task_id="task_loaded",
        task_snapshot=task_snapshot,
    )
    trace = EpisodeTrace(
        episode_id=1,
        run_id=run.run_id,
        total_reward=1.0,
        success=True,
        task_snapshot=task_snapshot,
    )
    snapshot = TrainingHistorySnapshot(
        runs=[run],
        checkpoints=[checkpoint],
        episodes_by_run={run.run_id: [trace]},
        run_task_snapshots={run.run_id: task_snapshot},
    )
    notifications: list[None] = []
    service.history_changed.connect(lambda: notifications.append(None))

    service.load_history(snapshot)
    run.status = TrainingStatus.STOPPED
    checkpoint.label = "Mutated"
    trace.total_reward = -1.0

    loaded = service.history_snapshot(deep=False)
    assert notifications
    assert service.status == TrainingStatus.IDLE
    assert service._checkpoint_counter == 7
    assert loaded.runs[0].status == TrainingStatus.FINISHED
    assert loaded.checkpoints[0].label == "Loaded Checkpoint"
    assert loaded.episodes_by_run["run_loaded"][0].total_reward == 1.0
    assert service._pending_episodes_by_run == {"run_loaded": []}


def test_training_service_renames_checkpoint_label_and_emits_history_changed() -> None:
    service = TrainingService(_history_registry())
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_001",
        label="Old name",
        created_at="2026-06-17 10:00:00",
        reason="test",
    )
    service.load_history(
        TrainingHistorySnapshot(
            runs=[],
            checkpoints=[checkpoint],
            episodes_by_run={},
            run_task_snapshots={},
        )
    )
    notifications: list[None] = []
    service.history_changed.connect(lambda: notifications.append(None))

    renamed = service.rename_checkpoint("checkpoint_001", "New name")

    assert renamed.label == "New name"
    assert service.history_snapshot().checkpoints[0].label == "New name"
    assert notifications == [None]


def test_training_service_rejects_history_and_checkpoint_import_while_live() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="Live Task", task_id="task_live")
    config = RunConfig(max_steps=20, seed=61)
    empty_snapshot = TrainingHistorySnapshot(
        runs=[],
        checkpoints=[],
        episodes_by_run={},
        run_task_snapshots={},
    )
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_001",
        label="Imported",
        created_at="now",
        reason="imported",
    )

    service.start(task, config)
    try:
        with pytest.raises(RuntimeError, match="history while training is active"):
            service.load_history(empty_snapshot)
        with pytest.raises(RuntimeError, match="checkpoint while training is active"):
            service.import_checkpoint(checkpoint)
    finally:
        service.stop()


def test_training_service_pause_resume_and_stop_update_aggregate_status() -> None:
    service = TrainingService(_history_registry())
    task = TaskDefinition(environment_id="tiny_env", name="Manual Control Task", task_id="task_control")
    config = RunConfig(max_steps=20, seed=67)

    service.start(task, config)
    assert service.status == TrainingStatus.RUNNING

    service.pause()
    assert service.status == TrainingStatus.PAUSED

    service.resume()
    assert service.status == TrainingStatus.RUNNING

    service.stop()
    assert service.status == TrainingStatus.STOPPED


def test_training_service_start_noops_for_empty_task_list() -> None:
    service = TrainingService(_history_registry())

    service.start_many([], RunConfig(max_steps=5))

    assert service.status == TrainingStatus.IDLE
    assert service.history_snapshot().runs == []


def test_training_service_rejects_start_without_registry_or_backend_factory() -> None:
    task = TaskDefinition(environment_id="tiny_env", name="No Registry Task", task_id="task_no_registry")

    with pytest.raises(RuntimeError, match="no environment registry"):
        TrainingService().start(task, RunConfig(max_steps=5))

    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="no_create_env",
            display_name="No Create Env",
            description="Missing create_env test plugin",
            backend=_NoCreateBackend(),
            gui_extension=None,
        )
    )

    with pytest.raises(RuntimeError, match="does not expose create_env"):
        TrainingService(registry).start(
            TaskDefinition(environment_id="no_create_env", name="No Create Task", task_id="task_no_create"),
            RunConfig(max_steps=5),
        )


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
