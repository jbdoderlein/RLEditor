from __future__ import annotations

from collections import deque

import pytest
from gymnasium.spaces import Discrete

from rleditor.core.models import (
    Breakpoint,
    Checkpoint,
    RunConfig,
    TaskDefinition,
    TaskSnapshot,
    TrainingMetrics,
    TrainingStatus,
)
from rleditor.infra.training_runner import TrainingRunner


class _NeverDoneDiscreteEnv:
    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(1)
        self.closed = False

    def reset(self, *, seed: int | None = None):
        _ = seed
        return 0, {}

    def step(self, action: int):
        _ = action
        return 0, 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


class _ResetFailureEnv(_NeverDoneDiscreteEnv):
    def reset(self, *, seed: int | None = None):
        _ = seed
        raise RuntimeError("reset failed")


class _InvalidStepEnv(_NeverDoneDiscreteEnv):
    def step(self, action: int):
        _ = action
        return "invalid-step"


class _SampleFallbackActionSpace:
    n = 2

    def sample(self):
        return object()


class _SampleFallbackEnv:
    action_space = _SampleFallbackActionSpace()
    observation_space = Discrete(1)

    def __init__(self) -> None:
        self.actions: list[int] = []
        self.closed = False

    def reset(self, *, seed: int | None = None):
        _ = seed
        return 0, {}

    def step(self, action: int):
        self.actions.append(int(action))
        return 0, 1.0, True, False, {"is_success": True}

    def close(self) -> None:
        self.closed = True


def _task() -> TaskDefinition:
    return TaskDefinition(environment_id="runner_env", name="Runner Task", task_id="task_runner")


def test_training_runner_start_background_is_noop_until_running() -> None:
    runner = TrainingRunner()

    runner.start_background()

    assert runner.status == TrainingStatus.IDLE
    assert runner._worker_thread is None


def test_training_runner_closes_env_and_raises_when_reset_fails() -> None:
    runner = TrainingRunner()
    env = _ResetFailureEnv()

    with pytest.raises(RuntimeError, match="could not be reset"):
        runner.start(
            _task(),
            RunConfig(max_steps=1),
            run_id="run_reset_failure",
            env_factory=lambda _task: env,
        )

    assert runner.status == TrainingStatus.IDLE
    assert env.closed is True


def test_training_runner_invalid_step_result_stops_and_cleans_up_env() -> None:
    runner = TrainingRunner()
    env = _InvalidStepEnv()

    runner.start(
        _task(),
        RunConfig(max_steps=5),
        run_id="run_invalid_step",
        env_factory=lambda _task: env,
    )
    runner._on_tick()

    assert runner.status == TrainingStatus.STOPPED
    assert env.closed is True


def test_training_runner_non_q_algorithm_sample_fallback_uses_zero_action() -> None:
    runner = TrainingRunner()
    env = _SampleFallbackEnv()

    runner.start(
        _task(),
        RunConfig(algorithm="sample_policy", max_steps=5, max_episodes=1),
        run_id="run_sample_fallback",
        env_factory=lambda _task: env,
    )
    runner._on_tick()

    assert runner.status == TrainingStatus.FINISHED
    assert env.actions == [0]
    assert runner._metrics.cumulative_reward == pytest.approx(1.0)
    assert runner.export_learner_state() == {}


def test_training_runner_restores_only_valid_q_values_from_checkpoint() -> None:
    runner = TrainingRunner()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_q_restore",
        label="Q Restore",
        created_at="now",
        reason="test",
        task_snapshot=TaskSnapshot(environment_id="runner_env", task_name="Runner Task"),
        metadata={
            "algorithm": "q_learning",
            "learner_state": {
                "algorithm": "q_learning",
                "q_values": [
                    {"state_key": "0", "action": "1", "value": "2.5"},
                    {"state_key": "bad", "action": "x", "value": 1.0},
                    ["not", "a", "mapping"],
                ],
            },
        },
    )

    runner.start(
        _task(),
        RunConfig(max_steps=5, epsilon=0.0, hyperparameters={"epsilon": 0.0}),
        run_id="run_restore_filter",
        env_factory=lambda _task: _NeverDoneDiscreteEnv(),
        initial_checkpoint=checkpoint,
    )

    assert runner._q_values == {("0", 1): 2.5}
    runner.stop()


def test_training_runner_rule_matching_covers_all_breakpoint_metrics() -> None:
    runner = TrainingRunner()
    runner._metrics = TrainingMetrics(
        step=10,
        episode=3,
        mean_reward=1.2,
        episode_reward_mean=1.1,
        success_rate=0.75,
        exploration_rate=0.05,
        value_loss=0.01,
        policy_loss=0.02,
    )
    runner._episode_rewards = deque([0.0, 2.0, 4.0], maxlen=100)

    matching_rules = [
        Breakpoint(kind="max_step", value=10),
        Breakpoint(kind="episode_count_gte", value=3),
        Breakpoint(kind="mean_reward_gte", value=2.0, window=2),
        Breakpoint(kind="episode_reward_mean_gte", value=1.0),
        Breakpoint(kind="success_rate_gte", value=0.7),
        Breakpoint(kind="exploration_lte", value=0.05),
        Breakpoint(kind="value_loss_lte", value=0.01),
        Breakpoint(kind="policy_loss_lte", value=0.02),
    ]

    assert all(runner._rule_matches(rule) for rule in matching_rules)
    assert runner._windowed_reward_mean(0) == pytest.approx(1.2)
    assert runner._windowed_reward_mean(2) == pytest.approx(3.0)
    assert runner._rule_matches(Breakpoint(kind="unknown", value=0.0)) is False


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (Breakpoint(kind="max_step", value=10), "Breakpoint hit: max_step >= 10"),
        (Breakpoint(kind="episode_count_gte", value=3), "Breakpoint hit: episode_count >= 3"),
        (Breakpoint(kind="mean_reward_gte", value=1.25), "Breakpoint hit: mean_reward >= 1.250"),
        (
            Breakpoint(kind="episode_reward_mean_gte", value=1.5),
            "Breakpoint hit: episode_reward_mean >= 1.500",
        ),
        (Breakpoint(kind="success_rate_gte", value=0.8), "Breakpoint hit: success_rate >= 80.0%"),
        (Breakpoint(kind="exploration_lte", value=0.1), "Breakpoint hit: exploration_rate <= 10.0%"),
        (Breakpoint(kind="value_loss_lte", value=0.01), "Breakpoint hit: value_loss <= 0.010"),
        (Breakpoint(kind="policy_loss_lte", value=0.02), "Breakpoint hit: policy_loss <= 0.020"),
        (Breakpoint(kind="custom_metric", value=1.0), "Breakpoint hit: custom_metric"),
    ],
)
def test_training_runner_breakpoint_messages(rule: Breakpoint, expected: str) -> None:
    runner = TrainingRunner()

    assert runner._build_breakpoint_message(rule) == expected
