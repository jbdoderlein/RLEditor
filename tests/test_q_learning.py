from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from rleditor.core.models import Checkpoint, RunConfig, TaskDefinition, TaskSnapshot, TrainingStatus
from rleditor.infra.evaluation_runner import evaluate_policy
from rleditor.infra.training_runner import TrainingRunner


class _TwoStepChainEnv:
    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(3)
        self.actions: list[int] = []
        self._state = 0

    def reset(self, *, seed: int | None = None):
        _ = seed
        self._state = 0
        return self._state, {}

    def step(self, action: int):
        self.actions.append(int(action))
        if self._state == 0:
            self._state = 1
            return 1, 0.0, False, False, {}
        self._state = 2
        return 2, 1.0, True, False, {"is_success": True}

    def close(self) -> None:
        return


class _OneStepChoiceEnv:
    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(1)
        self.actions: list[int] = []

    def reset(self, *, seed: int | None = None):
        _ = seed
        return 0, {}

    def step(self, action: int):
        action = int(action)
        self.actions.append(action)
        reward = 2.0 if action == 1 else -1.0
        return 0, reward, True, False, {"is_success": action == 1}

    def close(self) -> None:
        return


class _NeverDoneDiscreteEnv:
    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(1)

    def reset(self, *, seed: int | None = None):
        _ = seed
        return 0, {}

    def step(self, action: int):
        _ = action
        return 0, 0.0, False, False, {}

    def close(self) -> None:
        return


class _ContinuousActionEnv:
    def __init__(self) -> None:
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Discrete(1)
        self.closed = False

    def reset(self, *, seed: int | None = None):
        _ = seed
        return 0, {}

    def step(self, action):
        _ = action
        return 0, 0.0, True, False, {}

    def close(self) -> None:
        self.closed = True


class _TieBreakingRandom:
    def __init__(self) -> None:
        self.choices: list[int] = []

    def random(self) -> float:
        return 1.0

    def choice(self, values: list[int]) -> int:
        self.choices = list(values)
        return self.choices[-1]


def _task() -> TaskDefinition:
    return TaskDefinition(environment_id="test_env", name="Q Test Task", task_id="task_q")


def _q_values_by_key(runner: TrainingRunner) -> dict[tuple[str, int], float]:
    learner_state = runner.export_learner_state()
    return {
        (str(entry["state_key"]), int(entry["action"])): float(entry["value"])
        for entry in learner_state["q_values"]
    }


def test_q_learning_runner_applies_bellman_update_for_bootstrap_and_terminal_steps() -> None:
    runner = TrainingRunner()
    env = _TwoStepChainEnv()
    config = RunConfig(
        max_steps=4,
        max_episodes=2,
        seed=123,
        learning_rate=0.5,
        gamma=0.9,
        epsilon=0.0,
        hyperparameters={
            "learning_rate": 0.5,
            "gamma": 0.9,
            "epsilon": 0.0,
            "epsilon_min": 0.0,
        },
    )

    runner.start(_task(), config, run_id="run_q_update", env_factory=lambda _task: env)
    runner._q_values[("0", 1)] = -1.0
    runner._q_values[("1", 1)] = -1.0
    for _ in range(4):
        runner._on_tick()

    q_values = _q_values_by_key(runner)

    assert runner.status == TrainingStatus.FINISHED
    assert env.actions == [0, 0, 0, 0]
    assert q_values[("1", 0)] == pytest.approx(0.75)
    assert q_values[("0", 0)] == pytest.approx(0.225)
    assert q_values[("0", 1)] == pytest.approx(-1.0)


def test_q_learning_exploitation_breaks_equal_q_ties_randomly() -> None:
    runner = TrainingRunner()
    env = _NeverDoneDiscreteEnv()
    config = RunConfig(
        algorithm="q_learning",
        epsilon=0.0,
        hyperparameters={"epsilon": 0.0, "epsilon_min": 0.0},
    )

    runner.start(_task(), config, run_id="run_q_tie_break", env_factory=lambda _task: env)
    random_source = _TieBreakingRandom()
    runner._random = random_source  # type: ignore[assignment]

    action = runner._select_action(0, env)

    assert random_source.choices == [0, 1]
    assert action == 1
    runner.stop()


def test_q_learning_runner_restores_checkpoint_and_exploits_best_known_action() -> None:
    runner = TrainingRunner()
    env = _OneStepChoiceEnv()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_q",
        label="Q checkpoint",
        created_at="now",
        reason="test",
        task_snapshot=TaskSnapshot(environment_id="test_env", task_name="Q Test Task"),
        metadata={
            "algorithm": "q_learning",
            "learner_state": {
                "algorithm": "q_learning",
                "q_values": [
                    {"state_key": "0", "action": 0, "value": -1.0},
                    {"state_key": "0", "action": 1, "value": 4.0},
                ],
            },
        },
    )
    config = RunConfig(
        max_steps=1,
        max_episodes=1,
        seed=321,
        learning_rate=0.0,
        epsilon=0.0,
        hyperparameters={
            "learning_rate": 0.0,
            "epsilon": 0.0,
            "epsilon_min": 0.0,
        },
    )

    runner.start(
        _task(),
        config,
        run_id="run_q_restore",
        env_factory=lambda _task: env,
        initial_checkpoint=checkpoint,
    )
    runner._on_tick()

    assert runner.status == TrainingStatus.FINISHED
    assert env.actions == [1]


def test_q_learning_runner_can_roll_out_without_updating_model() -> None:
    runner = TrainingRunner()
    env = _OneStepChoiceEnv()
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_q_no_update",
        label="Q checkpoint",
        created_at="now",
        reason="test",
        task_snapshot=TaskSnapshot(environment_id="test_env", task_name="Q Test Task"),
        metadata={
            "algorithm": "q_learning",
            "learner_state": {
                "algorithm": "q_learning",
                "q_values": [
                    {"state_key": "0", "action": 0, "value": -1.0},
                    {"state_key": "0", "action": 1, "value": 4.0},
                ],
            },
        },
    )
    config = RunConfig(
        algorithm="q_learning",
        max_steps=1,
        max_episodes=1,
        epsilon=0.0,
        hyperparameters={
            "epsilon": 0.0,
            "epsilon_min": 0.0,
            "disable_model_updates": True,
        },
        metadata={"disable_model_updates": True},
    )

    runner.start(
        _task(),
        config,
        run_id="run_q_no_update",
        env_factory=lambda _task: env,
        initial_checkpoint=checkpoint,
    )
    runner._on_tick()

    assert runner.status == TrainingStatus.FINISHED
    assert env.actions == [1]
    assert _q_values_by_key(runner) == {
        ("0", 0): pytest.approx(-1.0),
        ("0", 1): pytest.approx(4.0),
    }
    assert runner._metrics.cumulative_reward == pytest.approx(2.0)


def test_q_learning_evaluation_uses_greedy_action_from_exported_q_table() -> None:
    task = _task()
    config = RunConfig(algorithm="q_learning", max_steps=10)
    learner_state = {
        "algorithm": "q_learning",
        "q_values": [
            {"state_key": "0", "action": 0, "value": -1.0},
            {"state_key": "0", "action": 1, "value": 3.0},
        ],
    }

    result = evaluate_policy(
        task=task,
        config=config,
        learner_state=learner_state,
        env_factory=lambda _task: _OneStepChoiceEnv(),
        run_id="eval_q",
        episode_count=3,
        max_steps_per_episode=5,
        seed=77,
    )

    assert result.metrics.episode == 3
    assert result.metrics.success_rate == 1.0
    assert [trace.total_reward for trace in result.episodes] == [2.0, 2.0, 2.0]
    assert all(trace.steps[0].action == 1 for trace in result.episodes)


def test_q_learning_exploration_schedule_honors_decay_and_minimum() -> None:
    runner = TrainingRunner()
    config = RunConfig(
        max_steps=4,
        seed=11,
        epsilon=1.0,
        hyperparameters={
            "epsilon": 1.0,
            "epsilon_decay": 0.5,
            "epsilon_min": 0.25,
        },
    )

    runner.start(
        _task(),
        config,
        run_id="run_q_epsilon",
        env_factory=lambda _task: _NeverDoneDiscreteEnv(),
    )

    expected_rates = [1.0, 0.5, 0.25, 0.25]
    for expected_rate in expected_rates:
        runner._on_tick()
        assert runner._metrics.exploration_rate == pytest.approx(expected_rate)


def test_q_learning_exploration_schedule_decays_over_episode_budget() -> None:
    runner = TrainingRunner()
    config = RunConfig(
        max_steps=None,
        max_episodes=4,
        seed=13,
        epsilon=1.0,
        hyperparameters={
            "epsilon": 1.0,
            "epsilon_min": 0.25,
        },
    )

    runner.start(
        _task(),
        config,
        run_id="run_q_episode_epsilon",
        env_factory=lambda _task: _OneStepChoiceEnv(),
    )

    expected_rates = [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.25]
    for expected_rate in expected_rates:
        runner._on_tick()
        assert runner._metrics.exploration_rate == pytest.approx(expected_rate)


def test_q_learning_training_requires_discrete_action_space() -> None:
    runner = TrainingRunner()
    env = _ContinuousActionEnv()

    with pytest.raises(RuntimeError, match="Q-learning requires a discrete action_space\\.n"):
        runner.start(
            _task(),
            RunConfig(algorithm="q_learning", max_steps=1),
            run_id="run_q_continuous",
            env_factory=lambda _task: env,
        )

    assert runner.status == TrainingStatus.IDLE
    assert env.closed is True


def test_q_learning_evaluation_requires_discrete_action_space() -> None:
    with pytest.raises(ValueError, match="discrete action_space\\.n"):
        evaluate_policy(
            task=_task(),
            config=RunConfig(algorithm="q_learning", max_steps=1),
            learner_state={"algorithm": "q_learning", "q_values": []},
            env_factory=lambda _task: _ContinuousActionEnv(),
            run_id="eval_q_continuous",
            episode_count=1,
            max_steps_per_episode=1,
            seed=1,
        )
