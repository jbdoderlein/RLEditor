from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from rleditor.core.models import RunConfig, TaskDefinition
from rleditor.infra.evaluation_runner import (
    _action_space_size,
    _normalize_action_for_env,
    _q_values_from_learner_state,
    _state_key,
    evaluate_policy,
)


class _ClosedEnv:
    action_space = Discrete(2)

    def __init__(self) -> None:
        self.closed = False

    def reset(self, *, seed: int | None = None):
        _ = seed
        return 0, {}

    def step(self, action: int):
        _ = action
        return 0, 0.0, True, False, {}

    def close(self) -> None:
        self.closed = True


class _OldGymApiEnv:
    action_space = Discrete(2)

    def __init__(self) -> None:
        self.reset_called_without_seed = False
        self.actions: list[int] = []

    def reset(self):
        self.reset_called_without_seed = True
        return 0

    def step(self, action: int):
        self.actions.append(int(action))
        return 1, 1.0, True, {"is_success": True}

    def close(self) -> None:
        return


class _NeverDoneOldGymApiEnv:
    action_space = Discrete(2)

    def __init__(self) -> None:
        self.actions: list[int] = []

    def reset(self, *, seed: int | None = None):
        _ = seed
        return 0, {}

    def step(self, action: int):
        self.actions.append(int(action))
        return 0, 0.25, False, {}

    def close(self) -> None:
        return


class _InvalidStepEnv:
    action_space = Discrete(2)

    def __init__(self) -> None:
        self.closed = False

    def reset(self, *, seed: int | None = None):
        _ = seed
        return 0, {}

    def step(self, action: int):
        _ = action
        return "not-a-gym-step-result"

    def close(self) -> None:
        self.closed = True


class _ArrayAction:
    def __init__(self, value: int) -> None:
        self._value = value

    def reshape(self, shape):
        assert shape == -1
        return [self._value]


class _BadScalar:
    def item(self):
        raise RuntimeError("cannot scalarize")

    def tolist(self):
        raise RuntimeError("cannot list")

    def __repr__(self) -> str:
        return "<bad-scalar>"


def _task() -> TaskDefinition:
    return TaskDefinition(environment_id="eval_env", name="Evaluation Task", task_id="task_eval")


def _q_learner_state(*entries: dict[str, object]) -> dict[str, object]:
    return {
        "algorithm": "q_learning",
        "q_values": list(entries),
    }


def test_evaluate_policy_rejects_non_positive_episode_count() -> None:
    with pytest.raises(ValueError, match="episode_count"):
        evaluate_policy(
            task=_task(),
            config=RunConfig(algorithm="q_learning"),
            learner_state=_q_learner_state(),
            env_factory=lambda _task: _ClosedEnv(),
            run_id="eval_invalid_count",
            episode_count=0,
            max_steps_per_episode=1,
            seed=1,
        )


def test_evaluate_policy_closes_env_when_action_selector_fails() -> None:
    env = _ClosedEnv()

    with pytest.raises(ValueError, match="not implemented"):
        evaluate_policy(
            task=_task(),
            config=RunConfig(algorithm="unsupported_algo"),
            learner_state={},
            env_factory=lambda _task: env,
            run_id="eval_unsupported",
            episode_count=1,
            max_steps_per_episode=1,
            seed=1,
        )

    assert env.closed is True


def test_evaluate_policy_closes_env_when_step_result_is_invalid() -> None:
    env = _InvalidStepEnv()

    with pytest.raises(ValueError, match="invalid step result"):
        evaluate_policy(
            task=_task(),
            config=RunConfig(algorithm="q_learning"),
            learner_state=_q_learner_state(),
            env_factory=lambda _task: env,
            run_id="eval_bad_step",
            episode_count=1,
            max_steps_per_episode=3,
            seed=2,
        )

    assert env.closed is True


def test_evaluate_policy_supports_old_gym_reset_and_step_api() -> None:
    env = _OldGymApiEnv()

    result = evaluate_policy(
        task=_task(),
        config=RunConfig(algorithm="q_learning"),
        learner_state=_q_learner_state(
            {"state_key": "0", "action": 0, "value": -1.0},
            {"state_key": "0", "action": 1, "value": 2.0},
        ),
        env_factory=lambda _task: env,
        run_id="eval_old_gym",
        episode_count=1,
        max_steps_per_episode=4,
        seed=3,
    )

    trace = result.episodes[0]
    assert env.reset_called_without_seed is True
    assert env.actions == [1]
    assert trace.success is True
    assert trace.steps[0].terminated is True
    assert trace.steps[0].truncated is False
    assert result.metrics.reward_step == 1.0
    assert result.metrics.cumulative_reward == 1.0


def test_evaluate_policy_forces_failure_at_evaluation_step_limit() -> None:
    env = _NeverDoneOldGymApiEnv()

    result = evaluate_policy(
        task=_task(),
        config=RunConfig(algorithm="q_learning"),
        learner_state=_q_learner_state(),
        env_factory=lambda _task: env,
        run_id="eval_step_limit",
        episode_count=1,
        max_steps_per_episode=2,
        seed=4,
    )

    trace = result.episodes[0]
    assert len(trace.steps) == 2
    assert trace.success is False
    assert trace.steps[-1].terminated is False
    assert trace.steps[-1].truncated is True
    assert trace.steps[-1].info["forced_failure"] == "evaluation_max_steps_per_episode"
    assert result.metrics.step == 2
    assert result.metrics.success_rate == 0.0


def test_evaluation_helper_parses_only_valid_q_values() -> None:
    restored = _q_values_from_learner_state(
        {
            "q_values": [
                {"state_key": "s0", "action": "1", "value": "2.5"},
                {"state_key": "bad", "action": "x", "value": 1.0},
                ["not", "a", "mapping"],
            ]
        }
    )

    assert restored == {("s0", 1): 2.5}
    assert _q_values_from_learner_state({"q_values": "invalid"}) == {}


def test_evaluation_action_and_state_normalization_helpers_cover_array_like_values() -> None:
    env = _ClosedEnv()

    assert _normalize_action_for_env(np.array([1]), env) == 1
    assert _normalize_action_for_env(_ArrayAction(1), env) == 1
    assert _normalize_action_for_env(np.array(0), env) == 0
    assert _state_key(np.array([1, 2])) == "[1, 2]"
    assert _state_key(_BadScalar()) == "<bad-scalar>"


def test_action_space_size_rejects_missing_invalid_or_non_positive_sizes() -> None:
    class _NoInt:
        def __int__(self):
            raise TypeError("no integer")

    class _ActionSpace:
        def __init__(self, n) -> None:
            self.n = n

    assert _action_space_size(None) is None
    assert _action_space_size(object()) is None
    assert _action_space_size(_ActionSpace(_NoInt())) is None
    assert _action_space_size(_ActionSpace(0)) is None
    assert _action_space_size(_ActionSpace(3)) == 3
