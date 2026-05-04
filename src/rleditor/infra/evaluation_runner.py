from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

from rleditor.core.models import (
    EpisodeMoment,
    EpisodeStep,
    EpisodeTrace,
    RunConfig,
    TaskDefinition,
    TaskSnapshot,
    TrainingMetrics,
)
from rleditor.infra.stable_baselines_backend import (
    is_stable_baselines3_algorithm,
    load_stable_baselines3_model,
)


@dataclass(slots=True)
class EvaluationResult:
    run_id: str
    task: TaskDefinition
    task_snapshot: TaskSnapshot
    metrics: TrainingMetrics
    episodes: list[EpisodeTrace]


def evaluate_policy(
    *,
    task: TaskDefinition,
    config: RunConfig,
    learner_state: dict[str, Any],
    env_factory: Callable[[TaskDefinition], Any],
    run_id: str,
    episode_count: int,
    max_steps_per_episode: int | None,
) -> EvaluationResult:
    if episode_count <= 0:
        raise ValueError("Evaluation episode_count must be positive")

    env = env_factory(task)
    started_at = time.perf_counter()
    try:
        action_selector = _build_action_selector(
            algorithm=config.algorithm,
            learner_state=learner_state,
            env=env,
            config=config,
        )
        task_snapshot = _build_task_snapshot(task)
        episodes = [
            _run_episode(
                env=env,
                task_snapshot=task_snapshot,
                action_selector=action_selector,
                run_id=run_id,
                episode_id=episode_index + 1,
                seed=None if config.seed is None else config.seed + episode_index,
                max_steps_per_episode=max_steps_per_episode,
                algorithm=config.algorithm,
            )
            for episode_index in range(episode_count)
        ]
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    metrics = _metrics_from_episodes(
        episodes,
        elapsed=max(time.perf_counter() - started_at, 1e-6),
    )
    return EvaluationResult(
        run_id=run_id,
        task=task,
        task_snapshot=_build_task_snapshot(task),
        metrics=metrics,
        episodes=episodes,
    )


def _build_action_selector(
    *,
    algorithm: str,
    learner_state: dict[str, Any],
    env: Any,
    config: RunConfig,
) -> Callable[[Any, Any], Any]:
    if algorithm == "q_learning":
        return _build_q_learning_action_selector(learner_state, env)

    if is_stable_baselines3_algorithm(algorithm):
        model = load_stable_baselines3_model(
            algorithm=algorithm,
            env=env,
            learner_state=learner_state,
        )

        def select_action(raw_observation: Any, normalized_observation: Any) -> Any:
            _ = normalized_observation
            action, _state = model.predict(raw_observation, deterministic=True)
            return action

        return select_action

    msg = f"Evaluation is not implemented for algorithm: {config.algorithm}"
    raise ValueError(msg)


def _build_q_learning_action_selector(
    learner_state: dict[str, Any],
    env: Any,
) -> Callable[[Any, Any], int]:
    action_count = _action_space_size(getattr(env, "action_space", None))
    if action_count is None:
        raise ValueError("Q-learning evaluation requires a discrete action_space.n")

    q_values = _q_values_from_learner_state(learner_state)

    def select_action(raw_observation: Any, normalized_observation: Any) -> int:
        _ = raw_observation
        state_key = _state_key(normalized_observation)
        best_action = 0
        best_value = float("-inf")
        for action in range(action_count):
            value = q_values.get((state_key, action), 0.0)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    return select_action


def _run_episode(
    *,
    env: Any,
    task_snapshot: TaskSnapshot,
    action_selector: Callable[[Any, Any], Any],
    run_id: str,
    episode_id: int,
    seed: int | None,
    max_steps_per_episode: int | None,
    algorithm: str,
) -> EpisodeTrace:
    raw_observation = _reset_env(env, seed=seed)
    normalized_observation = _normalize_value(raw_observation)
    moments = [
        EpisodeMoment(
            episode_id=episode_id,
            moment_index=0,
            observation=normalized_observation,
            restorable_env_state=_maybe_export_restorable_state(env),
            metadata={"phase": "initial"},
        )
    ]
    steps: list[EpisodeStep] = []
    total_reward = 0.0
    done = False
    final_info: dict[str, Any] = {}

    while not done:
        action = action_selector(raw_observation, normalized_observation)
        env_action = _normalize_action_for_env(action, env)
        recorded_action = _normalize_value(env_action)
        step_result = env.step(env_action)
        if not isinstance(step_result, tuple) or len(step_result) not in {4, 5}:
            raise ValueError(f"Evaluation env returned an invalid step result: {step_result!r}")

        if len(step_result) == 5:
            next_raw_observation, reward, terminated, truncated, info = step_result
        else:
            next_raw_observation, reward, terminated, info = step_result
            truncated = False

        info_payload = dict(info) if isinstance(info, dict) else {}
        reward_value = float(reward)
        next_normalized_observation = _normalize_value(next_raw_observation)
        forced_by_step_limit = False

        done = bool(terminated) or bool(truncated)
        if (
            not done
            and max_steps_per_episode is not None
            and max_steps_per_episode > 0
            and (len(steps) + 1) >= max_steps_per_episode
        ):
            done = True
            truncated = True
            forced_by_step_limit = True
            info_payload["forced_failure"] = "evaluation_max_steps_per_episode"
            info_payload["max_steps_per_episode"] = max_steps_per_episode

        steps.append(
            EpisodeStep(
                t=len(steps),
                observation=normalized_observation,
                action=recorded_action,
                reward=reward_value,
                next_observation=next_normalized_observation,
                terminated=bool(terminated) and not forced_by_step_limit,
                truncated=bool(truncated),
                info=info_payload,
            )
        )
        moments.append(
            EpisodeMoment(
                episode_id=episode_id,
                moment_index=len(steps),
                observation=next_normalized_observation,
                action_taken=recorded_action,
                reward=reward_value,
                restorable_env_state=_maybe_export_restorable_state(env),
                metadata={
                    "terminated": bool(terminated) and not forced_by_step_limit,
                    "truncated": bool(truncated),
                    "info": info_payload,
                },
            )
        )

        total_reward += reward_value
        raw_observation = next_raw_observation
        normalized_observation = next_normalized_observation
        final_info = info_payload

    success = _episode_success(final_info, total_reward)
    return EpisodeTrace(
        episode_id=episode_id,
        run_id=run_id,
        total_reward=total_reward,
        success=success,
        steps=steps,
        moments=moments,
        task_snapshot=task_snapshot,
        initial_observation=moments[0].observation,
        metadata={
            "runner": "evaluation",
            "algorithm": algorithm,
            "trace_sample_rate": 1.0,
            "restorable_state_captured": any(
                moment.restorable_env_state is not None for moment in moments
            ),
            "final_info": final_info,
        },
    )


def _metrics_from_episodes(episodes: list[EpisodeTrace], *, elapsed: float) -> TrainingMetrics:
    rewards = [trace.total_reward for trace in episodes]
    lengths = [len(trace.steps) for trace in episodes]
    successes = [1 if trace.success else 0 for trace in episodes]
    step_count = sum(lengths)
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    success_rate = sum(successes) / len(successes) if successes else 0.0
    last_reward = episodes[-1].steps[-1].reward if episodes and episodes[-1].steps else 0.0
    return TrainingMetrics(
        step=step_count,
        episode=len(episodes),
        reward_step=last_reward,
        mean_reward=mean_reward,
        episode_reward_mean=mean_reward,
        success_rate=success_rate,
        episode_length_mean=mean_length,
        fps=step_count / elapsed if step_count > 0 else 0.0,
        exploration_rate=0.0,
        value_loss=None,
        policy_loss=None,
    )


def _reset_env(env: Any, *, seed: int | None) -> Any:
    try:
        reset_result = env.reset(seed=seed)
    except TypeError:
        reset_result = env.reset()

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        observation, _info = reset_result
        return observation
    return reset_result


def _episode_success(final_info: dict[str, Any], total_reward: float) -> bool:
    if final_info.get("forced_failure"):
        return False
    if "is_success" in final_info:
        return bool(final_info["is_success"])
    return total_reward > 0.0


def _maybe_export_restorable_state(env: Any) -> Any | None:
    export_state = getattr(env, "export_state", None)
    if not callable(export_state):
        return None
    try:
        return export_state()
    except Exception:
        return None


def _build_task_snapshot(task: TaskDefinition) -> TaskSnapshot:
    return TaskSnapshot(
        environment_id=task.environment_id,
        task_name=task.name,
        task_id=task.task_id,
        task_config=dict(task.config),
        reward_config=dict(task.reward_config),
        termination_config=dict(task.termination_config),
        metadata=dict(task.metadata),
    )


def _normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _normalize_action_for_env(action: Any, env: Any) -> Any:
    action_space = getattr(env, "action_space", None)
    if action_space is None or not hasattr(action_space, "n"):
        return action

    if isinstance(action, (str, int, float, bool)):
        return int(action)
    if hasattr(action, "item") and callable(getattr(action, "item")):
        try:
            return int(action.item())
        except Exception:
            pass
    if hasattr(action, "reshape") and callable(getattr(action, "reshape")):
        try:
            flattened = action.reshape(-1)
            if len(flattened) > 0:
                return int(flattened[0])
        except Exception:
            pass
    if hasattr(action, "tolist") and callable(getattr(action, "tolist")):
        try:
            as_list = action.tolist()
        except Exception:
            as_list = None
        if isinstance(as_list, list) and as_list:
            return int(as_list[0])
        if isinstance(as_list, (int, float, bool)):
            return int(as_list)

    return action


def _q_values_from_learner_state(learner_state: dict[str, Any]) -> dict[tuple[str, int], float]:
    restored: dict[tuple[str, int], float] = {}
    q_values_payload = learner_state.get("q_values", [])
    if not isinstance(q_values_payload, list):
        return restored

    for entry in q_values_payload:
        if not isinstance(entry, dict):
            continue
        try:
            state_key = str(entry.get("state_key", ""))
            action = int(entry.get("action", 0))
            value = float(entry.get("value", 0.0))
        except (TypeError, ValueError):
            continue
        restored[(state_key, action)] = value
    return restored


def _action_space_size(action_space: Any) -> int | None:
    if action_space is None or not hasattr(action_space, "n"):
        return None

    try:
        n_actions = int(action_space.n)
    except (TypeError, ValueError):
        return None

    if n_actions <= 0:
        return None
    return n_actions


def _state_key(state: Any) -> str:
    if isinstance(state, (str, int, float, bool)) or state is None:
        return str(state)
    if hasattr(state, "tolist"):
        try:
            return repr(state.tolist())
        except Exception:
            return repr(state)
    return repr(state)
