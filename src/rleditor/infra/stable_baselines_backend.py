from __future__ import annotations

import base64
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import gymnasium as gym

from rleditor.core.models import (
    EpisodeMoment,
    EpisodeStep,
    EpisodeTrace,
    RunConfig,
    TaskDefinition,
    TaskSnapshot,
)


SB3_DQN_ALGORITHM = "sb3_dqn"
SB3_PPO_ALGORITHM = "sb3_ppo"
SB3_ALGORITHMS = {SB3_DQN_ALGORITHM, SB3_PPO_ALGORITHM}


def is_stable_baselines3_algorithm(algorithm: str) -> bool:
    return algorithm in SB3_ALGORITHMS


@dataclass(slots=True)
class StableBaselines3EpisodeSummary:
    episode_id: int
    total_reward: float
    length: int
    success: bool
    final_info: dict[str, Any]


class StableBaselines3StepCallback:
    """Thin adapter around SB3 BaseCallback with lazy SB3 imports."""

    def __init__(self, on_step: Callable[[Any], bool]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, callback_on_step: Callable[[Any], bool]) -> None:
                super().__init__()
                self._callback_on_step = callback_on_step

            def _on_step(self) -> bool:
                return self._callback_on_step(self.model)

        self._callback = _Callback(on_step)

    @property
    def callback(self):
        return self._callback


class StableBaselines3TraceWrapper(gym.Wrapper):
    """Records episode summaries and optional full traces while SB3 owns stepping."""

    def __init__(
        self,
        env: gym.Env,
        *,
        task: TaskDefinition,
        config: RunConfig,
        run_id: str | None,
        trace_random: Random,
    ) -> None:
        super().__init__(env)
        self._task = task
        self._config = config
        self._run_id = run_id
        self._trace_random = trace_random
        self._next_episode_id = 1
        self._episode_id = 0
        self._current_observation: Any | None = None
        self._episode_reward_total = 0.0
        self._episode_step_counter = 0
        self._episode_seed: int | None = None
        self._record_current_episode_trace = False
        self._episode_steps_buffer: list[EpisodeStep] = []
        self._episode_moments_buffer: list[EpisodeMoment] = []
        self._completed_episode_summaries: list[StableBaselines3EpisodeSummary] = []
        self._completed_traces: list[EpisodeTrace] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        reset_result = self.env.reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, info = reset_result
        else:
            observation = reset_result
            info = {}

        self._episode_id = self._next_episode_id
        self._next_episode_id += 1
        self._episode_seed = seed
        self._current_observation = self._normalize_observation(observation)
        self._episode_reward_total = 0.0
        self._episode_step_counter = 0
        self._episode_steps_buffer = []
        self._episode_moments_buffer = []
        self._record_current_episode_trace = self._should_record_episode_trace()
        if self._record_current_episode_trace:
            self._episode_moments_buffer.append(
                EpisodeMoment(
                    episode_id=self._episode_id,
                    moment_index=0,
                    observation=self._current_observation,
                    restorable_env_state=self._maybe_export_restorable_state(),
                    metadata={"phase": "initial"},
                )
            )
        return observation, info

    def step(self, action):
        previous_observation = self._current_observation
        step_result = self.env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
        elif isinstance(step_result, tuple) and len(step_result) == 4:
            observation, reward, done, info = step_result
            terminated = bool(done)
            truncated = False
        else:
            return step_result

        info_payload = dict(info) if isinstance(info, dict) else {}
        reward_value = float(reward)
        normalized_observation = self._normalize_observation(observation)
        done = bool(terminated) or bool(truncated)

        max_steps_per_episode = self._config.max_steps_per_episode
        if (
            not done
            and max_steps_per_episode is not None
            and max_steps_per_episode > 0
            and (self._episode_step_counter + 1) >= max_steps_per_episode
        ):
            done = True
            truncated = True
            info_payload["forced_failure"] = "max_steps_per_episode"
            info_payload["max_steps_per_episode"] = max_steps_per_episode

        if self._record_current_episode_trace:
            self._episode_steps_buffer.append(
                EpisodeStep(
                    t=self._episode_step_counter,
                    observation=previous_observation,
                    action=self._normalize_action(action),
                    reward=reward_value,
                    next_observation=normalized_observation,
                    terminated=(
                        bool(terminated)
                        and info_payload.get("forced_failure") != "max_steps_per_episode"
                    ),
                    truncated=bool(truncated),
                    info=info_payload,
                )
            )
            self._episode_moments_buffer.append(
                EpisodeMoment(
                    episode_id=self._episode_id,
                    moment_index=self._episode_step_counter + 1,
                    observation=normalized_observation,
                    action_taken=self._normalize_action(action),
                    reward=reward_value,
                    restorable_env_state=self._maybe_export_restorable_state(),
                    metadata={
                        "terminated": bool(terminated)
                        and info_payload.get("forced_failure") != "max_steps_per_episode",
                        "truncated": bool(truncated),
                        "info": info_payload,
                    },
                )
            )

        self._episode_step_counter += 1
        self._episode_reward_total += reward_value
        self._current_observation = normalized_observation

        if done:
            success = self._episode_success(info_payload, self._episode_reward_total)
            self._completed_episode_summaries.append(
                StableBaselines3EpisodeSummary(
                    episode_id=self._episode_id,
                    total_reward=self._episode_reward_total,
                    length=self._episode_step_counter,
                    success=success,
                    final_info=info_payload,
                )
            )
            if self._record_current_episode_trace:
                self._completed_traces.append(self._build_episode_trace(info_payload, success))

        return observation, reward, terminated, truncated, info_payload

    def drain_episode_summaries(self) -> list[StableBaselines3EpisodeSummary]:
        summaries = self._completed_episode_summaries
        self._completed_episode_summaries = []
        return summaries

    def drain_traces(self) -> list[EpisodeTrace]:
        traces = self._completed_traces
        self._completed_traces = []
        return traces

    def _build_episode_trace(self, final_info: dict[str, Any], success: bool) -> EpisodeTrace:
        steps = list(self._episode_steps_buffer)
        moments = list(self._episode_moments_buffer)
        return EpisodeTrace(
            episode_id=self._episode_id,
            run_id=self._run_id,
            total_reward=self._episode_reward_total,
            success=success,
            steps=steps,
            moments=moments,
            initial_observation=steps[0].observation if steps else self._current_observation,
            task_snapshot=TaskSnapshot(
                environment_id=self._task.environment_id,
                task_name=self._task.name,
                task_id=self._task.task_id,
                task_config=dict(self._task.config),
                reward_config=dict(self._task.reward_config),
                termination_config=dict(self._task.termination_config),
                metadata={
                    "seed": self._config.seed,
                    "episode_seed": self._episode_seed,
                    "run_id": self._run_id,
                },
            ),
            metadata={
                "runner": "stable_baselines3",
                "algorithm": self._config.algorithm,
                "trace_sample_rate": self._config.episode_trace_sample_rate,
                "restorable_state_captured": any(
                    moment.restorable_env_state is not None for moment in moments
                ),
                "final_info": final_info,
            },
        )

    def _should_record_episode_trace(self) -> bool:
        rate = self._config.episode_trace_sample_rate
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        return self._trace_random.random() < rate

    def _episode_success(self, final_info: dict[str, Any], total_reward: float) -> bool:
        if final_info.get("forced_failure"):
            return False
        if "is_success" in final_info:
            return bool(final_info["is_success"])
        return total_reward > 0.0

    def _maybe_export_restorable_state(self) -> Any | None:
        export_state = getattr(self.env, "export_state", None)
        if not callable(export_state):
            return None
        try:
            return export_state()
        except Exception:
            return None

    def _normalize_observation(self, observation: Any) -> Any:
        if observation is None or isinstance(observation, (str, int, float, bool)):
            return observation
        if hasattr(observation, "item") and callable(getattr(observation, "item")):
            try:
                return observation.item()
            except Exception:
                pass
        if hasattr(observation, "tolist") and callable(getattr(observation, "tolist")):
            try:
                return observation.tolist()
            except Exception:
                pass
        return observation

    def _normalize_action(self, action: Any) -> Any:
        if action is None or isinstance(action, (str, int, float, bool)):
            return action
        if hasattr(action, "item") and callable(getattr(action, "item")):
            try:
                return action.item()
            except Exception:
                pass
        if hasattr(action, "tolist") and callable(getattr(action, "tolist")):
            try:
                return action.tolist()
            except Exception:
                pass
        return action


def create_stable_baselines3_model(
    *,
    algorithm: str,
    env: gym.Env,
    config: RunConfig,
    learner_state: dict[str, Any] | None = None,
) -> Any:
    if learner_state is not None and learner_state.get("algorithm") == algorithm:
        loaded = _load_model_from_learner_state(
            algorithm=algorithm,
            env=env,
            learner_state=learner_state,
        )
        if loaded is not None:
            return loaded

    if algorithm == SB3_DQN_ALGORITHM:
        from stable_baselines3 import DQN

        return DQN(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            seed=config.seed,
            verbose=0,
            **_dqn_kwargs(config),
        )

    if algorithm == SB3_PPO_ALGORITHM:
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            seed=config.seed,
            verbose=0,
            **_ppo_kwargs(config),
        )

    raise ValueError(f"Unsupported Stable-Baselines3 algorithm: {algorithm}")


def export_stable_baselines3_learner_state(model: Any, *, algorithm: str) -> dict[str, Any]:
    if model is None:
        return {}

    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
            path = Path(handle.name)
        model.save(path)
        payload = path.read_bytes()
    finally:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    return {
        "algorithm": algorithm,
        "backend": "stable_baselines3",
        "model_zip_base64": base64.b64encode(payload).decode("ascii"),
    }


def load_stable_baselines3_model(
    *,
    algorithm: str,
    env: gym.Env | None,
    learner_state: dict[str, Any],
) -> Any:
    model = _load_model_from_learner_state(
        algorithm=algorithm,
        env=env,
        learner_state=learner_state,
    )
    if model is None:
        raise ValueError(f"Cannot load Stable-Baselines3 learner state for algorithm: {algorithm}")
    return model


def _load_model_from_learner_state(
    *,
    algorithm: str,
    env: gym.Env | None,
    learner_state: dict[str, Any],
) -> Any | None:
    encoded = learner_state.get("model_zip_base64")
    if not isinstance(encoded, str) or not encoded:
        return None

    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
            path = Path(handle.name)
            handle.write(base64.b64decode(encoded))

        if algorithm == SB3_DQN_ALGORITHM:
            from stable_baselines3 import DQN

            return DQN.load(path, env=env)
        if algorithm == SB3_PPO_ALGORITHM:
            from stable_baselines3 import PPO

            return PPO.load(path, env=env)
    finally:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    return None


def _dqn_kwargs(config: RunConfig) -> dict[str, Any]:
    allowed_keys = {
        "buffer_size",
        "learning_starts",
        "batch_size",
        "tau",
        "train_freq",
        "gradient_steps",
        "target_update_interval",
        "exploration_fraction",
        "exploration_initial_eps",
        "exploration_final_eps",
        "policy_kwargs",
        "device",
    }
    return {
        key: value
        for key, value in config.hyperparameters.items()
        if key in allowed_keys
    }


def _ppo_kwargs(config: RunConfig) -> dict[str, Any]:
    allowed_keys = {
        "n_steps",
        "batch_size",
        "n_epochs",
        "gae_lambda",
        "clip_range",
        "clip_range_vf",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "use_sde",
        "sde_sample_freq",
        "target_kl",
        "policy_kwargs",
        "device",
    }
    return {
        key: value
        for key, value in config.hyperparameters.items()
        if key in allowed_keys
    }
