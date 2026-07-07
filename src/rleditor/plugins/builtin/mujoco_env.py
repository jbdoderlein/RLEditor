from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from typing import Any

# MuJoCo needs a render backend before it is imported by Gymnasium's MuJoCo envs.
# EGL is the safest default for Linux/headless Qt sessions; users can override it.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from gymnasium import error as gym_error
import numpy as np

from rleditor.core.models import TaskDefinition, TaskSnapshot

INVERTED_DOUBLE_PENDULUM_ENV_ID = "InvertedDoublePendulum-v5"
DEFAULT_UPRIGHT_ANGLE_THRESHOLD = 0.2


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _to_serializable(value.tolist())
        except Exception:
            return repr(value)
    return repr(value)


def _float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise ValueError(f"Expected a numeric sequence, got {value!r}")
    return [float(item) for item in value]


def _optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    return _float_list(value)


def _float_array(value: Any):
    return np.asarray(_float_list(value), dtype=np.float64)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _assign_sequence(target: Any, values: list[float]) -> None:
    if hasattr(target, "__setitem__"):
        target[:] = values
        return
    msg = "MuJoCo state target does not support item assignment"
    raise RuntimeError(msg)


@dataclass(slots=True, frozen=True)
class MujocoEnvState:
    """Serializable subset of a MuJoCo simulator state."""

    qpos: list[float]
    qvel: list[float]
    time: float | None = None
    ctrl: list[float] | None = None
    observation: Any | None = None
    last_action: Any | None = None
    terminated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "qpos": list(self.qpos),
            "qvel": list(self.qvel),
            "time": self.time,
            "ctrl": None if self.ctrl is None else list(self.ctrl),
            "observation": _to_serializable(self.observation),
            "last_action": _to_serializable(self.last_action),
            "terminated": self.terminated,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MujocoEnvState:
        return cls(
            qpos=_float_list(payload.get("qpos", [])),
            qvel=_float_list(payload.get("qvel", [])),
            time=(
                float(payload.get("time"))
                if payload.get("time") is not None
                else None
            ),
            ctrl=_optional_float_list(payload.get("ctrl")),
            observation=payload.get("observation"),
            last_action=payload.get("last_action"),
            terminated=bool(payload.get("terminated", False)),
        )


class MujocoExtendedEnv(gym.Wrapper):
    """Generic Gymnasium MuJoCo env with best-effort state import/export hooks."""

    def __init__(self, task: TaskDefinition, *, render_mode: str | None = None) -> None:
        self._task = deepcopy(task)
        self._render_mode = render_mode if render_mode is not None else self._task.config.get("render_mode")
        self._last_action: Any | None = None
        self._terminated = False
        self._env_id = str(self._task.config.get("env_id", INVERTED_DOUBLE_PENDULUM_ENV_ID))
        self._upright_angle_threshold = _optional_float(
            self._task.reward_config.get("upright_angle_threshold", DEFAULT_UPRIGHT_ANGLE_THRESHOLD)
        )
        env = self._build_env(self._task, render_mode=self._render_mode)
        super().__init__(env)

    @classmethod
    def from_task_snapshot(
        cls,
        task_snapshot: TaskSnapshot | None,
        *,
        render_mode: str | None = None,
    ) -> MujocoExtendedEnv | None:
        if task_snapshot is None:
            return None

        task = TaskDefinition(
            environment_id=task_snapshot.environment_id,
            name=task_snapshot.task_name,
            task_id=task_snapshot.task_id,
            config=deepcopy(task_snapshot.task_config),
            reward_config=deepcopy(task_snapshot.reward_config),
            termination_config=deepcopy(task_snapshot.termination_config),
            metadata=deepcopy(task_snapshot.metadata),
        )
        return cls(task, render_mode=render_mode)

    @property
    def task_definition(self) -> TaskDefinition:
        return deepcopy(self._task)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if options is None:
            reset_result = self.env.reset(seed=seed)
        else:
            reset_result = self.env.reset(seed=seed, options=options)

        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, info = reset_result
        else:
            observation = reset_result
            info = {}

        self._last_action = None
        self._terminated = False
        if self._apply_initial_state_override(self._task.config.get("initial_state")):
            observation = self._current_observation(fallback=observation)
            info = dict(info)
            info["initial_state_override"] = True
        return observation, info

    def step(self, action: Any):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            observation, reward, terminated, truncated, info = result
        else:
            observation, reward, terminated, info = result
            truncated = False

        info = dict(info)
        reward = float(reward)
        terminated = bool(terminated)
        if self._uses_custom_upright_threshold():
            self._annotate_upright_threshold(info)

        self._last_action = _to_serializable(action)
        self._terminated = bool(terminated) or bool(truncated)
        return observation, reward, terminated, truncated, info

    def export_state(self) -> MujocoEnvState:
        base_env = self._ensure_base_env_state()
        data = getattr(base_env, "data", None)
        if data is None or not hasattr(data, "qpos") or not hasattr(data, "qvel"):
            msg = "MuJoCo environment does not expose data.qpos/data.qvel for restorable state export"
            raise RuntimeError(msg)

        return MujocoEnvState(
            qpos=_float_list(getattr(data, "qpos")),
            qvel=_float_list(getattr(data, "qvel")),
            time=float(getattr(data, "time")) if getattr(data, "time", None) is not None else None,
            ctrl=_optional_float_list(getattr(data, "ctrl", None)),
            observation=_to_serializable(self._current_observation(fallback=None)),
            last_action=self._last_action,
            terminated=self._terminated,
        )

    def import_state(self, state: MujocoEnvState | dict[str, Any]) -> MujocoEnvState:
        resolved_state = self._coerce_env_state(state)
        base_env = self._ensure_base_env_state()
        data = getattr(base_env, "data", None)
        if data is None or not hasattr(data, "qpos") or not hasattr(data, "qvel"):
            msg = "MuJoCo environment does not expose data.qpos/data.qvel for restorable state import"
            raise RuntimeError(msg)

        set_state = getattr(base_env, "set_state", None)
        if callable(set_state):
            set_state(_float_array(resolved_state.qpos), _float_array(resolved_state.qvel))
        else:
            _assign_sequence(getattr(data, "qpos"), resolved_state.qpos)
            _assign_sequence(getattr(data, "qvel"), resolved_state.qvel)

        if resolved_state.time is not None and hasattr(data, "time"):
            setattr(data, "time", resolved_state.time)
        if resolved_state.ctrl is not None and hasattr(data, "ctrl"):
            _assign_sequence(getattr(data, "ctrl"), resolved_state.ctrl)

        self._last_action = resolved_state.last_action
        self._terminated = resolved_state.terminated
        return resolved_state

    def reinstantiate(self, *, render_mode: str | None = None) -> MujocoExtendedEnv:
        target_render_mode = self._render_mode if render_mode is None else render_mode
        return MujocoExtendedEnv(self._task, render_mode=target_render_mode)

    def _build_env(self, task: TaskDefinition, *, render_mode: str | None) -> gym.Env:
        env_id = str(task.config.get("env_id", INVERTED_DOUBLE_PENDULUM_ENV_ID))
        make_kwargs = task.config.get("make_kwargs", {})
        if not isinstance(make_kwargs, dict):
            make_kwargs = {}

        kwargs = dict(make_kwargs)
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        try:
            return gym.make(env_id, **kwargs)
        except gym_error.DependencyNotInstalled as exc:
            msg = (
                "MuJoCo is not installed. Install it with `uv sync --extra mujoco` "
                "or `uv pip install 'gymnasium[mujoco]'`."
            )
            raise RuntimeError(msg) from exc

    def _ensure_base_env_state(self) -> object:
        base_env = getattr(self, "unwrapped", self)
        data = getattr(base_env, "data", None)
        if data is None or not hasattr(data, "qpos") or not hasattr(data, "qvel"):
            self.env.reset()
        return base_env

    def _current_observation(self, *, fallback: Any) -> Any:
        base_env = getattr(self, "unwrapped", self)
        get_obs = getattr(base_env, "_get_obs", None)
        if callable(get_obs):
            try:
                return get_obs()
            except Exception:
                return fallback
        return fallback

    def _coerce_env_state(self, state: MujocoEnvState | dict[str, Any]) -> MujocoEnvState:
        if isinstance(state, MujocoEnvState):
            return state
        if isinstance(state, dict):
            return MujocoEnvState.from_dict(state)
        raise ValueError(f"Unsupported MuJoCo state payload: {state!r}")

    def _coerce_optional_initial_state(self, state: object) -> MujocoEnvState | None:
        if state is None:
            return None
        if isinstance(state, MujocoEnvState):
            return state
        if isinstance(state, dict) and "qpos" in state and "qvel" in state:
            return MujocoEnvState.from_dict(state)
        return None

    def _apply_initial_state_override(self, state: object) -> bool:
        resolved_state = self._coerce_optional_initial_state(state)
        if resolved_state is not None:
            self.import_state(resolved_state)
            return True

        if not isinstance(state, dict):
            return False

        cart_position = _optional_float(state.get("cart_position"))
        cart_velocity = _optional_float(state.get("cart_velocity"))
        if cart_position is None and cart_velocity is None:
            return False

        current_state = self.export_state()
        qpos = list(current_state.qpos)
        qvel = list(current_state.qvel)
        if cart_position is not None and qpos:
            qpos[0] = cart_position
        if cart_velocity is not None and qvel:
            qvel[0] = cart_velocity
        self.import_state(
            MujocoEnvState(
                qpos=qpos,
                qvel=qvel,
                time=current_state.time,
                ctrl=current_state.ctrl,
                observation=current_state.observation,
                last_action=current_state.last_action,
                terminated=current_state.terminated,
            )
        )
        return True

    def _uses_custom_upright_threshold(self) -> bool:
        return self._env_id == INVERTED_DOUBLE_PENDULUM_ENV_ID and self._upright_angle_threshold is not None

    def _annotate_upright_threshold(self, info: dict[str, Any]) -> None:
        threshold = self._upright_angle_threshold
        if threshold is None:
            return

        angles = self._current_pole_angles()
        if angles is None:
            return

        info["upright_angle_threshold"] = threshold
        info["upright_angle_healthy"] = all(abs(angle) < threshold for angle in angles)
        info["upright_angles"] = _to_serializable(angles)

    def _current_pole_angles(self) -> list[float] | None:
        base_env = getattr(self, "unwrapped", self)
        data = getattr(base_env, "data", None)
        qpos = getattr(data, "qpos", None)
        if qpos is None:
            return None
        try:
            values = _float_list(qpos)
        except ValueError:
            return None
        if len(values) < 3:
            return None
        return values[1:3]
