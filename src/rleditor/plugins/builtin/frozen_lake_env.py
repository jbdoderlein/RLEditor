from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, cast

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from rleditor.core.models import TaskDefinition, TaskSnapshot

FROZEN_LAKE_4X4_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]

FROZEN_LAKE_8X8_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

TILE_START = "S"
TILE_FROZEN = "F"
TILE_HOLE = "H"
TILE_GOAL = "G"
VALID_TILES = {TILE_START, TILE_FROZEN, TILE_HOLE, TILE_GOAL}
DEFAULT_SUCCESS_RATE = 1.0 / 3.0


def _default_reward_config() -> dict[str, float]:
    return {
        "tile:F": 0.0,
        "tile:H": 0.0,
        "tile:S": 0.0,
        "tile:G": 1.0,
    }


def _parse_size(value: object, *, fallback: int = 4) -> int:
    if isinstance(value, int):
        return max(2, value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text.endswith("x"):
            text = text[:-1]
        if "x" in text:
            text = text.split("x", 1)[0]
        if text.isdigit():
            return max(2, int(text))
    return fallback


def _map_from_task_config(config: dict[str, object], *, fallback_size: int = 4) -> list[str]:
    raw_map = config.get("map_desc")
    if isinstance(raw_map, list) and raw_map and all(isinstance(row, str) for row in raw_map):
        return [str(row) for row in raw_map]

    size = _parse_size(config.get("size", fallback_size), fallback=fallback_size)
    if size == 8:
        return list(FROZEN_LAKE_8X8_MAP)
    if size == 4:
        return list(FROZEN_LAKE_4X4_MAP)

    rows = [TILE_FROZEN * size for _ in range(size)]
    rows[0] = TILE_START + rows[0][1:]
    rows[-1] = rows[-1][:-1] + TILE_GOAL
    return rows


def _generate_random_map_desc(size: int, hole_probability: float) -> list[str]:
    if not 0.0 <= hole_probability < 1.0:
        raise ValueError("Frozen Lake hole_probability must be in [0, 1)")
    return [str(row) for row in generate_random_map(size=size, p=1.0 - hole_probability)]


def _parse_success_rate(value: object, *, fallback: float = DEFAULT_SUCCESS_RATE) -> float:
    if value is None:
        return fallback

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Frozen Lake success_rate must be a number in [0, 1]") from exc

    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("Frozen Lake success_rate must be a finite number in [0, 1]")
    return numeric


def _normalize_map_desc(raw_map: list[str], expected_size: int | None = None) -> list[list[str]]:
    rows = [list(row) for row in raw_map if row]
    if not rows:
        size = expected_size or 4
        rows = [list(TILE_FROZEN * size) for _ in range(size)]

    size = expected_size or min(len(rows), len(rows[0]))
    size = max(2, size)

    normalized: list[list[str]] = []
    for row in rows[:size]:
        values = [cell if cell in VALID_TILES else TILE_FROZEN for cell in row[:size]]
        while len(values) < size:
            values.append(TILE_FROZEN)
        normalized.append(values)

    while len(normalized) < size:
        normalized.append([TILE_FROZEN] * size)

    start_positions = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if normalized[r][c] == TILE_START
    ]
    goal_positions = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if normalized[r][c] == TILE_GOAL
    ]

    if not start_positions:
        normalized[0][0] = TILE_START
    elif len(start_positions) > 1:
        first = start_positions[0]
        for row, col in start_positions[1:]:
            normalized[row][col] = TILE_FROZEN
        normalized[first[0]][first[1]] = TILE_START

    if not goal_positions:
        normalized[-1][-1] = TILE_GOAL
    elif len(goal_positions) > 1:
        first = goal_positions[0]
        for row, col in goal_positions[1:]:
            normalized[row][col] = TILE_FROZEN
        normalized[first[0]][first[1]] = TILE_GOAL

    return normalized


def _to_rows(map_cells: list[list[str]]) -> list[str]:
    return ["".join(row) for row in map_cells]


def _coerce_reward_config(reward_config: dict[str, float]) -> dict[str, float]:
    merged = _default_reward_config()
    for key, value in reward_config.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if key in merged:
            merged[key] = numeric
            continue

        if key in {"F", "H", "S", "G"}:
            merged[f"tile:{key}"] = numeric
    return merged


def coerce_frozen_lake_state_index(state: object) -> int | None:
    if isinstance(state, Integral):
        return int(state)
    if hasattr(state, "item") and callable(getattr(state, "item")):
        try:
            scalar = cast(Any, state).item()
        except Exception:
            scalar = None
        if isinstance(scalar, Integral):
            return int(scalar)
    if isinstance(state, str) and state.isdigit():
        return int(state)
    return None


def _tile_at_state(unwrapped_env: object, state_index: int) -> str:
    desc = getattr(unwrapped_env, "desc", None)
    ncol = int(getattr(unwrapped_env, "ncol", 0) or 0)
    if desc is None or ncol <= 0:
        return TILE_FROZEN

    row = state_index // ncol
    col = state_index % ncol

    try:
        value = desc[row][col]
    except Exception:
        return TILE_FROZEN

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        try:
            return value.decode("utf-8")
        except Exception:
            pass
    return str(value)


class FrozenLakeRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_config: dict[str, float]) -> None:
        super().__init__(env)
        self._reward_config = _coerce_reward_config(reward_config)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        state_index = int(observation)
        tile = _tile_at_state(self.unwrapped, state_index)
        mapped_key = f"tile:{tile}"

        if mapped_key in self._reward_config:
            base_reward = reward
            reward = self._reward_config[mapped_key]
            info = dict(info)
            info["base_reward"] = base_reward
            info["reward_tile"] = tile

        return observation, reward, terminated, truncated, info


class FrozenLakeRegionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        region: dict[str, int],
        terminate_on_exit: bool,
        outside_reward: float,
    ) -> None:
        super().__init__(env)
        self._terminate_on_exit = terminate_on_exit
        self._outside_reward = outside_reward

        raw_row_min = int(region.get("row_min", 0))
        raw_row_max = int(region.get("row_max", 0))
        raw_col_min = int(region.get("col_min", 0))
        raw_col_max = int(region.get("col_max", 0))

        self._row_min = min(raw_row_min, raw_row_max)
        self._row_max = max(raw_row_min, raw_row_max)
        self._col_min = min(raw_col_min, raw_col_max)
        self._col_max = max(raw_col_min, raw_col_max)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        ncol = int(getattr(self.unwrapped, "ncol", 0) or 0)
        if ncol <= 0:
            return observation, reward, terminated, truncated, info

        state_index = int(observation)
        row = state_index // ncol
        col = state_index % ncol

        if (
            row < self._row_min
            or row > self._row_max
            or col < self._col_min
            or col > self._col_max
        ):
            info = dict(info)
            info["outside_region"] = True
            reward = self._outside_reward
            if self._terminate_on_exit:
                terminated = True

        return observation, reward, terminated, truncated, info


class FrozenLakeStartStateWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, start_state: int) -> None:
        super().__init__(env)
        self._start_state = int(start_state)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if options is None:
            reset_result = self.env.reset(seed=seed)
        else:
            reset_result = self.env.reset(seed=seed, options=options)

        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            _observation, info = reset_result
        else:
            info = {}

        base_env = getattr(self, "unwrapped", self)
        state_count = _state_count(base_env)
        if state_count is not None and not 0 <= self._start_state < state_count:
            raise ValueError(
                f"Frozen Lake start state {self._start_state} is out of range for {state_count} states"
            )

        setattr(base_env, "s", self._start_state)
        if hasattr(base_env, "lastaction"):
            setattr(base_env, "lastaction", None)

        info = dict(info)
        info["start_state_override"] = self._start_state
        return self._start_state, info


@dataclass(slots=True, frozen=True)
class FrozenLakeEnvState:
    state_index: int
    last_action: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "state_index": self.state_index,
            "last_action": self.last_action,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrozenLakeEnvState:
        return cls(
            state_index=int(payload.get("state_index", 0)),
            last_action=(
                int(payload.get("last_action"))
                if payload.get("last_action") is not None
                else None
            ),
        )


class FrozenLakeExtendedEnv(gym.Wrapper):
    """Frozen Lake environment extended with state import/export hooks."""

    def __init__(self, task: TaskDefinition, *, render_mode: str | None = None) -> None:
        self._task = deepcopy(task)
        self._render_mode = render_mode
        env = self._build_wrapped_env(self._task, render_mode=render_mode)
        super().__init__(env)

    @classmethod
    def from_task_snapshot(
        cls,
        task_snapshot: TaskSnapshot | None,
        *,
        render_mode: str | None = None,
    ) -> FrozenLakeExtendedEnv | None:
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

    def export_state(self) -> FrozenLakeEnvState:
        base_env = self._ensure_base_env_state()
        raw_state = getattr(base_env, "s", None)
        state_index = coerce_frozen_lake_state_index(raw_state)
        if state_index is None:
            raise RuntimeError("Frozen Lake environment does not expose a restorable state index")

        last_action = getattr(base_env, "lastaction", None)
        action_index = int(last_action) if isinstance(last_action, Integral) else None
        return FrozenLakeEnvState(state_index=state_index, last_action=action_index)

    def import_state(self, state: FrozenLakeEnvState | dict[str, Any] | int) -> FrozenLakeEnvState:
        resolved_state = self._coerce_env_state(state)
        base_env = self._ensure_base_env_state()

        state_count = self._state_count(base_env)
        if state_count is not None and not 0 <= resolved_state.state_index < state_count:
            raise ValueError(
                f"Frozen Lake state index {resolved_state.state_index} is out of range for {state_count} states"
            )

        setattr(base_env, "s", resolved_state.state_index)
        if hasattr(base_env, "lastaction"):
            setattr(base_env, "lastaction", resolved_state.last_action)
        return resolved_state

    def reinstantiate(self, *, render_mode: str | None = None) -> FrozenLakeExtendedEnv:
        target_render_mode = self._render_mode if render_mode is None else render_mode
        return FrozenLakeExtendedEnv(self._task, render_mode=target_render_mode)

    def _ensure_base_env_state(self) -> object:
        base_env = getattr(self, "unwrapped", self)
        if not hasattr(base_env, "s"):
            self.reset()
        return base_env

    def _coerce_env_state(self, state: FrozenLakeEnvState | dict[str, Any] | int) -> FrozenLakeEnvState:
        if isinstance(state, FrozenLakeEnvState):
            return state
        if isinstance(state, dict):
            return FrozenLakeEnvState.from_dict(state)

        state_index = coerce_frozen_lake_state_index(state)
        if state_index is None:
            raise ValueError(f"Unsupported Frozen Lake state payload: {state!r}")
        return FrozenLakeEnvState(state_index=state_index)

    def _state_count(self, base_env: object) -> int | None:
        return _state_count(base_env)

    def _build_wrapped_env(self, task: TaskDefinition, *, render_mode: str | None) -> gym.Env:
        map_rows = _map_from_task_config(task.config)
        success_rate = _parse_success_rate(task.config.get("success_rate"))
        env = gym.make(
            "FrozenLake-v1",
            desc=map_rows,
            is_slippery=bool(task.config.get("is_slippery", True)),
            success_rate=success_rate,
            render_mode=render_mode,
            # Keep episode limits explicit in RunConfig instead of Gymnasium's hidden default TimeLimit(100).
            max_episode_steps=-1,
        )

        wrapped_env: gym.Env = FrozenLakeRewardWrapper(env, task.reward_config)

        region = task.config.get("active_region")
        if isinstance(region, dict):
            terminate_on_exit = bool(task.config.get("enforce_region", True))
            outside_reward = float(task.config.get("outside_region_reward", -0.25))
            wrapped_env = FrozenLakeRegionWrapper(
                wrapped_env,
                region={
                    "row_min": int(region.get("row_min", 0)),
                    "row_max": int(region.get("row_max", 0)),
                    "col_min": int(region.get("col_min", 0)),
                    "col_max": int(region.get("col_max", 0)),
                },
                terminate_on_exit=terminate_on_exit,
                outside_reward=outside_reward,
            )

        raw_start_state = task.config.get("start_state")
        if raw_start_state is not None:
            start_state = coerce_frozen_lake_state_index(raw_start_state)
            if start_state is None:
                raise ValueError(f"Unsupported Frozen Lake start_state payload: {raw_start_state!r}")
            wrapped_env = FrozenLakeStartStateWrapper(wrapped_env, start_state=start_state)

        return wrapped_env


def _state_count(base_env: object) -> int | None:
    nrow = getattr(base_env, "nrow", None)
    ncol = getattr(base_env, "ncol", None)
    if isinstance(nrow, Integral) and isinstance(ncol, Integral):
        return int(nrow) * int(ncol)
    return None
