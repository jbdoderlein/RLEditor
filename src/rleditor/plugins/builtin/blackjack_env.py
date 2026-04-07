from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any, cast

import gymnasium as gym
from gymnasium.envs.toy_text.blackjack import sum_hand, usable_ace

from rleditor.core.models import TaskDefinition, TaskSnapshot


def _coerce_card(value: object) -> int | None:
    if isinstance(value, Integral):
        card = int(value)
    elif isinstance(value, str):
        text = value.strip().upper()
        if text == "A":
            return 1
        if text in {"J", "Q", "K"}:
            return 10
        if not text.isdigit():
            return None
        card = int(text)
    else:
        return None
    if 1 <= card <= 10:
        return card
    return None


def coerce_blackjack_hand(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None

    cards: list[int] = []
    for item in value:
        card = _coerce_card(item)
        if card is None:
            return None
        cards.append(card)
    if len(cards) < 2:
        return None
    return tuple(cards)


def coerce_blackjack_observation(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        player_sum = int(value[0])
        dealer_card = int(value[1])
        usable = int(value[2])
    except (TypeError, ValueError):
        return None
    return player_sum, dealer_card, 1 if usable else 0


def _card_label(card: int) -> str:
    if card == 1:
        return "A"
    return str(card)


@dataclass(slots=True, frozen=True)
class BlackjackEnvState:
    player_hand: tuple[int, ...]
    dealer_hand: tuple[int, ...]
    last_action: int | None = None
    terminated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "player_hand": list(self.player_hand),
            "dealer_hand": list(self.dealer_hand),
            "last_action": self.last_action,
            "terminated": self.terminated,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BlackjackEnvState:
        player_hand = coerce_blackjack_hand(payload.get("player_hand")) or (10, 10)
        dealer_hand = coerce_blackjack_hand(payload.get("dealer_hand")) or (10, 7)
        last_action = payload.get("last_action")
        return cls(
            player_hand=player_hand,
            dealer_hand=dealer_hand,
            last_action=int(last_action) if last_action is not None else None,
            terminated=bool(payload.get("terminated", False)),
        )


class BlackjackExtendedEnv(gym.Wrapper):
    """Blackjack environment extended with full-hand state import/export hooks."""

    def __init__(self, task: TaskDefinition, *, render_mode: str | None = None) -> None:
        self._task = deepcopy(task)
        self._render_mode = render_mode
        self._last_action: int | None = None
        self._terminated = False
        env = self._build_env(self._task, render_mode=render_mode)
        super().__init__(env)

    @classmethod
    def from_task_snapshot(
        cls,
        task_snapshot: TaskSnapshot | None,
        *,
        render_mode: str | None = None,
    ) -> BlackjackExtendedEnv | None:
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
        initial_state = self._coerce_optional_initial_state(self._task.config.get("initial_state"))
        if initial_state is not None:
            self.import_state(initial_state)
            observation = self._get_observation()
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

        self._last_action = int(action) if isinstance(action, Integral) else None
        self._terminated = bool(terminated) or bool(truncated)
        return observation, reward, terminated, truncated, info

    def export_state(self) -> BlackjackEnvState:
        base_env = self._ensure_base_env_state()
        player_hand = coerce_blackjack_hand(getattr(base_env, "player", None))
        dealer_hand = coerce_blackjack_hand(getattr(base_env, "dealer", None))
        if player_hand is None or dealer_hand is None:
            raise RuntimeError("Blackjack environment does not expose restorable player/dealer hands")
        return BlackjackEnvState(
            player_hand=player_hand,
            dealer_hand=dealer_hand,
            last_action=self._last_action,
            terminated=self._terminated,
        )

    def import_state(self, state: BlackjackEnvState | dict[str, Any]) -> BlackjackEnvState:
        resolved_state = self._coerce_env_state(state)
        base_env = self._ensure_base_env_state()
        setattr(base_env, "player", list(resolved_state.player_hand))
        setattr(base_env, "dealer", list(resolved_state.dealer_hand))
        self._last_action = resolved_state.last_action
        self._terminated = resolved_state.terminated
        self._sync_dealer_render_attrs(base_env, resolved_state.dealer_hand)
        return resolved_state

    def reinstantiate(self, *, render_mode: str | None = None) -> BlackjackExtendedEnv:
        target_render_mode = self._render_mode if render_mode is None else render_mode
        return BlackjackExtendedEnv(self._task, render_mode=target_render_mode)

    def _build_env(self, task: TaskDefinition, *, render_mode: str | None) -> gym.Env:
        return gym.make(
            "Blackjack-v1",
            natural=bool(task.config.get("natural", False)),
            sab=bool(task.config.get("sab", True)),
            render_mode=render_mode,
        )

    def _get_observation(self) -> tuple[int, int, int]:
        base_env = self._ensure_base_env_state()
        get_obs = getattr(base_env, "_get_obs", None)
        if callable(get_obs):
            observation = get_obs()
            coerced = coerce_blackjack_observation(observation)
            if coerced is not None:
                return coerced

        player = cast(list[int], getattr(base_env, "player", [10, 10]))
        dealer = cast(list[int], getattr(base_env, "dealer", [10, 7]))
        return int(sum_hand(player)), int(dealer[0]), int(usable_ace(player))

    def _ensure_base_env_state(self) -> object:
        base_env = getattr(self, "unwrapped", self)
        if not hasattr(base_env, "player") or not hasattr(base_env, "dealer"):
            self.env.reset()
        return base_env

    def _coerce_env_state(self, state: BlackjackEnvState | dict[str, Any]) -> BlackjackEnvState:
        if isinstance(state, BlackjackEnvState):
            return state
        if isinstance(state, dict):
            return BlackjackEnvState.from_dict(state)
        raise ValueError(f"Unsupported Blackjack state payload: {state!r}")

    def _coerce_optional_initial_state(self, state: object) -> BlackjackEnvState | None:
        if state is None:
            return None
        if isinstance(state, BlackjackEnvState):
            return state
        if isinstance(state, dict):
            return BlackjackEnvState.from_dict(state)
        return None

    def _sync_dealer_render_attrs(self, base_env: object, dealer_hand: tuple[int, ...]) -> None:
        showing_card = dealer_hand[0] if dealer_hand else 10
        setattr(base_env, "dealer_top_card_suit", "S")
        setattr(base_env, "dealer_top_card_value_str", _card_label(showing_card))
