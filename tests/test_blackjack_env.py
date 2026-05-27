from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from rleditor.core.models import EpisodeMoment, EpisodeStep, EpisodeTrace, TaskDefinition, TaskSnapshot
from rleditor.plugins.builtin.blackjack import BlackjackBackend, BlackjackEpisodeReplayWidget
from rleditor.plugins.builtin.blackjack_env import (
    BlackjackEnvState,
    BlackjackExtendedEnv,
    _card_label,
    _coerce_card,
    coerce_blackjack_hand,
    coerce_blackjack_observation,
)
from rleditor.plugins.registry import PluginRegistry, register_builtin_plugins


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        ("A", 1),
        (" j ", 10),
        ("10", 10),
        (0, None),
        (11, None),
        ("2.5", None),
        (object(), None),
    ],
)
def test_blackjack_card_coercion_accepts_only_supported_cards(value: object, expected: int | None) -> None:
    assert _coerce_card(value) == expected


def test_blackjack_hand_observation_and_label_helpers_coerce_editor_payloads() -> None:
    assert coerce_blackjack_hand(["A", "K", 5]) == (1, 10, 5)
    assert coerce_blackjack_hand(("3", "7")) == (3, 7)
    assert coerce_blackjack_hand(["A"]) is None
    assert coerce_blackjack_hand(["A", "bad"]) is None
    assert coerce_blackjack_hand("A,K") is None

    assert coerce_blackjack_observation(["21", "10", 2]) == (21, 10, 1)
    assert coerce_blackjack_observation((18, 3, 0)) == (18, 3, 0)
    assert coerce_blackjack_observation((18, "bad", 0)) is None
    assert coerce_blackjack_observation((18, 3)) is None

    assert _card_label(1) == "A"
    assert _card_label(10) == "10"


def test_blackjack_state_from_dict_defaults_invalid_hands_and_coerces_metadata() -> None:
    state = BlackjackEnvState.from_dict(
        {
            "player_hand": ["bad"],
            "dealer_hand": [10],
            "last_action": "1",
            "terminated": "yes",
        }
    )

    assert state == BlackjackEnvState(
        player_hand=(10, 10),
        dealer_hand=(10, 7),
        last_action=1,
        terminated=True,
    )


def test_blackjack_extended_env_supports_export_import_and_reinstantiate() -> None:
    task = TaskDefinition(
        environment_id="blackjack",
        name="Blackjack Test",
        config={"natural": False, "sab": True},
    )
    env = BlackjackExtendedEnv(task)

    try:
        env.reset(seed=123)
        state = env.export_state()
        env.step(1)
        changed_state = env.export_state()
        assert changed_state.player_hand != ()

        restored_state = env.import_state(state)
        assert restored_state == state
        assert env.export_state() == state

        env2 = env.reinstantiate()
        try:
            env2.reset(seed=999)
            env2.import_state(state)
            assert env2.export_state() == state
        finally:
            env2.close()
    finally:
        env.close()


def test_blackjack_extended_env_honors_initial_state_override() -> None:
    task = TaskDefinition(
        environment_id="blackjack",
        name="Blackjack Derived",
        config={
            "natural": False,
            "sab": True,
            "initial_state": BlackjackEnvState(
                player_hand=(10, 1),
                dealer_hand=(10, 7),
            ).to_dict(),
        },
    )
    env = BlackjackExtendedEnv(task)

    try:
        observation, info = env.reset(seed=321)
        assert observation == (21, 10, 1)
        assert info["initial_state_override"] is True
        assert env.export_state().player_hand == (10, 1)
        assert env.export_state().dealer_hand == (10, 7)
    finally:
        env.close()


def test_blackjack_extended_env_ignores_unsupported_initial_state_payload() -> None:
    task = TaskDefinition(
        environment_id="blackjack",
        name="Blackjack Invalid Initial",
        config={"initial_state": "not a blackjack state"},
    )
    env = BlackjackExtendedEnv(task)

    try:
        observation, info = env.reset(seed=321)

        assert isinstance(observation, tuple)
        assert len(observation) == 3
        assert "initial_state_override" not in info
    finally:
        env.close()


def test_blackjack_extended_env_can_be_rebuilt_from_task_snapshot() -> None:
    snapshot = TaskSnapshot(
        environment_id="blackjack",
        task_name="Snapshot Task",
        task_config={
            "natural": True,
            "sab": False,
            "initial_state": BlackjackEnvState(
                player_hand=(9, 2),
                dealer_hand=(6, 10),
            ).to_dict(),
        },
    )

    env = BlackjackExtendedEnv.from_task_snapshot(snapshot)
    assert env is not None

    try:
        observation, _info = env.reset(seed=99)
        assert observation == (11, 6, 0)
    finally:
        env.close()


def test_blackjack_extended_env_from_task_snapshot_returns_none_without_snapshot() -> None:
    assert BlackjackExtendedEnv.from_task_snapshot(None) is None


def test_blackjack_extended_env_rejects_unsupported_import_payload() -> None:
    task = TaskDefinition(environment_id="blackjack", name="Blackjack Test")
    env = BlackjackExtendedEnv(task)

    try:
        with pytest.raises(ValueError, match="Unsupported Blackjack state payload"):
            env.import_state(object())  # type: ignore[arg-type]
    finally:
        env.close()


def test_blackjack_extended_env_export_state_requires_restorable_base_hands() -> None:
    task = TaskDefinition(environment_id="blackjack", name="Blackjack Test")
    env = BlackjackExtendedEnv(task)

    try:
        env.reset(seed=123)
        setattr(env.unwrapped, "player", ["bad"])

        with pytest.raises(RuntimeError, match="does not expose restorable"):
            env.export_state()
    finally:
        env.close()


def test_blackjack_extended_env_falls_back_to_hand_derived_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskDefinition(environment_id="blackjack", name="Blackjack Test")
    env = BlackjackExtendedEnv(task)

    try:
        env.reset(seed=123)
        monkeypatch.setattr(env.unwrapped, "_get_obs", lambda: ("bad",), raising=False)
        setattr(env.unwrapped, "player", [1, 10])
        setattr(env.unwrapped, "dealer", [9, 7])

        assert env._get_observation() == (21, 9, 1)
    finally:
        env.close()


def test_blackjack_extended_env_normalizes_legacy_four_item_step_result(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskDefinition(environment_id="blackjack", name="Blackjack Test")
    env = BlackjackExtendedEnv(task)

    try:
        def legacy_step(_action: object) -> tuple[tuple[int, int, int], float, bool, dict[str, object]]:
            return (20, 10, 0), 1.0, True, {"legacy": True}

        monkeypatch.setattr(env.env, "step", legacy_step)

        observation, reward, terminated, truncated, info = env.step(1)

        assert observation == (20, 10, 0)
        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert info == {"legacy": True}
        assert env.export_state().last_action == 1
        assert env.export_state().terminated is True
    finally:
        env.close()


def test_blackjack_backend_derives_task_from_restorable_episode_moment() -> None:
    state = BlackjackEnvState(player_hand=(10, 5), dealer_hand=(9, 10), last_action=1)
    task = TaskDefinition(
        environment_id="blackjack",
        name="Blackjack Source",
        config={"natural": False, "sab": True},
    )
    trace = EpisodeTrace(
        episode_id=7,
        run_id="run_blackjack",
        total_reward=0.0,
        success=False,
        moments=[
            EpisodeMoment(
                episode_id=7,
                moment_index=0,
                observation=(15, 9, 0),
                restorable_env_state=state,
            )
        ],
    )

    options = BlackjackBackend().derive_task_from_episode(task, trace, 0)

    assert options is not None
    assert options.config_updates["initial_state"] == state.to_dict()
    assert options.source_episode_id == 7
    assert options.source_run_id == "run_blackjack"


def test_builtin_registry_registers_blackjack_plugin() -> None:
    registry = PluginRegistry()
    register_builtin_plugins(registry)

    plugin = registry.get_environment_plugin("blackjack")

    assert plugin.display_name == "Blackjack"
    assert plugin.backend.default_task().environment_id == "blackjack"


def test_blackjack_replay_widget_renders_gym_frame_from_restorable_state() -> None:
    _app()
    initial_state = BlackjackEnvState(player_hand=(10, 5), dealer_hand=(9, 10))
    next_state = BlackjackEnvState(player_hand=(10, 5, 6), dealer_hand=(9, 10), last_action=1)
    trace = EpisodeTrace(
        episode_id=1,
        run_id="run_blackjack_render",
        total_reward=0.0,
        success=False,
        initial_observation=(15, 9, 0),
        steps=[
            EpisodeStep(
                t=0,
                observation=(15, 9, 0),
                action=1,
                reward=0.0,
                next_observation=(21, 9, 0),
                terminated=False,
            )
        ],
        moments=[
            EpisodeMoment(
                episode_id=1,
                moment_index=0,
                observation=(15, 9, 0),
                restorable_env_state=initial_state,
            ),
            EpisodeMoment(
                episode_id=1,
                moment_index=1,
                observation=(21, 9, 0),
                action_taken=1,
                reward=0.0,
                restorable_env_state=next_state,
            ),
        ],
        task_snapshot=TaskSnapshot(
            environment_id="blackjack",
            task_name="Blackjack Render Test",
            task_config={"natural": False, "sab": True},
        ),
    )
    widget = BlackjackEpisodeReplayWidget()

    widget.set_frame(trace, 1)
    pixmap = widget.render_label.pixmap()

    assert pixmap is not None
    assert not pixmap.isNull()
