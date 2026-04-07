from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.core.models import EpisodeMoment, EpisodeStep, EpisodeTrace, TaskDefinition, TaskSnapshot
from rleditor.plugins.builtin.blackjack import BlackjackBackend, BlackjackEpisodeReplayWidget
from rleditor.plugins.builtin.blackjack_env import BlackjackEnvState, BlackjackExtendedEnv
from rleditor.plugins.registry import PluginRegistry, register_builtin_plugins


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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
