from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLayout, QSizePolicy

from rleditor.core.models import EpisodeMoment, EpisodeStep, EpisodeTrace, TaskDefinition, TaskSnapshot
from rleditor.plugins.builtin.frozen_lake import FrozenLakeBackend, FrozenLakeTaskEditorWidget
from rleditor.plugins.builtin.frozen_lake_env import FrozenLakeEnvState, FrozenLakeExtendedEnv


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _task_definition() -> TaskDefinition:
    return TaskDefinition(
        environment_id="frozen_lake",
        name="Frozen Lake Test",
        config={
            "size": 4,
            "is_slippery": False,
            "map_desc": [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG",
            ],
        },
        reward_config={
            "tile:F": 0.0,
            "tile:H": -1.0,
            "tile:S": 0.0,
            "tile:G": 1.0,
        },
    )


def test_frozen_lake_extended_env_supports_export_import_and_reinstantiate() -> None:
    env = FrozenLakeExtendedEnv(_task_definition())

    try:
        observation, _info = env.reset(seed=7)
        exported_state = env.export_state()

        assert exported_state.state_index == int(observation)

        env.import_state(FrozenLakeEnvState(state_index=5, last_action=2))
        restored_state = env.export_state()

        assert restored_state.state_index == 5
        assert restored_state.last_action == 2

        replay_env = env.reinstantiate(render_mode="rgb_array")
        try:
            replay_observation, _replay_info = replay_env.reset(seed=7)
            assert int(replay_observation) == int(observation)
            replay_env.import_state(exported_state)
            assert replay_env.export_state() == exported_state
        finally:
            replay_env.close()
    finally:
        env.close()


def test_frozen_lake_extended_env_honors_start_state_override() -> None:
    task = _task_definition()
    task.config["start_state"] = 6

    env = FrozenLakeExtendedEnv(task)
    try:
        observation, info = env.reset(seed=9)
        assert int(observation) == 6
        assert info["start_state_override"] == 6
        assert env.export_state().state_index == 6
    finally:
        env.close()


def test_frozen_lake_extended_env_honors_slippery_success_rate() -> None:
    task = _task_definition()
    task.config["is_slippery"] = True
    task.config["success_rate"] = 0.8

    env = FrozenLakeExtendedEnv(task)
    try:
        transitions = env.unwrapped.P[6][2]
        intended_probabilities = [
            probability
            for probability, next_state, _reward, _terminated in transitions
            if next_state == 7
        ]

        assert sum(probability for probability, *_rest in transitions) == pytest.approx(1.0)
        assert intended_probabilities == [pytest.approx(0.8)]
    finally:
        env.close()


def test_frozen_lake_extended_env_rejects_invalid_success_rate() -> None:
    task = _task_definition()
    task.config["success_rate"] = 1.5

    with pytest.raises(ValueError, match="success_rate"):
        FrozenLakeExtendedEnv(task)


def test_frozen_lake_extended_env_can_be_rebuilt_from_task_snapshot() -> None:
    task = _task_definition()
    snapshot = TaskSnapshot(
        environment_id=task.environment_id,
        task_name=task.name,
        task_id=task.task_id,
        task_config=dict(task.config),
        reward_config=dict(task.reward_config),
        termination_config=dict(task.termination_config),
        metadata={"source": "unit_test"},
    )

    env = FrozenLakeExtendedEnv.from_task_snapshot(snapshot)
    assert env is not None

    try:
        observation, _info = env.reset(seed=3)
        assert env.export_state().state_index == int(observation)
        assert env.task_definition.config["map_desc"] == task.config["map_desc"]
    finally:
        env.close()


def test_frozen_lake_backend_derives_start_state_from_episode_moment() -> None:
    task = _task_definition()
    trace = EpisodeTrace(
        episode_id=8,
        run_id="run_alpha",
        total_reward=0.0,
        success=False,
        initial_observation=0,
        steps=[
            EpisodeStep(
                t=0,
                observation=0,
                action=1,
                reward=0.0,
                next_observation=4,
                terminated=False,
            )
        ],
        moments=[
            EpisodeMoment(
                episode_id=8,
                moment_index=0,
                observation=0,
                restorable_env_state={"state_index": 0},
            ),
            EpisodeMoment(
                episode_id=8,
                moment_index=1,
                observation=4,
                action_taken=1,
                reward=0.0,
                restorable_env_state={"state_index": 4},
            ),
        ],
    )

    options = FrozenLakeBackend().derive_task_from_episode(task, trace, 1)
    assert options is not None
    assert options.config_updates["start_state"] == 4
    assert options.start_state == 4
    assert options.source_episode_id == 8
    assert options.source_moment_index == 1
    assert options.source_run_id == "run_alpha"


def test_frozen_lake_task_editor_grid_uses_fixed_spacing() -> None:
    _app()
    widget = FrozenLakeTaskEditorWidget(_task_definition(), lambda _task: None)

    assert widget.grid_layout.sizeConstraint() == QLayout.SizeConstraint.SetFixedSize
    assert widget.grid_host.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    assert widget.grid_host.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed


def test_frozen_lake_task_editor_accepts_custom_grid_size() -> None:
    _app()
    changed_tasks: list[TaskDefinition] = []
    widget = FrozenLakeTaskEditorWidget(_task_definition(), changed_tasks.append)

    widget.size_spin.setValue(5)

    assert len(widget._map) == 5
    assert all(len(row) == 5 for row in widget._map)
    assert widget._task.config["size"] == 5
    assert len(widget._task.config["map_desc"]) == 5
    assert widget.start_state_spin.maximum() == 24
    assert changed_tasks


def test_frozen_lake_task_editor_updates_success_rate() -> None:
    _app()
    task = _task_definition()
    task.config["is_slippery"] = True
    changed_tasks: list[TaskDefinition] = []
    widget = FrozenLakeTaskEditorWidget(task, changed_tasks.append)

    assert widget.success_rate_spin.isEnabled()

    widget.success_rate_spin.setValue(0.8)

    assert widget._task.config["success_rate"] == pytest.approx(0.8)

    widget.slippery_checkbox.setChecked(False)

    assert not widget.success_rate_spin.isEnabled()
    assert widget._task.config["is_slippery"] is False
