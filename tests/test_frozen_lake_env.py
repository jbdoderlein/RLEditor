from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gymnasium as gym
import pytest
from gymnasium.spaces import Discrete
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLayout, QSizePolicy

from rleditor.core.models import EpisodeMoment, EpisodeStep, EpisodeTrace, TaskDefinition, TaskSnapshot
from rleditor.plugins.builtin.frozen_lake import (
    FrozenLakeBackend,
    FrozenLakeEpisodeReplayWidget,
    FrozenLakeTaskEditorWidget,
)
from rleditor.plugins.builtin.frozen_lake_env import (
    FrozenLakeEnvState,
    FrozenLakeExtendedEnv,
    FrozenLakeRegionWrapper,
    FrozenLakeRewardWrapper,
    FrozenLakeStartStateWrapper,
    TILE_FROZEN,
    TILE_GOAL,
    TILE_HOLE,
    TILE_START,
    _coerce_reward_config,
    _generate_random_map_desc,
    _map_from_task_config,
    _normalize_map_desc,
    _parse_size,
    _parse_success_rate,
    _state_count,
    _tile_at_state,
    coerce_frozen_lake_state_index,
)


class _FakeFrozenLakeBase(gym.Env):
    action_space = Discrete(4)
    observation_space = Discrete(4)

    def __init__(
        self,
        *,
        observation: int = 0,
        nrow: int = 2,
        ncol: int = 2,
        desc=None,
    ) -> None:
        super().__init__()
        self.observation = observation
        self.nrow = nrow
        self.ncol = ncol
        self.desc = desc if desc is not None else [[b"S", b"H"], [b"F", b"G"]]
        self.s = 0
        self.lastaction = 2

    def reset(self, *, seed: int | None = None, options=None):
        _ = seed
        _ = options
        return self.s, {"reset": True}

    def step(self, action: int):
        _ = action
        return self.observation, 7.0, False, False, {"source": "fake"}


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


def test_frozen_lake_extended_env_starts_on_map_start_tile() -> None:
    task = _task_definition()
    task.config["map_desc"] = [
        "FFFF",
        "FSFF",
        "FFFF",
        "FFFG",
    ]

    env = FrozenLakeExtendedEnv(task)
    try:
        observation, _info = env.reset(seed=9)
        assert int(observation) == 5
        assert env.export_state().state_index == 5
    finally:
        env.close()


def test_frozen_lake_extended_env_does_not_apply_gym_time_limit() -> None:
    task = _task_definition()
    task.config["is_slippery"] = False
    task.config["map_desc"] = [
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ]

    env = FrozenLakeExtendedEnv(task)
    try:
        env.reset(seed=9)
        terminated = False
        truncated = False
        for _step in range(101):
            _observation, _reward, terminated, truncated, _info = env.step(0)

        assert terminated is False
        assert truncated is False
    finally:
        env.close()


def test_frozen_lake_random_map_hole_probability_zero_has_no_holes() -> None:
    map_desc = _generate_random_map_desc(size=8, hole_probability=0.0)

    assert all("H" not in row for row in map_desc)


def test_frozen_lake_random_map_rejects_impossible_hole_probability() -> None:
    with pytest.raises(ValueError, match="hole_probability"):
        _generate_random_map_desc(size=4, hole_probability=1.0)


def test_frozen_lake_config_helpers_normalize_sizes_maps_and_rewards() -> None:
    assert _parse_size(1) == 2
    assert _parse_size("8x8") == 8
    assert _parse_size("6x") == 6
    assert _parse_size("invalid", fallback=5) == 5

    assert len(_map_from_task_config({"size": 8})) == 8
    custom_map = _map_from_task_config({"size": 3})
    assert custom_map == ["SFF", "FFF", "FFG"]

    normalized = _normalize_map_desc(["SXS", "G", "GGG"], expected_size=3)
    assert normalized == [
        [TILE_START, TILE_FROZEN, TILE_FROZEN],
        [TILE_GOAL, TILE_FROZEN, TILE_FROZEN],
        [TILE_FROZEN, TILE_FROZEN, TILE_FROZEN],
    ]
    assert _normalize_map_desc([], expected_size=2) == [
        [TILE_START, TILE_FROZEN],
        [TILE_FROZEN, TILE_GOAL],
    ]

    rewards = _coerce_reward_config(
        {
            "F": "0.5",
            "tile:H": "-2.0",
            "tile:G": object(),
            "unknown": 99.0,
        }
    )
    assert rewards["tile:F"] == pytest.approx(0.5)
    assert rewards["tile:H"] == pytest.approx(-2.0)
    assert rewards["tile:G"] == pytest.approx(1.0)


def test_frozen_lake_success_rate_and_state_helpers_reject_invalid_values() -> None:
    class _BadItem:
        def item(self):
            raise RuntimeError("cannot scalarize")

    assert _parse_success_rate(None) == pytest.approx(1.0 / 3.0)
    assert _parse_success_rate("0.75") == pytest.approx(0.75)
    with pytest.raises(ValueError, match="success_rate"):
        _parse_success_rate("bad")
    with pytest.raises(ValueError, match="success_rate"):
        _parse_success_rate(float("inf"))

    assert coerce_frozen_lake_state_index("7") == 7
    assert coerce_frozen_lake_state_index(_BadItem()) is None
    assert coerce_frozen_lake_state_index("7.0") is None

    assert _state_count(type("StateCount", (), {"nrow": 2, "ncol": 3})()) == 6
    assert _state_count(type("NoStateCount", (), {"nrow": "2", "ncol": 3})()) is None


def test_frozen_lake_tile_lookup_handles_bytes_decode_and_missing_desc() -> None:
    class _BadDecode:
        def decode(self, encoding: str):
            _ = encoding
            raise UnicodeError("bad tile")

        def __str__(self) -> str:
            return "Z"

    assert _tile_at_state(type("NoDesc", (), {})(), 0) == TILE_FROZEN
    assert _tile_at_state(_FakeFrozenLakeBase(), 1) == TILE_HOLE
    assert _tile_at_state(_FakeFrozenLakeBase(desc=[[_BadDecode()]]), 0) == "Z"
    assert _tile_at_state(_FakeFrozenLakeBase(), 99) == TILE_FROZEN


def test_frozen_lake_reward_wrapper_maps_tile_rewards_and_preserves_base_reward_info() -> None:
    wrapper = FrozenLakeRewardWrapper(
        _FakeFrozenLakeBase(observation=1),
        {"tile:H": -3.0},
    )

    observation, reward, terminated, truncated, info = wrapper.step(0)

    assert observation == 1
    assert reward == pytest.approx(-3.0)
    assert terminated is False
    assert truncated is False
    assert info["base_reward"] == pytest.approx(7.0)
    assert info["reward_tile"] == TILE_HOLE


def test_frozen_lake_region_wrapper_marks_and_optionally_terminates_outside_region() -> None:
    wrapper = FrozenLakeRegionWrapper(
        _FakeFrozenLakeBase(observation=2),
        region={"row_min": 0, "row_max": 0, "col_min": 0, "col_max": 1},
        terminate_on_exit=True,
        outside_reward=-0.5,
    )

    observation, reward, terminated, _truncated, info = wrapper.step(0)

    assert observation == 2
    assert reward == pytest.approx(-0.5)
    assert terminated is True
    assert info["outside_region"] is True

    ncol_missing_wrapper = FrozenLakeRegionWrapper(
        _FakeFrozenLakeBase(observation=2, ncol=0),
        region={"row_min": 0, "row_max": 0, "col_min": 0, "col_max": 1},
        terminate_on_exit=True,
        outside_reward=-0.5,
    )
    _observation, reward, terminated, _truncated, info = ncol_missing_wrapper.step(0)

    assert reward == pytest.approx(7.0)
    assert terminated is False
    assert "outside_region" not in info


def test_frozen_lake_start_state_wrapper_validates_range_and_resets_last_action() -> None:
    base_env = _FakeFrozenLakeBase()
    wrapper = FrozenLakeStartStateWrapper(base_env, start_state=3)

    observation, info = wrapper.reset(seed=5, options={"ignored": True})

    assert observation == 3
    assert info["reset"] is True
    assert info["start_state_override"] == 3
    assert base_env.s == 3
    assert base_env.lastaction is None

    with pytest.raises(ValueError, match="out of range"):
        FrozenLakeStartStateWrapper(_FakeFrozenLakeBase(), start_state=4).reset(seed=5)


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


def test_frozen_lake_extended_env_rejects_invalid_state_payloads() -> None:
    assert FrozenLakeExtendedEnv.from_task_snapshot(None) is None

    task = _task_definition()
    task.config["start_state"] = "not-a-state"
    with pytest.raises(ValueError, match="start_state"):
        FrozenLakeExtendedEnv(task)

    env = FrozenLakeExtendedEnv(_task_definition())
    try:
        env.reset(seed=3)
        with pytest.raises(ValueError, match="Unsupported Frozen Lake state payload"):
            env.import_state(object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="out of range"):
            env.import_state(999)
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


def test_frozen_lake_episode_replay_grid_scrolls_large_maps() -> None:
    _app()
    widget = FrozenLakeEpisodeReplayWidget()
    map_desc = [
        "S" + ("F" * 15),
        *(["F" * 16] * 14),
        ("F" * 15) + "G",
    ]

    widget._ensure_map_grid(map_desc)

    assert widget.grid_scroll.widget() is widget.grid_host
    assert widget.grid_layout.sizeConstraint() == QLayout.SizeConstraint.SetFixedSize
    assert widget.grid_host.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    assert widget.grid_host.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
    assert widget._cells[0].maximumWidth() == 42
    assert widget._cells[0].maximumHeight() == 42


def test_frozen_lake_episode_replay_preview_scales_to_label_bounds() -> None:
    _app()
    widget = FrozenLakeEpisodeReplayWidget()
    widget.render_label.setFixedSize(120, 80)
    pixmap = QPixmap(512, 512)

    widget._set_render_pixmap(pixmap)

    scaled = widget.render_label.pixmap()
    assert scaled is not None
    assert scaled.width() <= widget.render_label.width()
    assert scaled.height() <= widget.render_label.height()


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


def test_frozen_lake_task_editor_limits_hole_probability_to_valid_generation_range() -> None:
    _app()
    widget = FrozenLakeTaskEditorWidget(_task_definition(), lambda _task: None)

    assert widget.hole_probability.minimum() == pytest.approx(0.0)
    assert widget.hole_probability.maximum() == pytest.approx(0.95)


def test_frozen_lake_task_editor_moving_start_clears_start_override() -> None:
    _app()
    task = _task_definition()
    task.config["start_state"] = 0
    changed_tasks: list[TaskDefinition] = []
    widget = FrozenLakeTaskEditorWidget(task, changed_tasks.append)

    widget.paint_combo.setCurrentIndex(widget.paint_combo.findData(TILE_START))
    widget._paint_cell(1, 1)

    assert widget._task.config["map_desc"] == [
        "FFFF",
        "FSFH",
        "FFFH",
        "HFFG",
    ]
    assert "start_state" not in widget._task.config
    assert not widget.start_override_checkbox.isChecked()

    env = FrozenLakeExtendedEnv(task)
    try:
        observation, _info = env.reset(seed=9)
        assert int(observation) == 5
    finally:
        env.close()


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
