from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.core.models import EpisodeMoment, EpisodeStep, EpisodeTrace, TaskDefinition, TaskSnapshot
from rleditor.plugins.builtin import mujoco_env
from rleditor.plugins.builtin.mujoco import MujocoBackend, MujocoEpisodeReplayWidget, MujocoTaskEditorWidget
from rleditor.plugins.builtin.mujoco_env import MujocoEnvState, MujocoExtendedEnv
from rleditor.plugins.registry import PluginRegistry, register_builtin_plugins


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeMujocoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_id: str, kwargs: dict[str, object]) -> None:
        super().__init__()
        self.env_id = env_id
        self.kwargs = kwargs
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.data = SimpleNamespace(
            qpos=np.array([1.0, 2.0], dtype=np.float64),
            qvel=np.array([0.1, 0.2], dtype=np.float64),
            ctrl=np.array([0.0, 0.0], dtype=np.float64),
            time=0.0,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        _ = seed, options
        self.data.qpos[:] = [1.0, 2.0]
        self.data.qvel[:] = [0.1, 0.2]
        self.data.ctrl[:] = [0.0, 0.0]
        self.data.time = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        self.data.qpos[:] = self.data.qpos + 0.1
        self.data.time += 0.01
        return self._get_obs(), 1.0, False, False, {}

    def set_state(self, qpos, qvel) -> None:
        assert qpos.shape == (2,)
        assert qvel.shape == (2,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def render(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = 64
        frame[:, :, 1] = 120
        frame[:, :, 2] = 180
        return frame


class _FakeInvertedDoublePendulumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_id: str, kwargs: dict[str, object]) -> None:
        super().__init__()
        self.env_id = env_id
        self.kwargs = kwargs
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.data = SimpleNamespace(
            qpos=np.array([0.0, 0.1, -0.1], dtype=np.float64),
            qvel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            ctrl=np.array([0.0], dtype=np.float64),
            time=0.0,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        _ = seed, options
        self.data.qpos[:] = [0.0, 0.1, -0.1]
        self.data.qvel[:] = [0.0, 0.0, 0.0]
        self.data.ctrl[:] = [0.0]
        self.data.time = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        self.data.time += 0.01
        return self._get_obs(), 10.0, False, False, {"reward_survive": 10.0}

    def set_state(self, qpos, qvel) -> None:
        assert qpos.shape == (3,)
        assert qvel.shape == (3,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel, np.zeros(3, dtype=np.float64)])


def _patch_gym_make(monkeypatch) -> None:
    def fake_make(env_id: str, **kwargs):
        if env_id == "InvertedDoublePendulum-v5":
            return _FakeInvertedDoublePendulumEnv(env_id, kwargs)
        return _FakeMujocoEnv(env_id, kwargs)

    monkeypatch.setattr(mujoco_env.gym, "make", fake_make)


def test_mujoco_extended_env_exports_and_imports_simulator_state(monkeypatch) -> None:
    _patch_gym_make(monkeypatch)
    task = TaskDefinition(
        environment_id="mujoco",
        name="MuJoCo Test",
        config={"env_id": "FakeMuJoCo-v0", "make_kwargs": {"frame_skip": 2}},
    )

    env = MujocoExtendedEnv(task)
    env.reset(seed=7)
    env.step(np.array([0.3, -0.2], dtype=np.float32))

    state = env.export_state()
    assert state.qpos == [1.1, 2.1]
    assert state.qvel == [0.1, 0.2]
    assert state.ctrl == [0.30000001192092896, -0.20000000298023224]
    assert state.last_action == [0.30000001192092896, -0.20000000298023224]
    assert state.observation == [1.1, 2.1, 0.1, 0.2]

    restored_state = MujocoEnvState.from_dict(state.to_dict())
    env.import_state(
        MujocoEnvState(
            qpos=[4.0, 5.0],
            qvel=[0.4, 0.5],
            time=2.5,
            ctrl=[0.9, -0.9],
        )
    )

    base_env = env.unwrapped
    assert restored_state.qpos == [1.1, 2.1]
    assert base_env.data.qpos.tolist() == [4.0, 5.0]
    assert base_env.data.qvel.tolist() == [0.4, 0.5]
    assert base_env.data.ctrl.tolist() == [0.9, -0.9]
    assert base_env.data.time == 2.5


def test_mujoco_extended_env_can_start_from_initial_state(monkeypatch) -> None:
    _patch_gym_make(monkeypatch)
    task = TaskDefinition(
        environment_id="mujoco",
        name="MuJoCo Initial State",
        config={
            "env_id": "FakeMuJoCo-v0",
            "initial_state": {
                "qpos": [3.0, 4.0],
                "qvel": [0.3, 0.4],
            },
        },
    )

    env = MujocoExtendedEnv(task)
    observation, info = env.reset(seed=11)

    assert info["initial_state_override"] is True
    assert observation.tolist() == [3.0, 4.0, 0.3, 0.4]


def test_mujoco_extended_env_can_be_recreated_from_task_snapshot(monkeypatch) -> None:
    _patch_gym_make(monkeypatch)
    snapshot = TaskSnapshot(
        environment_id="mujoco",
        task_name="MuJoCo Snapshot",
        task_config={"env_id": "FakeMuJoCo-v0"},
    )

    env = MujocoExtendedEnv.from_task_snapshot(snapshot)

    assert env is not None
    assert env.task_definition.name == "MuJoCo Snapshot"


def test_mujoco_backend_is_registered_without_requiring_mujoco_install() -> None:
    registry = PluginRegistry()

    register_builtin_plugins(registry)
    plugin = registry.get_environment_plugin("mujoco")
    task = MujocoBackend().default_task()

    assert plugin.display_name == "Inverted Double Pendulum"
    assert plugin.gui_extension is not None
    assert task.environment_id == "mujoco"
    assert task.config["env_id"] == "InvertedDoublePendulum-v5"
    assert task.reward_config["upright_angle_threshold"] == 0.2
    assert task.metadata["control_type"] == "continuous"


def test_mujoco_task_editor_updates_inverted_double_pendulum_fields() -> None:
    _app()
    task = MujocoBackend().default_task()
    changed: list[TaskDefinition] = []
    widget = MujocoTaskEditorWidget(task, changed.append)

    widget.cart_position_spin.setValue(1.25)
    widget.cart_velocity_spin.setValue(-0.75)
    widget.upright_threshold_spin.setValue(0.35)

    assert changed
    assert task.config["env_id"] == "InvertedDoublePendulum-v5"
    assert task.config["initial_state"] == {
        "cart_position": 1.25,
        "cart_velocity": -0.75,
    }
    assert task.reward_config["upright_angle_threshold"] == 0.35
    assert task.metadata["preferred_algorithm"] == "sb3_ppo"
    assert task.metadata["supported_algorithms"] == ["sb3_ppo"]


def test_mujoco_extended_env_can_start_from_cart_position_and_velocity(monkeypatch) -> None:
    _patch_gym_make(monkeypatch)
    task = MujocoBackend().default_task()
    task.config["initial_state"] = {
        "cart_position": 1.25,
        "cart_velocity": -0.75,
    }

    env = MujocoExtendedEnv(task)
    observation, info = env.reset(seed=11)

    assert info["initial_state_override"] is True
    assert observation.tolist()[:6] == [1.25, 0.1, -0.1, -0.75, 0.0, 0.0]


def test_mujoco_extended_env_reports_upright_angle_threshold_without_terminating(monkeypatch) -> None:
    _patch_gym_make(monkeypatch)
    task = MujocoBackend().default_task()
    task.reward_config["upright_angle_threshold"] = 0.05

    env = MujocoExtendedEnv(task)
    env.reset(seed=11)
    _observation, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))

    assert reward == 10.0
    assert terminated is False
    assert truncated is False
    assert info["reward_survive"] == 10.0
    assert info["upright_angle_threshold"] == 0.05
    assert info["upright_angle_healthy"] is False


def test_mujoco_replay_widget_renders_frame_and_simulator_state(monkeypatch) -> None:
    _app()
    _patch_gym_make(monkeypatch)
    state0 = MujocoEnvState(qpos=[1.0, 2.0], qvel=[0.1, 0.2], time=0.0)
    state1 = MujocoEnvState(
        qpos=[1.1, 2.1],
        qvel=[0.1, 0.2],
        time=0.01,
        ctrl=[0.3, -0.2],
        last_action=[0.3, -0.2],
    )
    trace = EpisodeTrace(
        episode_id=1,
        run_id="run_mujoco",
        total_reward=1.0,
        success=True,
        initial_observation=[1.0, 2.0, 0.1, 0.2],
        steps=[
            EpisodeStep(
                t=0,
                observation=[1.0, 2.0, 0.1, 0.2],
                action=[0.3, -0.2],
                reward=1.0,
                next_observation=[1.1, 2.1, 0.1, 0.2],
                terminated=False,
            )
        ],
        moments=[
            EpisodeMoment(episode_id=1, moment_index=0, observation=[1.0, 2.0, 0.1, 0.2], restorable_env_state=state0),
            EpisodeMoment(
                episode_id=1,
                moment_index=1,
                observation=[1.1, 2.1, 0.1, 0.2],
                action_taken=[0.3, -0.2],
                reward=1.0,
                restorable_env_state=state1,
            ),
        ],
        task_snapshot=TaskSnapshot(
            environment_id="mujoco",
            task_name="MuJoCo Render Test",
            task_config={"env_id": "FakeMuJoCo-v0"},
        ),
    )
    widget = MujocoEpisodeReplayWidget()

    widget.set_frame(trace, 1)
    pixmap = widget.render_label.pixmap()

    assert pixmap is not None
    assert not pixmap.isNull()
    assert widget.qpos_label.text() == "[1.1, 2.1]"
    assert widget.qvel_label.text() == "[0.1, 0.2]"
    assert widget.ctrl_label.text() == "[0.3, -0.2]"
    assert "0.3" in widget.action_label.text()
