from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Protocol, cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel, QTextEdit, QVBoxLayout

from rleditor.core.models import EpisodeTrace, TaskDefinition, TaskDerivationOptions
from rleditor.plugins.base import EnvironmentPlugin, EpisodeReplayWidget
from rleditor.plugins.builtin.mujoco_env import MujocoEnvState, MujocoExtendedEnv


COMMON_MUJOCO_ENV_IDS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Humanoid-v5",
    "Reacher-v5",
    "Pusher-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
]


def _ensure_mujoco_metadata(task: TaskDefinition) -> None:
    task.metadata["control_type"] = "continuous"
    task.metadata["preferred_algorithm"] = "random"
    task.metadata["supported_algorithms"] = ["random"]
    task.metadata["state_transfer"] = "best_effort_mujoco_qpos_qvel"


class _FrameArrayLike(Protocol):
    shape: tuple[int, int, int]

    def tobytes(self) -> bytes: ...


def _short_sequence(values: object, *, max_items: int = 6) -> str:
    if values is None:
        return "-"
    if hasattr(values, "tolist") and callable(getattr(values, "tolist")):
        try:
            values = values.tolist()
        except Exception:
            return str(values)
    if not isinstance(values, list | tuple):
        return str(values)
    rendered = [f"{float(value):.4g}" for value in values[:max_items]]
    suffix = " ..." if len(values) > max_items else ""
    return "[" + ", ".join(rendered) + suffix + "]"


def _state_from_trace(trace: EpisodeTrace, moment_index: int) -> MujocoEnvState | None:
    if 0 <= moment_index < len(trace.moments):
        state = trace.moments[moment_index].restorable_env_state
        if isinstance(state, MujocoEnvState):
            return state
        if isinstance(state, dict):
            try:
                return MujocoEnvState.from_dict(state)
            except (TypeError, ValueError):
                return None
    return None


def _state_cache_key(state: MujocoEnvState | None) -> object:
    if state is None:
        return None
    return (
        tuple(state.qpos),
        tuple(state.qvel),
        state.time,
        None if state.ctrl is None else tuple(state.ctrl),
        repr(state.last_action),
        state.terminated,
    )


class MujocoBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="mujoco",
            name="MuJoCo HalfCheetah Default",
            task_id="task_mujoco_half_cheetah_default",
            config={
                "env_id": "HalfCheetah-v5",
                "render_mode": None,
                "make_kwargs": {},
            },
            metadata={
                "control_type": "continuous",
                "preferred_algorithm": "random",
                "supported_algorithms": ["random"],
                "state_transfer": "best_effort_mujoco_qpos_qvel",
                "notes": (
                    "This adapter exposes Gymnasium MuJoCo envs and simulator state hooks. "
                    "The current tabular Q-learning runner is not a valid learner for this task."
                ),
            },
        )

    def create_env(self, task: TaskDefinition) -> MujocoExtendedEnv:
        _ensure_mujoco_metadata(task)
        return MujocoExtendedEnv(task)

    def derive_task_from_episode(
        self,
        source_task: TaskDefinition,
        trace,
        moment_index: int,
    ) -> TaskDerivationOptions | None:
        _ = source_task, trace, moment_index
        return None


class MujocoEpisodeReplayWidget(EpisodeReplayWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._render_cache_key: tuple[object, ...] | None = None
        self._render_frames: list[QPixmap | None] = []
        self._render_error: str | None = None

        root = QVBoxLayout(self)
        self.summary_label = QLabel("No replay frame selected.", self)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.action_label = QLabel("", self)
        self.action_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.render_label = QLabel("MuJoCo frame preview unavailable.", self)
        self.render_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.render_label.setMinimumHeight(260)
        self.render_label.setStyleSheet(
            "QLabel { border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; }"
        )

        state_group = QGroupBox("Simulator State", self)
        state_layout = QFormLayout(state_group)
        self.qpos_label = QLabel("-", state_group)
        self.qvel_label = QLabel("-", state_group)
        self.ctrl_label = QLabel("-", state_group)
        self.time_label = QLabel("-", state_group)
        for label in (self.qpos_label, self.qvel_label, self.ctrl_label, self.time_label):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        state_layout.addRow("qpos", self.qpos_label)
        state_layout.addRow("qvel", self.qvel_label)
        state_layout.addRow("ctrl", self.ctrl_label)
        state_layout.addRow("time", self.time_label)

        self.observation_view = QTextEdit(self)
        self.observation_view.setReadOnly(True)
        self.observation_view.setMinimumHeight(72)

        root.addWidget(self.summary_label)
        root.addWidget(self.action_label)
        root.addWidget(self.render_label)
        root.addWidget(state_group)
        root.addWidget(QLabel("Observation / State Payload", self))
        root.addWidget(self.observation_view)

    def set_frame(self, trace: EpisodeTrace, step_index: int) -> None:
        transition_count = len(trace.steps)
        timeline_index = min(max(step_index, 0), transition_count)
        state = _state_from_trace(trace, timeline_index)
        self._update_render_preview(trace, timeline_index)

        if transition_count == 0:
            self.summary_label.setText(f"Episode {trace.episode_id} has no recorded transitions.")
            self.action_label.setText("Action: -")
            self._render_state(state, trace.initial_observation)
            return

        if timeline_index == 0:
            observation = trace.initial_observation
            if observation is None and trace.steps:
                observation = trace.steps[0].observation
            self.summary_label.setText(f"Step 0/{transition_count} | initial MuJoCo state")
            self.action_label.setText("Action: pending")
        else:
            step = trace.steps[timeline_index - 1]
            done_suffix = " | DONE" if step.terminated or step.truncated else ""
            self.summary_label.setText(
                f"Step {timeline_index}/{transition_count} | reward={step.reward:.3f}{done_suffix}"
            )
            self.action_label.setText(f"Action: {_short_sequence(step.action, max_items=8)}")
            observation = step.next_observation

        self._render_state(state, observation)

    def _render_state(self, state: MujocoEnvState | None, observation: object) -> None:
        if state is None:
            self.qpos_label.setText("Unavailable")
            self.qvel_label.setText("Unavailable")
            self.ctrl_label.setText("Unavailable")
            self.time_label.setText("-")
            self.observation_view.setPlainText(str(observation))
            return

        self.qpos_label.setText(_short_sequence(state.qpos))
        self.qvel_label.setText(_short_sequence(state.qvel))
        self.ctrl_label.setText(_short_sequence(state.ctrl))
        self.time_label.setText("-" if state.time is None else f"{state.time:.6g}")
        self.observation_view.setPlainText(
            "observation="
            f"{_short_sequence(observation, max_items=12)}\n"
            f"last_action={_short_sequence(state.last_action, max_items=8)}\n"
            f"terminated={state.terminated}"
        )

    def _update_render_preview(self, trace: EpisodeTrace, step_index: int) -> None:
        frames = self._get_or_build_render_frames(trace)
        if step_index < 0 or step_index >= len(frames):
            self.render_label.setPixmap(QPixmap())
            self.render_label.setText("Frame preview unavailable for this step.")
            return

        pixmap = frames[step_index]
        if pixmap is None or pixmap.isNull():
            self.render_label.setPixmap(QPixmap())
            if self._render_error:
                self.render_label.setText(f"MuJoCo frame preview unavailable: {self._render_error}")
            else:
                self.render_label.setText("MuJoCo frame preview unavailable.")
            return

        scaled = pixmap.scaled(
            520,
            340,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.render_label.setText("")
        self.render_label.setPixmap(scaled)

    def _get_or_build_render_frames(self, trace: EpisodeTrace) -> list[QPixmap | None]:
        replay_states = tuple(_state_from_trace(trace, index) for index in range(len(trace.steps) + 1))
        task_config = (
            tuple(sorted((str(key), repr(value)) for key, value in trace.task_snapshot.task_config.items()))
            if trace.task_snapshot is not None
            else ()
        )
        cache_key: tuple[object, ...] = (
            trace.run_id,
            trace.episode_id,
            tuple(_state_cache_key(state) for state in replay_states),
            task_config,
        )
        if cache_key == self._render_cache_key:
            return self._render_frames

        self._render_cache_key = cache_key
        self._render_frames = self._build_render_frames(trace, replay_states)
        return self._render_frames

    def _build_render_frames(
        self,
        trace: EpisodeTrace,
        replay_states: tuple[MujocoEnvState | None, ...],
    ) -> list[QPixmap | None]:
        self._render_error = None
        try:
            env = MujocoExtendedEnv.from_task_snapshot(trace.task_snapshot, render_mode="rgb_array")
        except Exception as exc:
            self._render_error = str(exc)
            return [None for _state in replay_states]
        if env is None:
            self._render_error = "missing task snapshot"
            return [None for _state in replay_states]

        frames: list[QPixmap | None] = []
        try:
            env.reset()
            for state in replay_states:
                if state is None:
                    frames.append(None)
                    continue
                env.import_state(state)
                frame = env.render()
                frames.append(self._to_pixmap(frame))
        except Exception as exc:
            self._render_error = str(exc)
            return [None for _state in replay_states]
        finally:
            try:
                env.close()
            except Exception:
                pass

        if len(frames) < len(replay_states):
            frames.extend([None] * (len(replay_states) - len(frames)))
        return frames

    def _to_pixmap(self, frame: object) -> QPixmap | None:
        if frame is None:
            return None
        frame_shape = getattr(frame, "shape", None)
        if frame_shape is None or not hasattr(frame, "tobytes"):
            return None

        shape = frame_shape
        if not isinstance(shape, tuple) or len(shape) != 3:
            return None

        height, width, channels = shape
        if channels not in (3, 4):
            return None

        frame_like = cast(_FrameArrayLike, frame)
        data = frame_like.tobytes()
        if channels == 3:
            image = QImage(data, width, height, channels * width, QImage.Format.Format_RGB888)
        else:
            image = QImage(data, width, height, channels * width, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(image.copy())


class MujocoTaskEditorWidget(QGroupBox):
    def __init__(
        self,
        task: TaskDefinition,
        on_task_changed: Callable[[TaskDefinition], None],
    ) -> None:
        super().__init__("MuJoCo Task")
        self._task = task
        self._on_task_changed = on_task_changed
        _ensure_mujoco_metadata(self._task)

        root = QVBoxLayout(self)

        notice = QLabel(
            "Minimal backend-only MuJoCo setup. Training uses random continuous actions for now; "
            "install gymnasium[mujoco] to create real MuJoCo envs.",
            self,
        )
        notice.setWordWrap(True)
        notice.setObjectName("SubtitleLabel")

        form_group = QGroupBox("Gymnasium Environment", self)
        form = QFormLayout(form_group)
        self.env_id_combo = QComboBox(form_group)
        self.env_id_combo.setEditable(True)
        self.env_id_combo.addItems(COMMON_MUJOCO_ENV_IDS)
        self.env_id_combo.setCurrentText(str(self._task.config.get("env_id", "HalfCheetah-v5")))
        self.env_id_combo.currentTextChanged.connect(self._on_env_id_changed)
        form.addRow("Env ID", self.env_id_combo)

        self.algorithm_label = QLabel("random continuous-action sampling", form_group)
        self.algorithm_label.setWordWrap(True)
        form.addRow("Current policy", self.algorithm_label)

        root.addWidget(notice)
        root.addWidget(form_group)

    def _on_env_id_changed(self, value: str) -> None:
        env_id = value.strip()
        if not env_id:
            return
        self._task.config = deepcopy(self._task.config)
        self._task.config["env_id"] = env_id
        _ensure_mujoco_metadata(self._task)
        self._on_task_changed(self._task)


class MujocoGuiExtension:
    def create_task_editor_widget(
        self,
        task: TaskDefinition,
        on_task_changed: Callable[[TaskDefinition], None],
    ) -> MujocoTaskEditorWidget:
        return MujocoTaskEditorWidget(task, on_task_changed)

    def create_episode_replay_widget(self, parent=None):
        return MujocoEpisodeReplayWidget(parent)


def build_mujoco_plugin() -> EnvironmentPlugin:
    return EnvironmentPlugin(
        plugin_id="mujoco",
        display_name="MuJoCo",
        description="Continuous-control Gymnasium MuJoCo environments with best-effort simulator state export/import.",
        backend=MujocoBackend(),
        gui_extension=MujocoGuiExtension(),
    )
