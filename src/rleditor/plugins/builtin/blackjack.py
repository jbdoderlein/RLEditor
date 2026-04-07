from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from gymnasium.envs.toy_text.blackjack import sum_hand, usable_ace
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from rleditor.core.models import EpisodeTrace, TaskDefinition, TaskDerivationOptions
from rleditor.plugins.base import EnvironmentPlugin, EpisodeReplayWidget
from rleditor.plugins.builtin.blackjack_env import (
    BlackjackEnvState,
    BlackjackExtendedEnv,
    coerce_blackjack_hand,
    coerce_blackjack_observation,
)


ACTION_LABELS = {
    0: "STICK",
    1: "HIT",
}


class _FrameArrayLike(Protocol):
    shape: tuple[int, int, int]

    def tobytes(self) -> bytes: ...


def _cards_text(cards: tuple[int, ...]) -> str:
    return " ".join("A" if card == 1 else str(card) for card in cards)


def _parse_hand_text(value: str, *, fallback: tuple[int, ...]) -> tuple[int, ...]:
    normalized = value.replace(",", " ").strip()
    if not normalized:
        return fallback
    raw_cards = normalized.split()
    parsed = coerce_blackjack_hand(raw_cards)
    return parsed if parsed is not None else fallback


def _state_from_trace(trace: EpisodeTrace, moment_index: int) -> BlackjackEnvState | None:
    if 0 <= moment_index < len(trace.moments):
        moment = trace.moments[moment_index]
        restorable_env_state = moment.restorable_env_state
        if isinstance(restorable_env_state, BlackjackEnvState):
            return restorable_env_state
        if isinstance(restorable_env_state, dict):
            return BlackjackEnvState.from_dict(restorable_env_state)

    return None


class BlackjackEpisodeReplayWidget(EpisodeReplayWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._render_cache_key: tuple[object, ...] | None = None
        self._render_frames: list[QPixmap | None] = []

        root = QVBoxLayout(self)

        self.summary_label = QLabel("No replay frame selected.", self)
        self.action_label = QLabel("", self)
        self.action_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.render_label = QLabel("Gymnasium frame preview unavailable.", self)
        self.render_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.render_label.setMinimumHeight(180)
        self.render_label.setStyleSheet(
            "QLabel { border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; }"
        )

        state_group = QGroupBox("Table State", self)
        state_layout = QFormLayout(state_group)
        self.player_label = QLabel("-", state_group)
        self.dealer_label = QLabel("-", state_group)
        self.observation_label = QLabel("-", state_group)
        self.player_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.dealer_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.observation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        state_layout.addRow("Player", self.player_label)
        state_layout.addRow("Dealer", self.dealer_label)
        state_layout.addRow("Observation", self.observation_label)

        root.addWidget(self.summary_label)
        root.addWidget(self.action_label)
        root.addWidget(self.render_label)
        root.addWidget(state_group)

    def set_frame(self, trace: EpisodeTrace, step_index: int) -> None:
        if not trace.steps:
            self.summary_label.setText("Episode has no steps.")
            self.action_label.setText("")
            self.render_label.setPixmap(QPixmap())
            self.render_label.setText("Gymnasium frame preview unavailable.")
            self.player_label.setText("-")
            self.dealer_label.setText("-")
            self.observation_label.setText("-")
            return

        transition_count = len(trace.steps)
        timeline_index = min(max(step_index, 0), transition_count)
        state = _state_from_trace(trace, timeline_index)
        self._update_render_preview(trace, timeline_index)

        if timeline_index == 0:
            observation = trace.initial_observation
            if observation is None and trace.steps:
                observation = trace.steps[0].observation
            self.summary_label.setText(f"Step 0/{transition_count} | initial hand")
            self.action_label.setText("Decision: pending")
        else:
            step = trace.steps[timeline_index - 1]
            action_label = ACTION_LABELS.get(int(step.action), f"A{step.action}")
            done_suffix = " | DONE" if (step.terminated or step.truncated) else ""
            self.summary_label.setText(
                f"Step {timeline_index}/{transition_count} | reward={step.reward:.2f}{done_suffix}"
            )
            self.action_label.setText(
                f"Decision: observation={step.observation} -> action={action_label} ({step.action}) "
                f"-> next={step.next_observation}"
            )
            observation = step.next_observation

        self._render_state(state, observation)

    def _render_state(self, state: BlackjackEnvState | None, observation: object) -> None:
        observation_tuple = coerce_blackjack_observation(observation)
        if state is None:
            self.player_label.setText("Full hand unavailable")
            self.dealer_label.setText("Full hand unavailable")
            self.observation_label.setText(str(observation_tuple or observation))
            return

        player_total = sum_hand(list(state.player_hand))
        dealer_total = sum_hand(list(state.dealer_hand))
        player_usable = "yes" if usable_ace(list(state.player_hand)) else "no"

        self.player_label.setText(
            f"{_cards_text(state.player_hand)} | total={player_total} | usable ace={player_usable}"
        )
        self.dealer_label.setText(
            f"{_cards_text(state.dealer_hand)} | visible={state.dealer_hand[0]} | total={dealer_total}"
        )
        self.observation_label.setText(str(observation_tuple or observation))

    def _update_render_preview(self, trace: EpisodeTrace, step_index: int) -> None:
        frames = self._get_or_build_render_frames(trace)
        if step_index < 0 or step_index >= len(frames):
            self.render_label.setPixmap(QPixmap())
            self.render_label.setText("Frame preview unavailable for this step.")
            return

        pixmap = frames[step_index]
        if pixmap is None or pixmap.isNull():
            self.render_label.setPixmap(QPixmap())
            self.render_label.setText("Gymnasium frame preview unavailable.")
            return

        scaled = pixmap.scaled(
            420,
            300,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.render_label.setText("")
        self.render_label.setPixmap(scaled)

    def _get_or_build_render_frames(self, trace: EpisodeTrace) -> list[QPixmap | None]:
        replay_states = tuple(_state_from_trace(trace, index) for index in range(len(trace.steps) + 1))
        task_config = (
            tuple(sorted(trace.task_snapshot.task_config.items()))
            if trace.task_snapshot is not None
            else ()
        )
        cache_key: tuple[object, ...] = (
            trace.run_id,
            trace.episode_id,
            replay_states,
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
        replay_states: tuple[BlackjackEnvState | None, ...],
    ) -> list[QPixmap | None]:
        frames: list[QPixmap | None] = []
        env = BlackjackExtendedEnv.from_task_snapshot(trace.task_snapshot, render_mode="rgb_array")
        if env is None:
            return [None for _state in replay_states]

        try:
            env.reset()
            for state in replay_states:
                if state is None:
                    frames.append(None)
                    continue
                env.import_state(state)
                frame = env.render()
                frames.append(self._to_pixmap(frame))
        except Exception:
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

        frame_like = cast(_FrameArrayLike, frame)
        shape = frame_shape
        if not isinstance(shape, tuple) or len(shape) != 3:
            return None

        height, width, channels = shape
        if channels not in (3, 4):
            return None

        data = frame_like.tobytes()
        if channels == 3:
            image = QImage(data, width, height, channels * width, QImage.Format.Format_RGB888)
        else:
            image = QImage(data, width, height, channels * width, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(image.copy())


class BlackjackBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="blackjack",
            name="Blackjack Default",
            config={
                "natural": False,
                "sab": True,
            },
        )

    def create_env(self, task: TaskDefinition):
        return BlackjackExtendedEnv(task)

    def derive_task_from_episode(
        self,
        source_task: TaskDefinition,
        trace: EpisodeTrace,
        moment_index: int,
    ) -> TaskDerivationOptions | None:
        state = _state_from_trace(trace, moment_index)
        if state is None:
            return None

        return TaskDerivationOptions(
            config_updates={"initial_state": state.to_dict()},
            derivation_reason="start_from_episode_moment",
            source_episode_id=trace.episode_id,
            source_moment_index=moment_index,
            source_run_id=trace.run_id,
            start_state=state.to_dict(),
        )


class BlackjackTaskEditorWidget(QGroupBox):
    def __init__(
        self,
        task: TaskDefinition,
        on_task_changed: Callable[[TaskDefinition], None],
    ) -> None:
        super().__init__("Blackjack Task")
        self._task = task
        self._on_task_changed = on_task_changed
        self._ensure_task_defaults()

        root = QVBoxLayout(self)

        rules_group = QGroupBox("Rules", self)
        rules_layout = QFormLayout(rules_group)
        self.natural_checkbox = QCheckBox("Natural blackjack pays 1.5 when Sutton-Barto rules are disabled", rules_group)
        self.sab_checkbox = QCheckBox("Use Sutton-Barto rules", rules_group)
        self.natural_checkbox.setChecked(bool(self._task.config.get("natural", False)))
        self.sab_checkbox.setChecked(bool(self._task.config.get("sab", True)))
        rules_layout.addRow("", self.sab_checkbox)
        rules_layout.addRow("", self.natural_checkbox)

        start_group = QGroupBox("Initial State Override", self)
        start_layout = QFormLayout(start_group)
        self.start_override_checkbox = QCheckBox("Start every episode from fixed hands", start_group)
        self.player_hand_input = QLineEdit(start_group)
        self.dealer_hand_input = QLineEdit(start_group)
        self.player_hand_input.setPlaceholderText("Example: 10 1")
        self.dealer_hand_input.setPlaceholderText("Example: 10 7")

        initial_state = self._initial_state()
        has_initial_state = initial_state is not None
        self.start_override_checkbox.setChecked(has_initial_state)
        self.player_hand_input.setText(_cards_text(initial_state.player_hand if initial_state else (10, 1)))
        self.dealer_hand_input.setText(_cards_text(initial_state.dealer_hand if initial_state else (10, 7)))
        self.player_hand_input.setEnabled(has_initial_state)
        self.dealer_hand_input.setEnabled(has_initial_state)

        start_layout.addRow("", self.start_override_checkbox)
        start_layout.addRow("Player cards", self.player_hand_input)
        start_layout.addRow("Dealer cards", self.dealer_hand_input)

        root.addWidget(rules_group)
        root.addWidget(start_group)

        self.natural_checkbox.stateChanged.connect(lambda _state: self._emit_task_change())
        self.sab_checkbox.stateChanged.connect(lambda _state: self._emit_task_change())
        self.start_override_checkbox.stateChanged.connect(lambda _state: self._on_start_override_changed())
        self.player_hand_input.editingFinished.connect(self._emit_task_change)
        self.dealer_hand_input.editingFinished.connect(self._emit_task_change)
        self._emit_task_change()

    def _ensure_task_defaults(self) -> None:
        self._task.config.setdefault("natural", False)
        self._task.config.setdefault("sab", True)
        initial_state = self._initial_state()
        if initial_state is not None:
            self._task.config["initial_state"] = initial_state.to_dict()

    def _initial_state(self) -> BlackjackEnvState | None:
        raw_state = self._task.config.get("initial_state")
        if isinstance(raw_state, BlackjackEnvState):
            return raw_state
        if isinstance(raw_state, dict):
            return BlackjackEnvState.from_dict(raw_state)
        return None

    def _on_start_override_changed(self) -> None:
        enabled = self.start_override_checkbox.isChecked()
        self.player_hand_input.setEnabled(enabled)
        self.dealer_hand_input.setEnabled(enabled)
        self._emit_task_change()

    def _emit_task_change(self) -> None:
        self._task.config["natural"] = self.natural_checkbox.isChecked()
        self._task.config["sab"] = self.sab_checkbox.isChecked()
        if self.start_override_checkbox.isChecked():
            player_hand = _parse_hand_text(self.player_hand_input.text(), fallback=(10, 1))
            dealer_hand = _parse_hand_text(self.dealer_hand_input.text(), fallback=(10, 7))
            state = BlackjackEnvState(player_hand=player_hand, dealer_hand=dealer_hand)
            self._task.config["initial_state"] = state.to_dict()
            self.player_hand_input.setText(_cards_text(player_hand))
            self.dealer_hand_input.setText(_cards_text(dealer_hand))
        else:
            self._task.config.pop("initial_state", None)

        self._on_task_changed(self._task)


class BlackjackGuiExtension:
    def create_task_editor_widget(
        self,
        task: TaskDefinition,
        on_task_changed: Callable[[TaskDefinition], None],
    ) -> QWidget:
        return BlackjackTaskEditorWidget(task, on_task_changed)

    def create_episode_replay_widget(self, parent: QWidget | None = None) -> EpisodeReplayWidget | None:
        return BlackjackEpisodeReplayWidget(parent)


def build_blackjack_plugin() -> EnvironmentPlugin:
    return EnvironmentPlugin(
        plugin_id="blackjack",
        display_name="Blackjack",
        description="Toy-text card game with discrete hit/stick actions.",
        backend=BlackjackBackend(),
        gui_extension=BlackjackGuiExtension(),
    )
