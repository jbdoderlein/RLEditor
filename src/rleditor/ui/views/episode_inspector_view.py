from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rleditor.core.models import EpisodeTrace
from rleditor.plugins.base import EnvironmentPlugin, EpisodeReplayWidget


class EpisodeInspectorView(QWidget):
    create_task_from_moment_requested = Signal(object, int)

    def __init__(self) -> None:
        super().__init__()
        self._episodes: list[EpisodeTrace] = []
        self._current_trace: EpisodeTrace | None = None
        self._replay_widget: EpisodeReplayWidget | None = None
        self._can_create_task_from_moment = False

        layout = QVBoxLayout(self)

        title = QLabel("Episode Inspector")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Explore captured trajectories with a timeline slider."
        )
        subtitle.setObjectName("SubtitleLabel")

        controls_group = QGroupBox("Replay Controls", self)
        controls_layout = QFormLayout(controls_group)

        self.episode_combo = QComboBox(controls_group)
        self.episode_combo.currentIndexChanged.connect(self._on_episode_selected)

        step_row = QWidget(controls_group)
        step_row_layout = QHBoxLayout(step_row)
        step_row_layout.setContentsMargins(0, 0, 0, 0)

        self.step_slider = QSlider(Qt.Orientation.Horizontal, step_row)
        self.step_slider.setMinimum(0)
        self.step_slider.setMaximum(0)
        self.step_slider.setEnabled(False)
        self.step_slider.valueChanged.connect(self._on_step_changed)

        self.step_label = QLabel("--", step_row)

        step_row_layout.addWidget(self.step_slider, 1)
        step_row_layout.addWidget(self.step_label)

        controls_layout.addRow("Episode", self.episode_combo)
        controls_layout.addRow("Step", step_row)

        self.capture_count_label = QLabel("Captured episodes: 0", controls_group)
        controls_layout.addRow("History", self.capture_count_label)

        self.create_task_btn = QPushButton("Create Task From Here", controls_group)
        self.create_task_btn.setEnabled(False)
        self.create_task_btn.clicked.connect(self._emit_create_task_from_current_moment)
        controls_layout.addRow("", self.create_task_btn)

        self.summary_label = QLabel("No episode captured yet.", self)

        self.plugin_group = QGroupBox("Environment View", self)
        self.plugin_layout = QVBoxLayout(self.plugin_group)
        self.plugin_placeholder = QLabel(
            "No plugin visualizer available for this environment.",
            self.plugin_group,
        )
        self.plugin_layout.addWidget(self.plugin_placeholder)

        self.viewer = QTextEdit(self)
        self.viewer.setReadOnly(True)
        self.viewer.setPlaceholderText("Transition details will appear here.")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(controls_group)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.plugin_group)
        layout.addWidget(self.viewer, 1)
        self._render_empty_state()

    def set_context(self, plugin: EnvironmentPlugin | None) -> None:
        self._can_create_task_from_moment = bool(
            plugin is not None and callable(getattr(plugin.backend, "derive_task_from_episode", None))
        )
        self._set_plugin_replay_widget(plugin)
        self._render_current_step()

    def clear_episodes(self) -> None:
        self._episodes.clear()
        self._current_trace = None
        self.episode_combo.blockSignals(True)
        self.episode_combo.clear()
        self.episode_combo.blockSignals(False)
        self.step_slider.blockSignals(True)
        self.step_slider.setEnabled(False)
        self.step_slider.setMinimum(0)
        self.step_slider.setMaximum(0)
        self.step_slider.setValue(0)
        self.step_slider.blockSignals(False)
        self.step_label.setText("--")
        self.capture_count_label.setText("Captured episodes: 0")
        self.create_task_btn.setEnabled(False)
        self._render_empty_state()

    def set_episode(self, trace: EpisodeTrace, *, focus: bool = True) -> None:
        self._episodes.append(trace)
        index = len(self._episodes) - 1
        if focus:
            self.episode_combo.addItem(self._build_episode_label(trace), index)
            self.episode_combo.setCurrentIndex(index)
        else:
            previous_index = self.episode_combo.currentIndex()
            previous_signal_state = self.episode_combo.blockSignals(True)
            self.episode_combo.addItem(self._build_episode_label(trace), index)
            self.episode_combo.setCurrentIndex(previous_index)
            self.episode_combo.blockSignals(previous_signal_state)
        self.capture_count_label.setText(f"Captured episodes: {len(self._episodes)}")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._current_trace is None and self._episodes:
            self.episode_combo.setCurrentIndex(len(self._episodes) - 1)

    def focus_episode(self, trace: EpisodeTrace) -> None:
        for index, existing_trace in enumerate(self._episodes):
            if self._same_trace(existing_trace, trace):
                self.episode_combo.setCurrentIndex(index)
                return
        self.set_episode(trace)

    def _set_plugin_replay_widget(self, plugin: EnvironmentPlugin | None) -> None:
        while self.plugin_layout.count() > 0:
            item = self.plugin_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._replay_widget = None
        self.plugin_placeholder = QLabel(
            "No plugin visualizer available for this environment.",
            self.plugin_group,
        )

        if plugin is None or plugin.gui_extension is None:
            self.plugin_layout.addWidget(self.plugin_placeholder)
            return

        replay_widget = plugin.gui_extension.create_episode_replay_widget(self.plugin_group)
        if replay_widget is None:
            self.plugin_layout.addWidget(self.plugin_placeholder)
            return

        self._replay_widget = replay_widget
        self.plugin_layout.addWidget(replay_widget)

    def _on_episode_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._episodes):
            self._current_trace = None
            self._render_empty_state()
            return

        self._current_trace = self._episodes[index]
        self._sync_slider_with_trace(self._current_trace)
        self._render_current_step()

    def _sync_slider_with_trace(self, trace: EpisodeTrace) -> None:
        transition_count = self._timeline_length(trace)
        self.step_slider.blockSignals(True)
        if transition_count == 0:
            self.step_slider.setEnabled(False)
            self.step_slider.setMinimum(0)
            self.step_slider.setMaximum(0)
            self.step_slider.setValue(0)
            self.step_label.setText("0 / 0")
        else:
            self.step_slider.setEnabled(True)
            self.step_slider.setMinimum(0)
            self.step_slider.setMaximum(transition_count)
            self.step_slider.setValue(0)
            self.step_label.setText(f"0 / {transition_count}")
        self.step_slider.blockSignals(False)

    def _on_step_changed(self, _value: int) -> None:
        self._render_current_step()

    def _render_empty_state(self) -> None:
        self.summary_label.setText("No episode captured yet.")
        self.viewer.setPlainText("Episode transition details will appear here.")
        self.create_task_btn.setEnabled(False)

    def _render_current_step(self) -> None:
        trace = self._current_trace
        if trace is None:
            self._render_empty_state()
            return

        if self._timeline_length(trace) == 0:
            self.summary_label.setText(
                f"Episode {trace.episode_id} | reward={trace.total_reward:.3f} | no steps"
            )
            self.viewer.setPlainText("Episode has no transitions.")
            self.create_task_btn.setEnabled(False)
            return

        timeline_index = self.step_slider.value()
        transition_count = self._timeline_length(trace)
        self.step_label.setText(f"{timeline_index} / {transition_count}")
        run_label = trace.run_id or "unknown"
        self.summary_label.setText(
            f"Run {run_label} | Episode {trace.episode_id} | reward={trace.total_reward:.3f} | success={trace.success}"
        )
        self.viewer.setPlainText(self._format_step_details(trace, timeline_index))
        self.create_task_btn.setEnabled(self._can_create_task_from_moment)

        if self._replay_widget is not None:
            self._replay_widget.set_frame(trace, timeline_index)

    def _build_episode_label(self, trace: EpisodeTrace) -> str:
        context_label = ""
        task_snapshot = trace.task_snapshot
        task_name = task_snapshot.task_name if task_snapshot is not None else ""
        if task_name:
            context_label = f"[{task_name}] "
        return (
            f"{context_label}Episode {trace.episode_id} "
            f"(steps={len(trace.steps)}, reward={trace.total_reward:.2f})"
        )

    def _format_step_details(self, trace: EpisodeTrace, timeline_index: int) -> str:
        lines = [
            f"Run ID: {trace.run_id or 'unknown'}",
            f"Episode {trace.episode_id}",
            f"Total reward: {trace.total_reward:.3f}",
            f"Success: {trace.success}",
        ]

        task_snapshot = trace.task_snapshot
        if task_snapshot is not None:
            environment_id = task_snapshot.environment_id
            task_name = task_snapshot.task_name
            lines.extend(
                [
                    f"Environment: {environment_id}",
                    f"Task: {task_name}",
                ]
            )

        if timeline_index == 0:
            moment = self._moment_at(trace, timeline_index)
            initial_observation = trace.initial_observation
            if initial_observation is None and trace.steps:
                initial_observation = trace.steps[0].observation
            lines.extend(
                [
                    "",
                    f"Step index: 0/{len(trace.steps)}",
                    "Phase: initial observation before the first action",
                    f"observation: {initial_observation}",
                ]
            )
            if moment is not None and moment.restorable_env_state is not None:
                lines.append(f"restorable_env_state: {moment.restorable_env_state}")
            return "\n".join(lines)

        step = trace.steps[timeline_index - 1]
        moment = self._moment_at(trace, timeline_index)
        lines.extend(
            [
                "",
                f"Step index: {timeline_index}/{len(trace.steps)}",
                f"t: {step.t}",
                f"observation: {step.observation}",
                f"action: {step.action}",
                f"next_observation: {step.next_observation}",
                f"reward: {step.reward:.3f}",
                f"terminated: {step.terminated}",
                f"truncated: {step.truncated}",
            ]
        )
        if moment is not None and moment.restorable_env_state is not None:
            lines.append(f"restorable_env_state: {moment.restorable_env_state}")
        return "\n".join(lines)

    def _emit_create_task_from_current_moment(self) -> None:
        trace = self._current_trace
        if trace is None:
            return
        self.create_task_from_moment_requested.emit(trace, self.step_slider.value())

    def _timeline_length(self, trace: EpisodeTrace) -> int:
        if trace.moments:
            return max(0, len(trace.moments) - 1)
        return len(trace.steps)

    def _moment_at(self, trace: EpisodeTrace, moment_index: int):
        if 0 <= moment_index < len(trace.moments):
            return trace.moments[moment_index]
        return None

    def _same_trace(self, left: EpisodeTrace, right: EpisodeTrace) -> bool:
        if left is right:
            return True
        return left.run_id == right.run_id and left.episode_id == right.episode_id
