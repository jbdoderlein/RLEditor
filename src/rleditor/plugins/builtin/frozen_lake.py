from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rleditor.core.models import EpisodeTrace, TaskDefinition, TaskDerivationOptions
from rleditor.plugins.base import EnvironmentPlugin, EpisodeReplayWidget
from rleditor.plugins.builtin.frozen_lake_env import (
    DEFAULT_SUCCESS_RATE,
    FrozenLakeEnvState,
    FrozenLakeExtendedEnv,
    TILE_FROZEN,
    TILE_GOAL,
    TILE_HOLE,
    TILE_START,
    VALID_TILES,
    _coerce_reward_config,
    _default_reward_config,
    _generate_random_map_desc,
    _map_from_task_config,
    _normalize_map_desc,
    _parse_success_rate,
    _parse_size,
    _to_rows,
    coerce_frozen_lake_state_index,
)


class _FrameArrayLike(Protocol):
    shape: tuple[int, int, int]

    def tobytes(self) -> bytes: ...


def _state_index_from_trace(trace: EpisodeTrace, moment_index: int) -> int | None:
    if 0 <= moment_index < len(trace.moments):
        moment = trace.moments[moment_index]
        state_index = coerce_frozen_lake_state_index(moment.observation)
        if state_index is not None:
            return state_index

        restorable_env_state = moment.restorable_env_state
        if isinstance(restorable_env_state, FrozenLakeEnvState):
            return restorable_env_state.state_index
        if isinstance(restorable_env_state, dict):
            return coerce_frozen_lake_state_index(restorable_env_state.get("state_index"))

    if moment_index == 0:
        initial_observation = trace.initial_observation
        if initial_observation is None and trace.steps:
            initial_observation = trace.steps[0].observation
        return coerce_frozen_lake_state_index(initial_observation)

    if 0 < moment_index <= len(trace.steps):
        return coerce_frozen_lake_state_index(trace.steps[moment_index - 1].next_observation)

    return None


class FrozenLakeEpisodeReplayWidget(EpisodeReplayWidget):
    ACTION_LABELS = {
        0: "LEFT",
        1: "DOWN",
        2: "RIGHT",
        3: "UP",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cells: dict[int, QLabel] = {}
        self._active_map_shape: tuple[int, int] | None = None
        self._render_cache_key: tuple[object, ...] | None = None
        self._render_frames: list[QPixmap | None] = []
        self._current_render_pixmap: QPixmap | None = None

        root = QVBoxLayout(self)
        self.summary_label = QLabel("No replay frame selected.", self)
        self.action_label = QLabel("", self)
        self.action_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.render_label = QLabel("Gymnasium frame preview unavailable.", self)
        self.render_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.render_label.setMinimumHeight(160)
        self.render_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.render_label.setStyleSheet(
            "QLabel { border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; }"
        )

        self.grid_host = QWidget(self)
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(4)
        self.grid_layout.setVerticalSpacing(4)
        self.grid_layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.grid_host.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.grid_scroll = QScrollArea(self)
        self.grid_scroll.setWidgetResizable(False)
        self.grid_scroll.setMinimumHeight(150)
        self.grid_scroll.setWidget(self.grid_host)

        root.addWidget(self.summary_label)
        root.addWidget(self.action_label)
        root.addWidget(self.render_label, 1)
        root.addWidget(self.grid_scroll, 1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_scaled_render_pixmap()

    def set_frame(
        self,
        trace: EpisodeTrace,
        step_index: int,
    ) -> None:
        if not trace.steps:
            self.summary_label.setText("Episode has no steps.")
            self.action_label.setText("")
            self._current_render_pixmap = None
            self.render_label.setPixmap(QPixmap())
            self.render_label.setText("Gymnasium frame preview unavailable.")
            return

        transition_count = len(trace.steps)
        timeline_index = min(max(step_index, 0), transition_count)
        task_config = self._resolve_task_config(trace)
        map_desc = _map_from_task_config(task_config)
        self._ensure_map_grid(map_desc)
        self._reset_cell_styles(map_desc)
        self._update_render_preview(trace, timeline_index, task_config)

        current_index: int | None
        if timeline_index == 0:
            initial_observation = trace.initial_observation
            if initial_observation is None:
                initial_observation = trace.steps[0].observation
            current_index = self._as_state_index(initial_observation)
            self.summary_label.setText(f"Step 0/{transition_count} | initial state")
            self.action_label.setText(
                f"Initial position: observation={initial_observation}"
            )
        else:
            step = trace.steps[timeline_index - 1]
            current_index = self._as_state_index(step.next_observation)
            action_name = self.ACTION_LABELS.get(int(step.action), f"A{step.action}")
            done_suffix = " | DONE" if (step.terminated or step.truncated) else ""
            self.summary_label.setText(
                f"Step {timeline_index}/{transition_count} | reward={step.reward:.2f}{done_suffix}"
            )
            self.action_label.setText(
                "Decision: "
                f"observation={step.observation} -> action={action_name} ({step.action}) "
                f"-> next={step.next_observation}"
            )

        if current_index is not None and current_index in self._cells:
            self._cells[current_index].setStyleSheet(
                self._cells[current_index].styleSheet()
                + "QLabel { border: 2px solid #1d4ed8; }"
            )

    def _resolve_task_config(self, trace: EpisodeTrace) -> dict[str, object]:
        if trace.task_snapshot is None:
            return {}
        return {str(key): value for key, value in trace.task_snapshot.task_config.items()}

    def _update_render_preview(
        self,
        trace: EpisodeTrace,
        step_index: int,
        task_config: dict[str, object],
    ) -> None:
        frames = self._get_or_build_render_frames(trace, task_config)
        if step_index < 0 or step_index >= len(frames):
            self._current_render_pixmap = None
            self.render_label.setPixmap(QPixmap())
            self.render_label.setText("Frame preview unavailable for this step.")
            return

        pixmap = frames[step_index]
        if pixmap is None or pixmap.isNull():
            self._current_render_pixmap = None
            self.render_label.setPixmap(QPixmap())
            self.render_label.setText("Gymnasium frame preview unavailable.")
            return

        self._set_render_pixmap(pixmap)

    def _set_render_pixmap(self, pixmap: QPixmap) -> None:
        self._current_render_pixmap = pixmap
        self._refresh_scaled_render_pixmap()

    def _refresh_scaled_render_pixmap(self) -> None:
        pixmap = self._current_render_pixmap
        if pixmap is None or pixmap.isNull():
            return

        target_width = max(1, self.render_label.width() - 12)
        target_height = max(1, self.render_label.height() - 12)
        scaled = pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.render_label.setText("")
        self.render_label.setPixmap(scaled)

    def _get_or_build_render_frames(
        self,
        trace: EpisodeTrace,
        task_config: dict[str, object],
    ) -> list[QPixmap | None]:
        initial_observation = trace.initial_observation
        if initial_observation is None and trace.steps:
            initial_observation = trace.steps[0].observation

        replay_states = (
            self._as_env_state(initial_observation, None),
            *(self._as_env_state(step.next_observation, step.action) for step in trace.steps),
        )
        map_desc = tuple(_map_from_task_config(task_config))

        cache_key: tuple[object, ...] = (
            trace.run_id,
            trace.episode_id,
            replay_states,
            map_desc,
            bool(task_config.get("is_slippery", True)),
            _parse_success_rate(task_config.get("success_rate")),
        )

        if cache_key == self._render_cache_key:
            return self._render_frames

        self._render_cache_key = cache_key
        self._render_frames = self._build_render_frames(trace, replay_states, map_desc, task_config)
        return self._render_frames

    def _build_render_frames(
        self,
        trace: EpisodeTrace,
        replay_states: tuple[FrozenLakeEnvState | None, ...],
        _map_desc: tuple[str, ...],
        _task_config: dict[str, object],
    ) -> list[QPixmap | None]:
        frames: list[QPixmap | None] = []
        env = FrozenLakeExtendedEnv.from_task_snapshot(trace.task_snapshot, render_mode="rgb_array")
        if env is None:
            return [None for _ in replay_states]

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
            return [None for _ in replay_states]
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

        # At this point frame provides ndarray-like shape/tobytes attributes.
        data = frame_like.tobytes()
        if channels == 3:
            image = QImage(data, width, height, channels * width, QImage.Format.Format_RGB888)
        else:
            image = QImage(data, width, height, channels * width, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(image.copy())

    def _ensure_map_grid(self, map_desc: list[str]) -> None:
        rows = len(map_desc)
        cols = len(map_desc[0]) if rows > 0 else 0
        shape = (rows, cols)
        if shape == self._active_map_shape:
            return

        self._active_map_shape = shape
        self._cells.clear()

        while self.grid_layout.count() > 0:
            item = self.grid_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for row in range(rows):
            for col in range(cols):
                state_index = row * cols + col
                tile = QLabel(f"{map_desc[row][col]}\n{state_index}", self.grid_host)
                tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
                tile.setFixedSize(42, 42)
                self.grid_layout.addWidget(tile, row, col)
                self._cells[state_index] = tile

    def _reset_cell_styles(self, map_desc: list[str]) -> None:
        cols = len(map_desc[0]) if map_desc else 0
        for state_index, label in self._cells.items():
            row = state_index // cols
            col = state_index % cols
            tile = map_desc[row][col]
            label.setStyleSheet(self._style_for_tile(tile))

    def _style_for_tile(self, tile: str) -> str:
        if tile == TILE_START:
            return "QLabel { background: #dbeafe; border: 1px solid #93c5fd; border-radius: 5px; }"
        if tile == TILE_GOAL:
            return "QLabel { background: #dcfce7; border: 1px solid #86efac; border-radius: 5px; }"
        if tile == TILE_HOLE:
            return "QLabel { background: #fee2e2; border: 1px solid #fca5a5; border-radius: 5px; }"
        return "QLabel { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 5px; }"

    def _as_state_index(self, state: object) -> int | None:
        return coerce_frozen_lake_state_index(state)

    def _as_env_state(self, state: object, last_action: int | None) -> FrozenLakeEnvState | None:
        state_index = self._as_state_index(state)
        if state_index is None:
            return None
        return FrozenLakeEnvState(state_index=state_index, last_action=last_action)


class FrozenLakeBackend:
    def default_task(self) -> TaskDefinition:
        size = 4
        hole_probability = 0.22
        return TaskDefinition(
            environment_id="frozen_lake",
            name="Frozen Lake Default",
            config={
                "size": size,
                "is_slippery": True,
                "success_rate": DEFAULT_SUCCESS_RATE,
                "hole_probability": hole_probability,
                "map_desc": _generate_random_map_desc(size, hole_probability),
            },
            reward_config=_default_reward_config(),
        )

    def create_env(self, task: TaskDefinition):
        return FrozenLakeExtendedEnv(task)

    def derive_task_from_episode(
        self,
        source_task: TaskDefinition,
        trace: EpisodeTrace,
        moment_index: int,
    ) -> TaskDerivationOptions | None:
        start_state = _state_index_from_trace(trace, moment_index)
        if start_state is None:
            return None

        return TaskDerivationOptions(
            config_updates={"start_state": start_state},
            derivation_reason="start_from_episode_moment",
            source_episode_id=trace.episode_id,
            source_moment_index=moment_index,
            source_run_id=trace.run_id,
            start_state=start_state,
        )


class FrozenLakeTaskEditorWidget(QGroupBox):
    TILE_LABELS = {
        TILE_FROZEN: "Frozen",
        TILE_HOLE: "Hole",
        TILE_START: "Start",
        TILE_GOAL: "Goal",
    }

    TILE_STYLES = {
        TILE_START: "QPushButton { background: #dbeafe; border: 1px solid #60a5fa; font-weight: 700; }",
        TILE_GOAL: "QPushButton { background: #dcfce7; border: 1px solid #4ade80; font-weight: 700; }",
        TILE_HOLE: "QPushButton { background: #fecaca; border: 1px solid #f87171; font-weight: 700; }",
        TILE_FROZEN: "QPushButton { background: #f8fafc; border: 1px solid #cbd5e1; font-weight: 700; }",
    }

    def __init__(
        self,
        task: TaskDefinition,
        on_task_changed: Callable[[TaskDefinition], None],
    ) -> None:
        super().__init__("Frozen Lake Task")
        self._task = task
        self._on_task_changed = on_task_changed
        self._buttons: dict[tuple[int, int], QPushButton] = {}
        self._map: list[list[str]] = []

        self._ensure_task_defaults()

        root = QVBoxLayout(self)

        controls = QGroupBox("Map Setup", self)
        controls_layout = QFormLayout(controls)

        self.size_spin = QSpinBox(controls)
        self.size_spin.setRange(2, 64)
        self.size_spin.setAccelerated(True)
        self.size_spin.setValue(len(self._map))
        self.size_spin.setToolTip("Custom square grid size. Large grids can become hard to edit visually.")

        self.slippery_checkbox = QCheckBox("Enable slippery dynamics", controls)
        self.slippery_checkbox.setChecked(bool(self._task.config.get("is_slippery", True)))

        self.success_rate_spin = QDoubleSpinBox(controls)
        self.success_rate_spin.setRange(0.0, 1.0)
        self.success_rate_spin.setSingleStep(0.05)
        self.success_rate_spin.setDecimals(6)
        self.success_rate_spin.setValue(_parse_success_rate(self._task.config.get("success_rate")))
        self.success_rate_spin.setToolTip(
            "Probability that the requested action is applied when slippery dynamics are enabled."
        )

        self.hole_probability = QDoubleSpinBox(controls)
        self.hole_probability.setRange(0, 0.95)
        self.hole_probability.setSingleStep(0.05)
        self.hole_probability.setDecimals(2)
        self.hole_probability.setValue(float(self._task.config.get("hole_probability", 0.22)))

        self.regenerate_button = QPushButton("Regenerate Random Map", controls)
        self.regenerate_button.clicked.connect(self._regenerate_random_map)

        controls_layout.addRow("Grid size", self.size_spin)
        controls_layout.addRow("Dynamics", self.slippery_checkbox)
        controls_layout.addRow("Success rate", self.success_rate_spin)
        controls_layout.addRow("Hole probability", self.hole_probability)
        controls_layout.addRow("", self.regenerate_button)

        start_group = QGroupBox("Start State Override", self)
        start_layout = QFormLayout(start_group)
        self.start_override_checkbox = QCheckBox(
            "Override the initial player state without moving the map Start tile",
            start_group,
        )
        self.start_state_spin = QSpinBox(start_group)
        self.start_state_spin.setRange(0, max(0, len(self._map) * len(self._map) - 1))
        self.start_state_hint = QLabel(
            "Useful for derived tasks created from episode moments.",
            start_group,
        )
        self.start_state_hint.setWordWrap(True)
        start_layout.addRow("", self.start_override_checkbox)
        start_layout.addRow("State index", self.start_state_spin)
        start_layout.addRow("", self.start_state_hint)

        paint_group = QGroupBox("Grid Editor", self)
        paint_layout = QFormLayout(paint_group)
        self.paint_combo = QComboBox(paint_group)
        self.paint_combo.addItem("Frozen (F)", TILE_FROZEN)
        self.paint_combo.addItem("Hole (H)", TILE_HOLE)
        self.paint_combo.addItem("Start (S)", TILE_START)
        self.paint_combo.addItem("Goal (G)", TILE_GOAL)
        self.reset_button = QPushButton("Reset To Empty Map", paint_group)
        self.reset_button.clicked.connect(self._reset_to_empty)
        self.editor_hint = QLabel(
            "Click cells to paint. Start and Goal stay unique.",
            paint_group,
        )

        paint_layout.addRow("Paint mode", self.paint_combo)
        paint_layout.addRow("", self.reset_button)
        paint_layout.addRow("", self.editor_hint)

        reward_group = QGroupBox("Reward Overrides", self)
        reward_layout = QFormLayout(reward_group)
        reward_values = _coerce_reward_config(self._task.reward_config)

        self.reward_frozen = QDoubleSpinBox(reward_group)
        self.reward_hole = QDoubleSpinBox(reward_group)
        self.reward_start = QDoubleSpinBox(reward_group)
        self.reward_goal = QDoubleSpinBox(reward_group)

        for spin in (self.reward_frozen, self.reward_hole, self.reward_start, self.reward_goal):
            spin.setRange(-10.0, 10.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)

        self.reward_frozen.setValue(reward_values["tile:F"])
        self.reward_hole.setValue(reward_values["tile:H"])
        self.reward_start.setValue(reward_values["tile:S"])
        self.reward_goal.setValue(reward_values["tile:G"])

        reward_layout.addRow("Frozen (F)", self.reward_frozen)
        reward_layout.addRow("Hole (H)", self.reward_hole)
        reward_layout.addRow("Start (S)", self.reward_start)
        reward_layout.addRow("Goal (G)", self.reward_goal)

        self.grid_host = QWidget(self)
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(4)
        self.grid_layout.setVerticalSpacing(4)
        self.grid_layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.grid_host.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        root.addWidget(controls)
        root.addWidget(start_group)
        root.addWidget(paint_group)
        root.addWidget(reward_group)
        root.addWidget(self.grid_host, 0, Qt.AlignmentFlag.AlignLeft)

        self.size_spin.valueChanged.connect(lambda _value: self._on_size_changed())
        self.slippery_checkbox.stateChanged.connect(lambda _state: self._on_slippery_changed())
        self.success_rate_spin.valueChanged.connect(lambda _value: self._emit_task_change())
        self.hole_probability.valueChanged.connect(lambda _value: self._emit_task_change())
        self.start_override_checkbox.stateChanged.connect(lambda _state: self._on_start_override_changed())
        self.start_state_spin.valueChanged.connect(lambda _value: self._emit_task_change())
        self.reward_frozen.valueChanged.connect(lambda _value: self._emit_task_change())
        self.reward_hole.valueChanged.connect(lambda _value: self._emit_task_change())
        self.reward_start.valueChanged.connect(lambda _value: self._emit_task_change())
        self.reward_goal.valueChanged.connect(lambda _value: self._emit_task_change())

        self._sync_success_rate_controls()
        self._sync_start_override_controls()
        self._rebuild_grid()
        self._emit_task_change()

    def _ensure_task_defaults(self) -> None:
        size = _parse_size(self._task.config.get("size", 4), fallback=4)
        self._task.config["size"] = size

        if "hole_probability" not in self._task.config:
            self._task.config["hole_probability"] = 0.22
        if "is_slippery" not in self._task.config:
            self._task.config["is_slippery"] = True
        self._task.config["success_rate"] = _parse_success_rate(
            self._task.config.get("success_rate")
        )

        map_rows = _map_from_task_config(self._task.config, fallback_size=size)
        self._map = _normalize_map_desc(map_rows, expected_size=size)
        self._task.config["map_desc"] = _to_rows(self._map)
        self._task.reward_config = _coerce_reward_config(self._task.reward_config)
        raw_start_state = self._task.config.get("start_state")
        if raw_start_state is not None:
            start_state = coerce_frozen_lake_state_index(raw_start_state)
            if start_state is None:
                self._task.config.pop("start_state", None)
            else:
                self._task.config["start_state"] = start_state

    def _on_size_changed(self) -> None:
        size = self._selected_grid_size()
        hole_probability = float(self.hole_probability.value())
        self._map = _normalize_map_desc(
            _generate_random_map_desc(size, hole_probability),
            expected_size=size,
        )
        self._clear_start_override()
        self._rebuild_grid()
        self._emit_task_change()

    def _regenerate_random_map(self) -> None:
        size = self._selected_grid_size()
        hole_probability = float(self.hole_probability.value())
        self._map = _normalize_map_desc(
            _generate_random_map_desc(size, hole_probability),
            expected_size=size,
        )
        self._clear_start_override()
        self._rebuild_grid()
        self._emit_task_change()

    def _reset_to_empty(self) -> None:
        size = self._selected_grid_size()
        self._map = [[TILE_FROZEN for _ in range(size)] for _ in range(size)]
        self._map[0][0] = TILE_START
        self._map[-1][-1] = TILE_GOAL
        self._clear_start_override()
        self._rebuild_grid()
        self._emit_task_change()

    def _selected_grid_size(self) -> int:
        return int(self.size_spin.value())

    def _rebuild_grid(self) -> None:
        self._sync_start_override_controls()
        self._buttons.clear()
        while self.grid_layout.count() > 0:
            item = self.grid_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for row_index, row in enumerate(self._map):
            for col_index, tile in enumerate(row):
                button = QPushButton(tile, self.grid_host)
                button.setFixedSize(34, 34)
                button.clicked.connect(
                    lambda _checked=False, r=row_index, c=col_index: self._paint_cell(r, c)
                )
                self.grid_layout.addWidget(button, row_index, col_index)
                self._buttons[(row_index, col_index)] = button
                self._apply_button_style(button, tile)

    def _paint_cell(self, row: int, col: int) -> None:
        target = str(self.paint_combo.currentData())
        if target not in VALID_TILES:
            return

        current = self._map[row][col]
        if current == target:
            return

        if current == TILE_START and target != TILE_START and self._count_tiles(TILE_START) == 1:
            return
        if current == TILE_GOAL and target != TILE_GOAL and self._count_tiles(TILE_GOAL) == 1:
            return

        if target == TILE_START:
            self._replace_unique_tile(TILE_START, row, col)
            self._clear_start_override()
        elif target == TILE_GOAL:
            self._replace_unique_tile(TILE_GOAL, row, col)
        else:
            self._map[row][col] = target

        self._refresh_buttons()
        self._emit_task_change()

    def _replace_unique_tile(self, tile: str, row: int, col: int) -> None:
        for r_index, row_values in enumerate(self._map):
            for c_index, value in enumerate(row_values):
                if value == tile:
                    self._map[r_index][c_index] = TILE_FROZEN
        self._map[row][col] = tile

    def _count_tiles(self, tile: str) -> int:
        return sum(1 for row in self._map for value in row if value == tile)

    def _refresh_buttons(self) -> None:
        for (row, col), button in self._buttons.items():
            tile = self._map[row][col]
            button.setText(tile)
            self._apply_button_style(button, tile)

    def _apply_button_style(self, button: QPushButton, tile: str) -> None:
        button.setStyleSheet(self.TILE_STYLES.get(tile, self.TILE_STYLES[TILE_FROZEN]))
        button.setToolTip(f"{self.TILE_LABELS.get(tile, 'Frozen')} ({tile})")

    def _sync_start_override_controls(self) -> None:
        max_state = max(0, len(self._map) * len(self._map) - 1)
        self.start_state_spin.setRange(0, max_state)

        raw_start_state = self._task.config.get("start_state")
        start_state = coerce_frozen_lake_state_index(raw_start_state)
        has_override = start_state is not None

        self.start_override_checkbox.blockSignals(True)
        self.start_override_checkbox.setChecked(has_override)
        self.start_override_checkbox.blockSignals(False)

        self.start_state_spin.blockSignals(True)
        self.start_state_spin.setEnabled(has_override)
        self.start_state_spin.setValue(min(max(0, start_state or 0), max_state))
        self.start_state_spin.blockSignals(False)

    def _clear_start_override(self) -> None:
        self._task.config.pop("start_state", None)
        self.start_override_checkbox.blockSignals(True)
        self.start_override_checkbox.setChecked(False)
        self.start_override_checkbox.blockSignals(False)
        self.start_state_spin.blockSignals(True)
        self.start_state_spin.setEnabled(False)
        self.start_state_spin.setValue(0)
        self.start_state_spin.blockSignals(False)

    def _sync_success_rate_controls(self) -> None:
        self.success_rate_spin.setEnabled(self.slippery_checkbox.isChecked())

    def _on_slippery_changed(self) -> None:
        self._sync_success_rate_controls()
        self._emit_task_change()

    def _on_start_override_changed(self) -> None:
        self.start_state_spin.setEnabled(self.start_override_checkbox.isChecked())
        self._emit_task_change()

    def _emit_task_change(self) -> None:
        self._task.config["size"] = len(self._map)
        self._task.config["is_slippery"] = self.slippery_checkbox.isChecked()
        self._task.config["success_rate"] = float(self.success_rate_spin.value())
        self._task.config["hole_probability"] = float(self.hole_probability.value())
        self._task.config["map_desc"] = _to_rows(self._map)
        if self.start_override_checkbox.isChecked():
            self._task.config["start_state"] = int(self.start_state_spin.value())
        else:
            self._task.config.pop("start_state", None)

        self._task.reward_config = {
            "tile:F": float(self.reward_frozen.value()),
            "tile:H": float(self.reward_hole.value()),
            "tile:S": float(self.reward_start.value()),
            "tile:G": float(self.reward_goal.value()),
        }

        self._on_task_changed(self._task)


class FrozenLakeGuiExtension:
    def create_task_editor_widget(
        self,
        task: TaskDefinition,
        on_task_changed: Callable[[TaskDefinition], None],
    ) -> QWidget:
        return FrozenLakeTaskEditorWidget(task, on_task_changed)

    def create_episode_replay_widget(self, parent: QWidget | None = None) -> EpisodeReplayWidget | None:
        return FrozenLakeEpisodeReplayWidget(parent)


def build_frozen_lake_plugin() -> EnvironmentPlugin:
    return EnvironmentPlugin(
        plugin_id="frozen_lake",
        display_name="Frozen Lake",
        description="Discrete navigation task with holes and sparse rewards.",
        backend=FrozenLakeBackend(),
        gui_extension=FrozenLakeGuiExtension(),
    )
