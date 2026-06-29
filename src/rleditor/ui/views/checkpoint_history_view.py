from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from math import hypot
from pathlib import Path
import re

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rleditor.application.services import TrainingHistorySnapshot
from rleditor.core.models import Checkpoint, EpisodeTrace, RunConfig, TaskSnapshot, TrainingRun
from rleditor.infra.stable_baselines_backend import (
    is_stable_baselines3_algorithm,
    load_stable_baselines3_model,
)


FROZEN_LAKE_ACTION_SYMBOLS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
}
DEFAULT_MAX_STEPS_PER_EPISODE = 100


@dataclass(slots=True)
class _LineageNode:
    node_id: str
    kind: str
    label: str
    sublabel: str
    checkpoint: Checkpoint | None
    rect: QRectF
    center: QPointF
    environment_id: str | None = None
    algorithm: str | None = None
    success_rate: float | None = None


@dataclass(slots=True)
class _LineageEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    target_checkpoint: Checkpoint
    source_checkpoint: Checkpoint | None
    run: TrainingRun | None
    task_snapshot: TaskSnapshot | None
    episodes: list[EpisodeTrace]
    source_point: QPointF
    target_point: QPointF


class _QTableDialog(QDialog):
    def __init__(self, checkpoint: Checkpoint, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Q table - {checkpoint.label or checkpoint.checkpoint_id}")
        self.resize(900, 620)

        self._checkpoint = checkpoint
        self._q_values = _q_values_from_checkpoint(checkpoint)
        self._map_rows = _frozen_lake_map_rows(checkpoint)
        self._action_count = _q_table_action_count(self._q_values, self._map_rows)
        self.policy_cells: dict[tuple[int, int], QLabel] = {}

        root = QVBoxLayout(self)
        root.addWidget(QLabel(self._summary_text(), self))

        if self._map_rows is not None:
            root.addWidget(QLabel("FrozenLake policy map: best action and max Q-value per state", self))
            root.addWidget(self._build_policy_map(), 1)
        else:
            root.addWidget(
                QLabel("Policy map display is available for FrozenLake Q-learning checkpoints.", self),
                1,
            )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _summary_text(self) -> str:
        return (
            f"Checkpoint: {self._checkpoint.checkpoint_id} | "
            f"stored Q values: {len(self._q_values)}"
        )

    def _build_policy_map(self) -> QWidget:
        assert self._map_rows is not None

        host = QWidget(self)
        grid = QGridLayout(host)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        max_values = [
            self._max_q_for_state(str(index))
            for index in range(len(self._map_rows) * len(self._map_rows[0]))
        ]
        finite_values = [value for value in max_values if value is not None]
        min_value = min(finite_values) if finite_values else 0.0
        max_value = max(finite_values) if finite_values else 1.0

        for row, map_row in enumerate(self._map_rows):
            for col, tile in enumerate(map_row):
                state_index = row * len(map_row) + col
                state_key = str(state_index)
                value = self._max_q_for_state(state_key)
                action = self._best_action_for_state(state_key)
                action_symbol = FROZEN_LAKE_ACTION_SYMBOLS.get(action, "-") if action is not None else "-"
                value_text = "--" if value is None else f"{value:.3f}"
                label = QLabel(f"{tile}\n{action_symbol}\n{value_text}", host)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(62, 52)
                label.setStyleSheet(
                    "QLabel { "
                    f"background: {_q_value_color(value, min_value=min_value, max_value=max_value)}; "
                    "border: 1px solid #cbd5e1; border-radius: 4px; "
                    "font-weight: 600; color: #0f172a; "
                    "}"
                )
                self.policy_cells[(row, col)] = label
                grid.addWidget(label, row, col)

        return host

    def _max_q_for_state(self, state_key: str) -> float | None:
        if self._action_count <= 0:
            return None
        values = [
            self._q_values.get((state_key, action), 0.0)
            for action in range(self._action_count)
        ]
        if not values:
            return None
        if all((state_key, action) not in self._q_values for action in range(self._action_count)):
            return None
        return max(values)

    def _best_action_for_state(self, state_key: str) -> int | None:
        if self._action_count <= 0:
            return None
        values = [
            (action, self._q_values.get((state_key, action), 0.0))
            for action in range(self._action_count)
        ]
        if not values:
            return None
        if all((state_key, action) not in self._q_values for action in range(self._action_count)):
            return None
        best_value = max(value for _action, value in values)
        best_actions = [action for action, value in values if value == best_value]
        return min(best_actions)


class _SB3PolicyDialog(QDialog):
    def __init__(self, checkpoint: Checkpoint, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Policy - {checkpoint.label or checkpoint.checkpoint_id}")
        self.resize(900, 620)

        self._checkpoint = checkpoint
        self._map_rows = _frozen_lake_map_rows(checkpoint)
        self._actions: dict[int, int] = {}
        self._load_error: str | None = None
        self.policy_cells: dict[tuple[int, int], QLabel] = {}

        root = QVBoxLayout(self)
        root.addWidget(QLabel(self._summary_text(), self))

        if self._map_rows is None:
            root.addWidget(
                QLabel("Policy map display is available for FrozenLake SB3 checkpoints.", self),
                1,
            )
        else:
            self._actions = self._predict_frozen_lake_actions()
            if self._load_error is not None:
                error_label = QLabel(f"Could not load SB3 policy: {self._load_error}", self)
                error_label.setWordWrap(True)
                root.addWidget(error_label, 1)
            else:
                root.addWidget(QLabel("FrozenLake greedy policy map: deterministic model action per state", self))
                root.addWidget(self._build_policy_map(), 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _summary_text(self) -> str:
        algorithm = self._checkpoint.metadata.get("algorithm")
        return f"Checkpoint: {self._checkpoint.checkpoint_id} | algorithm: {algorithm or 'unknown'}"

    def _predict_frozen_lake_actions(self) -> dict[int, int]:
        assert self._map_rows is not None
        learner_state = self._checkpoint.metadata.get("learner_state")
        if not isinstance(learner_state, dict):
            self._load_error = "missing learner state"
            return {}

        algorithm = str(self._checkpoint.metadata.get("algorithm") or learner_state.get("algorithm") or "")
        try:
            model = load_stable_baselines3_model(
                algorithm=algorithm,
                env=None,
                learner_state=learner_state,
            )
        except Exception as exc:
            self._load_error = str(exc)
            return {}

        state_count = len(self._map_rows) * len(self._map_rows[0])
        actions: dict[int, int] = {}
        for state_index in range(state_count):
            action = self._predict_action(model, state_index)
            if action is not None:
                actions[state_index] = action
        return actions

    def _predict_action(self, model: object, state_index: int) -> int | None:
        predict = getattr(model, "predict", None)
        if not callable(predict):
            self._load_error = "loaded learner does not expose predict()"
            return None

        observation_candidates = (
            state_index,
            np.asarray(state_index, dtype=np.int64),
            np.asarray([state_index], dtype=np.int64),
        )
        last_error: Exception | None = None
        for observation in observation_candidates:
            try:
                action, _state = predict(observation, deterministic=True)
                return _coerce_policy_action(action)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            self._load_error = str(last_error)
        return None

    def _build_policy_map(self) -> QWidget:
        assert self._map_rows is not None

        host = QWidget(self)
        grid = QGridLayout(host)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        for row, map_row in enumerate(self._map_rows):
            for col, tile in enumerate(map_row):
                state_index = row * len(map_row) + col
                action = self._actions.get(state_index)
                action_symbol = FROZEN_LAKE_ACTION_SYMBOLS.get(action, "-") if action is not None else "-"
                label = QLabel(f"{tile}\n{action_symbol}", host)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(62, 52)
                label.setStyleSheet(
                    "QLabel { "
                    "background: #eff6ff; "
                    "border: 1px solid #cbd5e1; border-radius: 4px; "
                    "font-weight: 600; color: #0f172a; "
                    "}"
                )
                self.policy_cells[(row, col)] = label
                grid.addWidget(label, row, col)

        return host


class _StateVisitHeatmapDialog(QDialog):
    def __init__(
        self,
        *,
        title: str,
        map_rows: list[str],
        visit_counts: dict[int, int],
        episode_count: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 620)
        self.visit_cells: dict[tuple[int, int], QLabel] = {}

        root = QVBoxLayout(self)
        total_visits = sum(visit_counts.values())
        root.addWidget(
            QLabel(
                f"Episodes: {episode_count} | recorded state visits: {total_visits}",
                self,
            )
        )
        root.addWidget(self._build_heatmap(map_rows, visit_counts), 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_heatmap(self, map_rows: list[str], visit_counts: dict[int, int]) -> QWidget:
        host = QWidget(self)
        grid = QGridLayout(host)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        max_count = max(visit_counts.values(), default=0)
        for row, map_row in enumerate(map_rows):
            for col, tile in enumerate(map_row):
                state_index = row * len(map_row) + col
                count = visit_counts.get(state_index, 0)
                label = QLabel(f"{tile}\n{state_index}\n{count}", host)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(62, 52)
                label.setStyleSheet(
                    "QLabel { "
                    f"background: {_visit_count_color(count, max_count=max_count)}; "
                    "border: 1px solid #cbd5e1; border-radius: 4px; "
                    "font-weight: 600; color: #0f172a; "
                    "}"
                )
                self.visit_cells[(row, col)] = label
                grid.addWidget(label, row, col)

        return host


class _CheckpointNodeActionDialog(QDialog):
    def __init__(
        self,
        checkpoint: Checkpoint,
        *,
        can_show_q_table: bool,
        can_show_policy: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Checkpoint actions - {checkpoint.label or checkpoint.checkpoint_id}")
        self._selected_action: str | None = None

        root = QVBoxLayout(self)
        label = QLabel(checkpoint.label or checkpoint.checkpoint_id, self)
        label.setWordWrap(True)
        root.addWidget(label)

        actions_layout = QHBoxLayout()
        self.rename_button = QPushButton("Rename", self)
        self.evaluate_button = QPushButton("Run Evaluation", self)
        self.show_q_table_button = QPushButton("Show Q Table", self)
        self.show_policy_button = QPushButton("Show Policy", self)

        self.show_q_table_button.setVisible(can_show_q_table)
        self.show_q_table_button.setEnabled(can_show_q_table)
        self.show_policy_button.setVisible(can_show_policy)
        self.show_policy_button.setEnabled(can_show_policy)

        for button, action in (
            (self.rename_button, "rename"),
            (self.evaluate_button, "evaluate"),
            (self.show_q_table_button, "q_table"),
            (self.show_policy_button, "policy"),
        ):
            button.clicked.connect(lambda _checked=False, action=action: self._accept_action(action))
            actions_layout.addWidget(button)
        actions_layout.addStretch(1)
        root.addLayout(actions_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    def _accept_action(self, action: str) -> None:
        self._selected_action = action
        self.accept()


class _NormalizeCheckpointDialog(QDialog):
    def __init__(
        self,
        *,
        checkpoint: Checkpoint,
        current_episodes: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Normalize checkpoint")
        self.resize(420, 180)
        self._current_episodes = max(0, int(current_episodes))

        root = QVBoxLayout(self)
        description = QLabel(
            f"Checkpoint: {checkpoint.label or checkpoint.checkpoint_id}",
            self,
        )
        description.setWordWrap(True)
        root.addWidget(description)

        form = QFormLayout()
        self.target_episodes_spin = QSpinBox(self)
        self.target_episodes_spin.setRange(self._current_episodes, 2_147_483_647)
        self.target_episodes_spin.setValue(self._current_episodes)
        self.target_episodes_spin.valueChanged.connect(self._update_accept_enabled)
        form.addRow("Target total episodes", self.target_episodes_spin)
        root.addLayout(form)

        self.delta_label = QLabel("", self)
        root.addWidget(self.delta_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        self.accept_button = buttons.addButton("Normalize", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._update_accept_enabled()

    def additional_episodes(self) -> int:
        return max(0, self.target_episodes_spin.value() - self._current_episodes)

    def _update_accept_enabled(self) -> None:
        additional_episodes = self.additional_episodes()
        self.delta_label.setText(f"Additional no-learning episodes: {additional_episodes}")
        self.accept_button.setEnabled(additional_episodes > 0)


class _TrainingEdgeEditDialog(QDialog):
    def __init__(self, config: RunConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Live edit training")
        self.resize(420, 320)
        self._original_config = RunConfig.from_dict(config.to_dict())

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.algorithm_combo = QComboBox(self)
        self.algorithm_combo.addItem("Q-learning", "q_learning")
        self.algorithm_combo.addItem("Stable-Baselines3 DQN", "sb3_dqn")
        self.algorithm_combo.addItem("Stable-Baselines3 PPO", "sb3_ppo")
        algorithm_index = self.algorithm_combo.findData(self._original_config.algorithm)
        if algorithm_index < 0:
            self.algorithm_combo.addItem(self._original_config.algorithm, self._original_config.algorithm)
            algorithm_index = self.algorithm_combo.findData(self._original_config.algorithm)
        self.algorithm_combo.setCurrentIndex(max(0, algorithm_index))
        self.algorithm_combo.setEnabled(False)

        self.episode_count_spin = QSpinBox(self)
        self.episode_count_spin.setRange(0, 1_000_000)
        self.episode_count_spin.setSpecialValueText("No limit")
        self.episode_count_spin.setValue(
            0 if self._original_config.max_episodes is None else self._original_config.max_episodes
        )

        self.max_steps_per_episode_spin = QSpinBox(self)
        self.max_steps_per_episode_spin.setRange(0, 10_000_000)
        self.max_steps_per_episode_spin.setSpecialValueText("No limit")
        self.max_steps_per_episode_spin.setValue(
            self._default_max_steps_per_episode_value(self._original_config)
        )

        self.max_steps_spin = QSpinBox(self)
        self.max_steps_spin.setRange(0, 50_000_000)
        self.max_steps_spin.setSpecialValueText("No limit")
        self.max_steps_spin.setValue(
            0 if self._original_config.max_steps is None else self._original_config.max_steps
        )

        self.learning_rate_spin = QDoubleSpinBox(self)
        self.learning_rate_spin.setRange(0.0, 1.0)
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setSingleStep(0.01)
        self.learning_rate_spin.setValue(self._original_config.learning_rate)

        self.discount_factor_spin = QDoubleSpinBox(self)
        self.discount_factor_spin.setRange(0.0, 1.0)
        self.discount_factor_spin.setDecimals(4)
        self.discount_factor_spin.setSingleStep(0.01)
        self.discount_factor_spin.setValue(self._original_config.gamma)

        self.epsilon_spin = QDoubleSpinBox(self)
        self.epsilon_spin.setRange(0.0, 1.0)
        self.epsilon_spin.setDecimals(4)
        self.epsilon_spin.setSingleStep(0.01)
        self.epsilon_spin.setValue(self._original_config.epsilon)

        self.trace_sample_rate_spin = QDoubleSpinBox(self)
        self.trace_sample_rate_spin.setRange(0.0, 100.0)
        self.trace_sample_rate_spin.setDecimals(1)
        self.trace_sample_rate_spin.setSingleStep(5.0)
        self.trace_sample_rate_spin.setSuffix(" %")
        self.trace_sample_rate_spin.setValue(self._original_config.episode_trace_sample_rate * 100.0)

        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setSpecialValueText("Random")
        self.seed_spin.setValue(-1 if self._original_config.seed is None else self._original_config.seed)

        form.addRow("Algorithm", self.algorithm_combo)
        form.addRow("Episodes", self.episode_count_spin)
        form.addRow("Max steps / episode", self.max_steps_per_episode_spin)
        form.addRow("Max steps", self.max_steps_spin)
        form.addRow("Learning rate", self.learning_rate_spin)
        form.addRow("Discount factor", self.discount_factor_spin)
        form.addRow("Epsilon", self.epsilon_spin)
        form.addRow("Recorded episodes", self.trace_sample_rate_spin)
        form.addRow("Seed", self.seed_spin)
        root.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        buttons.addButton("Edit", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def edited_config(self) -> RunConfig:
        config = RunConfig.from_dict(self._original_config.to_dict())
        config.max_episodes = self.episode_count_spin.value() if self.episode_count_spin.value() > 0 else None
        config.max_steps_per_episode = (
            self.max_steps_per_episode_spin.value()
            if self.max_steps_per_episode_spin.value() > 0
            else None
        )
        config.max_steps = self.max_steps_spin.value() if self.max_steps_spin.value() > 0 else None
        config.learning_rate = self.learning_rate_spin.value()
        config.gamma = self.discount_factor_spin.value()
        config.epsilon = self.epsilon_spin.value()
        config.episode_trace_sample_rate = self.trace_sample_rate_spin.value() / 100.0
        config.seed = self.seed_spin.value() if self.seed_spin.value() >= 0 else None

        config.hyperparameters = dict(config.hyperparameters)
        config.hyperparameters["learning_rate"] = config.learning_rate
        config.hyperparameters["gamma"] = config.gamma
        config.hyperparameters["epsilon"] = config.epsilon
        return config

    def _default_max_steps_per_episode_value(self, config: RunConfig) -> int:
        if config.max_steps_per_episode is not None:
            return int(config.max_steps_per_episode)
        return DEFAULT_MAX_STEPS_PER_EPISODE


class _TrainingReportChart(QWidget):
    def __init__(
        self,
        *,
        points: list[tuple[int, float]] | None = None,
        task_switches: list[tuple[int, str]] | None = None,
        total_episodes: int = 0,
        series: list[dict[str, object]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if series is None:
            series = [
                {
                    "label": "Training path",
                    "points": list(points or []),
                    "task_switches": list(task_switches or []),
                    "total_episodes": total_episodes,
                }
            ]
        colors = [
            QColor("#7c3aed"),
            QColor("#0f766e"),
            QColor("#b45309"),
            QColor("#2563eb"),
            QColor("#be123c"),
            QColor("#4b5563"),
        ]
        self.series: list[dict[str, object]] = []
        for index, item in enumerate(series):
            raw_points = item.get("points", []) if isinstance(item, dict) else []
            normalized_points: list[tuple[int, float]] = []
            if isinstance(raw_points, list):
                for point in raw_points:
                    if not isinstance(point, (tuple, list)) or len(point) != 2:
                        continue
                    try:
                        normalized_points.append((int(point[0]), float(point[1])))
                    except (TypeError, ValueError):
                        continue

            raw_switches = item.get("task_switches", []) if isinstance(item, dict) else []
            normalized_switches: list[tuple[int, str]] = []
            if isinstance(raw_switches, list):
                for task_switch in raw_switches:
                    if not isinstance(task_switch, (tuple, list)) or len(task_switch) != 2:
                        continue
                    try:
                        normalized_switches.append((int(task_switch[0]), str(task_switch[1])))
                    except (TypeError, ValueError):
                        continue

            try:
                item_total_episodes = (
                    int(item.get("total_episodes", 0))
                    if isinstance(item, dict)
                    else 0
                )
            except (TypeError, ValueError):
                item_total_episodes = 0

            self.series.append(
                {
                    "label": (
                        str(item.get("label", f"Path {index + 1}"))
                        if isinstance(item, dict)
                        else f"Path {index + 1}"
                    ),
                    "points": normalized_points,
                    "task_switches": normalized_switches,
                    "total_episodes": max(0, item_total_episodes),
                    "color": colors[index % len(colors)],
                }
            )

        self.points = list(self.series[0]["points"]) if self.series else []
        self.task_switches = list(self.series[0]["task_switches"]) if self.series else []
        self.total_episodes = max(
            [int(item["total_episodes"]) for item in self.series] or [max(0, int(total_episodes))]
        )
        self.setMinimumSize(620, 300)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(12, 12, -12, -12)
        metrics = painter.fontMetrics()
        plot_top_margin = 30 if len(self.series) > 1 else 12
        plot_rect = rect.adjusted(52, plot_top_margin, -16, -38)
        painter.fillRect(rect, QColor("#ffffff"))

        painter.setPen(QPen(QColor("#94a3b8"), 1.0))
        painter.drawLine(plot_rect.left(), plot_rect.top(), plot_rect.left(), plot_rect.bottom())
        painter.drawLine(plot_rect.left(), plot_rect.bottom(), plot_rect.right(), plot_rect.bottom())

        all_points = [
            point
            for item in self.series
            for point in item["points"]
            if isinstance(point, tuple) and len(point) == 2
        ]
        if not all_points:
            painter.setPen(QColor("#64748b"))
            painter.drawText(
                plot_rect,
                Qt.AlignmentFlag.AlignCenter,
                "No recorded reward traces are available for this path.",
            )
            return

        if len(self.series) > 1:
            legend_x = plot_rect.left()
            legend_y = rect.top() + metrics.ascent() + 2
            for item in self.series:
                label = str(item["label"])
                if len(label) > 34:
                    label = label[:31] + "..."
                color = item["color"]
                painter.setPen(QPen(color, 2.4))
                painter.drawLine(int(legend_x), int(legend_y - 4), int(legend_x + 16), int(legend_y - 4))
                painter.setPen(QColor("#334155"))
                painter.drawText(int(legend_x + 20), int(legend_y), label)
                legend_x += 32 + metrics.horizontalAdvance(label)
                if legend_x > plot_rect.right() - 160:
                    break

        max_x = max(self.total_episodes, max(episode for episode, _reward in all_points), 1)
        rewards = [reward for _episode, reward in all_points]
        min_y = min(0.0, min(rewards))
        max_y = max(0.0, max(rewards))
        y_span = max(max_y - min_y, 1e-9)

        def map_x(episode: int) -> float:
            return plot_rect.left() + (max(0, min(episode, max_x)) / max_x) * plot_rect.width()

        def map_y(reward: float) -> float:
            return plot_rect.bottom() - ((reward - min_y) / y_span) * plot_rect.height()

        switch_pen = QPen(QColor("#64748b"), 1.0, Qt.PenStyle.DashLine)
        painter.setPen(switch_pen)
        drawn_switches: set[tuple[int, str]] = set()
        task_switches = [
            task_switch
            for item in self.series
            for task_switch in item["task_switches"]
            if isinstance(task_switch, tuple) and len(task_switch) == 2
        ]
        for episode, task_name in task_switches:
            if (episode, task_name) in drawn_switches:
                continue
            drawn_switches.add((episode, task_name))
            if episode <= 0 or episode >= max_x:
                continue
            x = map_x(episode)
            painter.drawLine(int(x), plot_rect.top(), int(x), plot_rect.bottom())
            if len(self.series) == 1:
                label = task_name if len(task_name) <= 22 else task_name[:19] + "..."
                painter.drawText(
                    int(x) + 4,
                    plot_rect.top() + metrics.ascent() + 2,
                    label,
                )

        for item in self.series:
            item_points = list(item["points"])
            if not item_points:
                continue
            path = QPainterPath()
            first_episode, first_reward = item_points[0]
            path.moveTo(map_x(first_episode), map_y(first_reward))
            for episode, reward in item_points[1:]:
                path.lineTo(map_x(episode), map_y(reward))

            painter.setPen(QPen(item["color"], 2.2))
            painter.drawPath(path)

        painter.setPen(QColor("#334155"))
        painter.drawText(int(rect.left()), int(plot_rect.top() + metrics.ascent()), f"{max_y:.2f}")
        painter.drawText(int(rect.left()), int(plot_rect.bottom()), f"{min_y:.2f}")
        painter.drawText(int(plot_rect.left()), int(rect.bottom()), "0")
        max_x_label = str(max_x)
        painter.drawText(
            int(plot_rect.right() - metrics.horizontalAdvance(max_x_label)),
            int(rect.bottom()),
            max_x_label,
        )
        painter.drawText(
            int(plot_rect.center().x() - metrics.horizontalAdvance("Training episode") / 2),
            int(rect.bottom()),
            "Training episode",
        )


class _TrainingReportDialog(QDialog):
    def __init__(self, report: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training report")
        self.resize(760, 460)
        self.report = report
        self.reports = self._normalized_reports(report)

        root = QVBoxLayout(self)
        if len(self.reports) == 1:
            single_report = self.reports[0]
            total_episodes = int(single_report.get("total_episodes", 0))
            recorded_episodes = int(single_report.get("recorded_episode_count", 0))
            target_label = str(single_report.get("target_label", "Selected checkpoint"))
            root.addWidget(QLabel(f"Path: {target_label}", self))
            root.addWidget(QLabel(f"Total training episodes: {total_episodes}", self))
            root.addWidget(QLabel(f"Recorded reward episodes: {recorded_episodes}", self))
        else:
            root.addWidget(QLabel(f"Selected checkpoints: {len(self.reports)}", self))
            summary = QTextEdit(self)
            summary.setReadOnly(True)
            summary.setMaximumHeight(150)
            summary.setHtml(self._summary_html())
            root.addWidget(summary)

        root.addWidget(
            _TrainingReportChart(
                series=self._chart_series(),
                parent=self,
            ),
            1,
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.export_json_button = buttons.addButton(
            "Export JSON",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.export_json_button.clicked.connect(self._export_json)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _normalized_reports(self, report: dict[str, object]) -> list[dict[str, object]]:
        reports = report.get("reports")
        if isinstance(reports, list):
            normalized = [item for item in reports if isinstance(item, dict)]
            if normalized:
                return normalized
        return [report]

    def _summary_html(self) -> str:
        rows = [
            "<tr><th>Checkpoint</th><th>Total training episodes</th><th>Recorded reward episodes</th></tr>"
        ]
        for report in self.reports:
            label = escape(str(report.get("target_label", "Selected checkpoint")))
            total_episodes = escape(str(int(report.get("total_episodes", 0))))
            recorded_episodes = escape(str(int(report.get("recorded_episode_count", 0))))
            rows.append(
                "<tr>"
                f"<td>{label}</td>"
                f"<td>{total_episodes}</td>"
                f"<td>{recorded_episodes}</td>"
                "</tr>"
            )
        return (
            "<table cellspacing='0' cellpadding='6' width='100%'>"
            "<thead>"
            f"{rows[0]}"
            "</thead>"
            "<tbody>"
            f"{''.join(rows[1:])}"
            "</tbody>"
            "</table>"
        )

    def _chart_series(self) -> list[dict[str, object]]:
        series: list[dict[str, object]] = []
        for index, report in enumerate(self.reports, start=1):
            label = str(report.get("target_label") or report.get("target_checkpoint_id") or f"Checkpoint {index}")
            series.append(
                {
                    "label": label,
                    "points": list(report.get("cumulative_reward_points", [])),
                    "task_switches": list(report.get("task_switches", [])),
                    "total_episodes": int(report.get("total_episodes", 0)),
                }
            )
        return series

    def _export_json(self) -> None:
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Training Report",
            self._default_export_filename(),
            "JSON Files (*.json);;All Files (*)",
        )
        if not selected_path:
            return

        path = Path(selected_path).expanduser()
        if path.suffix == "":
            path = path.with_suffix(".json")

        try:
            path.write_text(
                json.dumps(self.report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Export Training Report",
                f"Could not write training report:\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Export Training Report",
            f"Training report exported to:\n{path}",
        )

    def _default_export_filename(self) -> str:
        if len(self.reports) > 1:
            return "training_report_selected_checkpoints.json"
        report = self.reports[0] if self.reports else {}
        checkpoint_id = str(report.get("target_checkpoint_id") or "selected_checkpoint")
        safe_checkpoint_id = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in checkpoint_id
        )
        return f"training_report_{safe_checkpoint_id}.json"


def _q_values_from_checkpoint(checkpoint: Checkpoint) -> dict[tuple[str, int], float]:
    learner_state = checkpoint.metadata.get("learner_state")
    if not isinstance(learner_state, dict):
        return {}

    q_values_payload = learner_state.get("q_values", [])
    if not isinstance(q_values_payload, list):
        return {}

    q_values: dict[tuple[str, int], float] = {}
    for entry in q_values_payload:
        if not isinstance(entry, dict):
            continue
        try:
            state_key = str(entry.get("state_key", ""))
            action = int(entry.get("action", 0))
            value = float(entry.get("value", 0.0))
        except (TypeError, ValueError):
            continue
        q_values[(state_key, action)] = value
    return q_values


def _checkpoint_has_q_learning_state(checkpoint: Checkpoint | None) -> bool:
    if checkpoint is None:
        return False
    metadata = checkpoint.metadata
    if metadata.get("algorithm") == "q_learning":
        return True
    learner_state = metadata.get("learner_state")
    return isinstance(learner_state, dict) and learner_state.get("algorithm") == "q_learning"


def _checkpoint_has_sb3_policy_state(checkpoint: Checkpoint | None) -> bool:
    if checkpoint is None:
        return False
    metadata = checkpoint.metadata
    algorithm = str(metadata.get("algorithm", ""))
    if not is_stable_baselines3_algorithm(algorithm):
        return False
    learner_state = metadata.get("learner_state")
    return (
        isinstance(learner_state, dict)
        and learner_state.get("backend") == "stable_baselines3"
        and isinstance(learner_state.get("model_zip_base64"), str)
    )


def _coerce_policy_action(action: object) -> int | None:
    if hasattr(action, "item") and callable(getattr(action, "item")):
        try:
            return int(action.item())
        except (TypeError, ValueError):
            return None
    if hasattr(action, "tolist") and callable(getattr(action, "tolist")):
        try:
            action = action.tolist()
        except Exception:
            return None
    if isinstance(action, list | tuple):
        if not action:
            return None
        return _coerce_policy_action(action[0])
    try:
        return int(action)
    except (TypeError, ValueError):
        return None


def _frozen_lake_map_rows(checkpoint: Checkpoint) -> list[str] | None:
    task_snapshot = checkpoint.task_snapshot
    return _frozen_lake_map_rows_from_task_snapshot(task_snapshot)


def _frozen_lake_map_rows_from_task_snapshot(task_snapshot: TaskSnapshot | None) -> list[str] | None:
    if task_snapshot is None or task_snapshot.environment_id != "frozen_lake":
        return None
    task_config = task_snapshot.task_config
    raw_map = task_config.get("map_desc")
    if isinstance(raw_map, list) and raw_map and all(isinstance(row, str) for row in raw_map):
        rows = [str(row) for row in raw_map]
        width = len(rows[0])
        if width > 0 and all(len(row) == width for row in rows):
            return rows

    size = _parse_map_size(task_config.get("size"), fallback=4)
    rows = ["F" * size for _ in range(size)]
    rows[0] = "S" + rows[0][1:]
    rows[-1] = rows[-1][:-1] + "G"
    return rows


def _parse_map_size(value: object, *, fallback: int) -> int:
    if isinstance(value, int):
        return max(2, value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text.endswith("x"):
            text = text[:-1]
        if "x" in text:
            text = text.split("x", 1)[0]
        if text.isdigit():
            return max(2, int(text))
    return fallback


def _q_table_action_count(
    q_values: dict[tuple[str, int], float],
    map_rows: list[str] | None,
) -> int:
    if map_rows is not None:
        q_action_count = max((action for _state, action in q_values), default=-1) + 1
        return max(4, q_action_count)
    if q_values:
        return max(action for _state, action in q_values) + 1
    return 0


def _q_value_color(value: float | None, *, min_value: float, max_value: float) -> str:
    if value is None:
        return "#f8fafc"
    span = max(max_value - min_value, 1e-9)
    ratio = max(0.0, min(1.0, (value - min_value) / span))
    red = int(239 - 100 * ratio)
    green = int(246 - 80 * ratio)
    blue = 255
    return QColor(red, green, blue).name()


def _visit_count_color(count: int, *, max_count: int) -> str:
    if count <= 0 or max_count <= 0:
        return "#f8fafc"
    ratio = max(0.0, min(1.0, count / max_count))
    red = int(239 - 116 * ratio)
    green = int(246 - 103 * ratio)
    blue = int(255 - 63 * ratio)
    return QColor(red, green, blue).name()


def _coerce_state_index(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _coerce_state_index(value.item())
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        return int(text) if text.isdigit() else None
    return None


def _state_index_from_restorable_state(value: object) -> int | None:
    if isinstance(value, dict):
        return _coerce_state_index(value.get("state_index"))
    state_index = getattr(value, "state_index", None)
    return _coerce_state_index(state_index)


class CheckpointGraphWidget(QWidget):
    node_selected = Signal(object)
    edge_selected = Signal(object)
    node_double_clicked = Signal(object)
    edge_double_clicked = Signal(object)
    delete_key_pressed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._snapshot = TrainingHistorySnapshot([], [], {}, {})
        self._nodes: dict[str, _LineageNode] = {}
        self._edges: list[_LineageEdge] = []
        self._selected_node_id: str | None = None
        self._selected_node_ids: list[str] = []
        self._selected_edge_id: str | None = None
        self._content_size = QSize(900, 520)
        self.setMinimumSize(220, 140)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def selected_node_id(self) -> str | None:
        return self._selected_node_id

    @property
    def selected_node_ids(self) -> tuple[str, ...]:
        return tuple(self._selected_node_ids)

    @property
    def selected_edge_id(self) -> str | None:
        return self._selected_edge_id

    def node_for_id(self, node_id: str) -> _LineageNode | None:
        return self._nodes.get(node_id)

    def selected_nodes(self) -> list[_LineageNode]:
        return [
            self._nodes[node_id]
            for node_id in self._selected_node_ids
            if node_id in self._nodes
        ]

    def edge_for_id(self, edge_id: str) -> _LineageEdge | None:
        for edge in self._edges:
            if edge.edge_id == edge_id:
                return edge
        return None

    def select_node(self, node_id: str, *, additive: bool = False, toggle: bool = False) -> None:
        if node_id not in self._nodes:
            self._selected_node_id = None
            self._selected_node_ids = []
            return

        if additive:
            selected_node_ids = list(self._selected_node_ids)
            if node_id in selected_node_ids:
                if toggle:
                    selected_node_ids.remove(node_id)
                else:
                    selected_node_ids.remove(node_id)
                    selected_node_ids.append(node_id)
            else:
                selected_node_ids.append(node_id)
            self._selected_node_ids = selected_node_ids
            self._selected_node_id = selected_node_ids[-1] if selected_node_ids else None
        else:
            self._selected_node_ids = [node_id]
            self._selected_node_id = node_id

        self._selected_edge_id = None
        self.update()

    def select_nodes(self, node_ids: list[str] | tuple[str, ...]) -> None:
        selected_node_ids: list[str] = []
        for node_id in node_ids:
            if node_id in self._nodes and node_id not in selected_node_ids:
                selected_node_ids.append(node_id)
        self._selected_node_ids = selected_node_ids
        self._selected_node_id = selected_node_ids[-1] if selected_node_ids else None
        self._selected_edge_id = None
        self.update()

    def select_edge(self, edge_id: str) -> None:
        if self.edge_for_id(edge_id) is None:
            self._selected_edge_id = None
            return
        self._selected_edge_id = edge_id
        self._selected_node_id = None
        self._selected_node_ids = []
        self.update()

    def clear_selection(self) -> None:
        self._selected_node_id = None
        self._selected_node_ids = []
        self._selected_edge_id = None
        self.update()

    def set_history(self, snapshot: TrainingHistorySnapshot) -> None:
        self._snapshot = snapshot
        self._rebuild_layout()
        self.update()

    def sizeHint(self) -> QSize:
        return self._content_size

    def mousePressEvent(self, event) -> None:
        position = event.position()
        for node in reversed(list(self._nodes.values())):
            if node.rect.contains(position):
                modifiers = event.modifiers()
                additive = bool(
                    modifiers
                    & (
                        Qt.KeyboardModifier.ControlModifier
                        | Qt.KeyboardModifier.ShiftModifier
                    )
                )
                toggle = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                self.select_node(node.node_id, additive=additive, toggle=toggle)
                self.node_selected.emit(node)
                return

        edge = self._edge_at(position)
        if edge is not None:
            self.select_edge(edge.edge_id)
            self.edge_selected.emit(edge)
            return

        self._selected_edge_id = None
        self.update()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        position = event.position()
        for node in reversed(list(self._nodes.values())):
            if node.rect.contains(position):
                if node.kind == "checkpoint":
                    self.select_node(node.node_id)
                    self.node_selected.emit(node)
                    self.node_double_clicked.emit(node)
                    return
                break

        edge = self._edge_at(position)
        if edge is not None:
            self.select_edge(edge.edge_id)
            self.edge_selected.emit(edge)
            self.edge_double_clicked.emit(edge)
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Delete:
            self.delete_key_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#fbfdff"))

        if not self._nodes:
            painter.setPen(QColor("#64748b"))
            painter.drawText(
                self.rect().adjusted(24, 24, -24, -24),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                "No checkpoints yet.\nStart training to create the first lineage.",
            )
            return

        for edge in self._edges:
            selected = self._selected_edge_id == edge.edge_id
            painter.setPen(QPen(QColor("#0f766e") if selected else QColor("#94a3b8"), 3 if selected else 2))
            painter.drawLine(edge.source_point, edge.target_point)

        for node in self._nodes.values():
            selected = node.node_id in self._selected_node_ids
            if node.kind == "root":
                fill = QColor("#f1f5f9")
                border = QColor("#0f766e") if selected else QColor("#94a3b8")
            else:
                fill = QColor("#dbeafe") if selected else QColor("#ffffff")
                border = QColor("#2563eb") if selected else QColor("#94a3b8")

            painter.setPen(QPen(border, 2))
            painter.setBrush(fill)
            painter.drawRoundedRect(node.rect, 12, 12)

            painter.setPen(QColor("#0f172a"))
            title_rect = node.rect.adjusted(10, 8, -10, -24)
            sub_rect = node.rect.adjusted(10, 28, -10, -8)
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, node.label)
            painter.setPen(QColor("#475569"))
            painter.drawText(sub_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, node.sublabel)
            if node.kind == "checkpoint":
                self._draw_success_indicator(painter, node)

    def _rebuild_layout(self) -> None:
        self._nodes = {}
        self._edges = []

        checkpoints_by_id = {checkpoint.checkpoint_id: checkpoint for checkpoint in self._snapshot.checkpoints}
        runs_by_id = {run.run_id: run for run in self._snapshot.runs}

        root_ids: set[str] = set()
        children: dict[str, list[str]] = {}

        for checkpoint in self._snapshot.checkpoints:
            parent_id = checkpoint.parent_checkpoint_id
            if parent_id and parent_id in checkpoints_by_id:
                children.setdefault(parent_id, []).append(checkpoint.checkpoint_id)
                continue

            root_id = self._root_id_for_checkpoint(checkpoint)
            root_ids.add(root_id)
            children.setdefault(root_id, []).append(checkpoint.checkpoint_id)

        for parent_id, child_ids in children.items():
            child_ids.sort(key=lambda checkpoint_id: self._checkpoint_sort_key(checkpoints_by_id[checkpoint_id]))
            children[parent_id] = child_ids

        next_x = 0.0
        positions: dict[str, tuple[float, int]] = {}

        def assign(node_id: str, depth: int) -> float:
            nonlocal next_x
            child_ids = children.get(node_id, [])
            if not child_ids:
                x = next_x
                positions[node_id] = (x, depth)
                next_x += 1.0
                return x

            child_positions = [assign(child_id, depth + 1) for child_id in child_ids]
            x = sum(child_positions) / len(child_positions)
            positions[node_id] = (x, depth)
            return x

        ordered_root_ids = sorted(root_ids)
        for index, root_id in enumerate(ordered_root_ids):
            assign(root_id, 0)
            if index < len(ordered_root_ids) - 1:
                next_x += 1.0

        horizontal_gap = 260.0
        vertical_gap = 140.0
        left_margin = 120.0
        top_margin = 70.0

        for root_id in ordered_root_ids:
            slot_x, depth = positions[root_id]
            center = QPointF(left_margin + slot_x * horizontal_gap, top_margin + depth * vertical_gap)
            environment_id, algorithm = self._root_parts(root_id)
            rect = QRectF(center.x() - 86.0, center.y() - 28.0, 172.0, 56.0)
            self._nodes[root_id] = _LineageNode(
                node_id=root_id,
                kind="root",
                label="Untrained Agent",
                sublabel=f"{environment_id} | {algorithm}",
                checkpoint=None,
                rect=rect,
                center=center,
                environment_id=environment_id,
                algorithm=algorithm,
            )

        for checkpoint in self._snapshot.checkpoints:
            slot_x, depth = positions.get(checkpoint.checkpoint_id, (0.0, 1))
            center = QPointF(left_margin + slot_x * horizontal_gap, top_margin + depth * vertical_gap)
            rect = QRectF(center.x() - 92.0, center.y() - 34.0, 184.0, 68.0)
            self._nodes[checkpoint.checkpoint_id] = _LineageNode(
                node_id=checkpoint.checkpoint_id,
                kind="checkpoint",
                label=self._checkpoint_title(checkpoint),
                sublabel=f"ep {checkpoint.episode} | step {checkpoint.step}",
                checkpoint=checkpoint,
                rect=rect,
                center=center,
                success_rate=self._checkpoint_success_rate(checkpoint),
            )

        for checkpoint in sorted(self._snapshot.checkpoints, key=self._checkpoint_sort_key):
            source_node_id = checkpoint.parent_checkpoint_id
            source_checkpoint = None
            if source_node_id is not None and source_node_id in checkpoints_by_id:
                source_checkpoint = checkpoints_by_id[source_node_id]
            else:
                source_node_id = self._root_id_for_checkpoint(checkpoint)

            source_node = self._nodes.get(source_node_id)
            target_node = self._nodes.get(checkpoint.checkpoint_id)
            if source_node is None or target_node is None:
                continue

            run = runs_by_id.get(checkpoint.run_id or "")
            task_snapshot = self._snapshot.run_task_snapshots.get(checkpoint.run_id or "") or checkpoint.task_snapshot
            self._edges.append(
                _LineageEdge(
                    edge_id=f"edge:{checkpoint.checkpoint_id}",
                    source_node_id=source_node_id,
                    target_node_id=checkpoint.checkpoint_id,
                    target_checkpoint=checkpoint,
                    source_checkpoint=source_checkpoint,
                    run=run,
                    task_snapshot=task_snapshot,
                    episodes=self._episodes_for_edge(checkpoint, source_checkpoint),
                    source_point=QPointF(source_node.center.x(), source_node.rect.bottom()),
                    target_point=QPointF(target_node.center.x(), target_node.rect.top()),
                )
            )

        width = int(left_margin * 2 + max(1.0, next_x) * horizontal_gap)
        max_depth = max((depth for _, depth in positions.values()), default=0)
        height = int(top_margin * 2 + (max_depth + 1) * vertical_gap)
        self._content_size = QSize(max(320, width), max(240, height))
        self.resize(self._content_size)
        self.updateGeometry()

    def _edge_at(self, point: QPointF) -> _LineageEdge | None:
        for edge in reversed(self._edges):
            if self._distance_to_segment(point, edge.source_point, edge.target_point) <= 10.0:
                return edge
        return None

    def _distance_to_segment(self, point: QPointF, start: QPointF, end: QPointF) -> float:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        if dx == 0.0 and dy == 0.0:
            return hypot(point.x() - start.x(), point.y() - start.y())
        projection = (
            ((point.x() - start.x()) * dx + (point.y() - start.y()) * dy)
            / (dx * dx + dy * dy)
        )
        projection = max(0.0, min(1.0, projection))
        nearest_x = start.x() + projection * dx
        nearest_y = start.y() + projection * dy
        return hypot(point.x() - nearest_x, point.y() - nearest_y)

    def _checkpoint_title(self, checkpoint: Checkpoint) -> str:
        raw_title = checkpoint.label.strip() if checkpoint.label else ""
        title = self._strip_checkpoint_number_prefix(raw_title)
        if not title:
            title = checkpoint.task_name or checkpoint.reason.replace("_", " ")
        if not title:
            title = checkpoint.checkpoint_id.replace("_", " ").title()
        if len(title) > 24:
            return title[:21] + "..."
        return title

    def _strip_checkpoint_number_prefix(self, label: str) -> str:
        return re.sub(
            r"^checkpoint[\s_-]*\d+\s*(?:\|\s*)?",
            "",
            label,
            count=1,
            flags=re.IGNORECASE,
        ).strip()

    def _checkpoint_success_rate(self, checkpoint: Checkpoint) -> float | None:
        metrics = checkpoint.metadata.get("evaluation_metrics")
        if not isinstance(metrics, dict) or "success_rate" not in metrics:
            metrics = checkpoint.metadata.get("training_metrics")
        if not isinstance(metrics, dict):
            return None
        try:
            return float(metrics.get("success_rate"))
        except (TypeError, ValueError):
            return None

    def _draw_success_indicator(self, painter: QPainter, node: _LineageNode) -> None:
        success_rate = node.success_rate if node.success_rate is not None else 0.0
        color = QColor("#16a34a") if success_rate >= 1.0 else QColor("#dc2626")
        radius = 6.0
        center = QPointF(node.rect.right() - 12.0, node.rect.bottom() - 12.0)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.setBrush(color)
        painter.drawEllipse(center, radius, radius)

    def _checkpoint_sort_key(self, checkpoint: Checkpoint) -> tuple[str, str, int, int, str]:
        return (
            checkpoint.created_at,
            checkpoint.run_id or "",
            checkpoint.episode,
            checkpoint.step,
            checkpoint.checkpoint_id,
        )

    def _root_id_for_checkpoint(self, checkpoint: Checkpoint) -> str:
        task_snapshot = checkpoint.task_snapshot
        environment_id = task_snapshot.environment_id if task_snapshot is not None else "unknown"
        algorithm = str(checkpoint.metadata.get("algorithm", "unknown"))
        return f"root:{environment_id}:{algorithm}"

    def _root_parts(self, root_id: str) -> tuple[str, str]:
        _prefix, environment_id, algorithm = root_id.split(":", 2)
        return environment_id, algorithm

    def _episodes_for_edge(
        self,
        target_checkpoint: Checkpoint,
        source_checkpoint: Checkpoint | None,
    ) -> list[EpisodeTrace]:
        if target_checkpoint.run_id is None:
            return []
        traces = self._snapshot.episodes_by_run.get(target_checkpoint.run_id, [])
        start_episode = 0
        if source_checkpoint is not None and source_checkpoint.run_id == target_checkpoint.run_id:
            start_episode = source_checkpoint.episode
        return [
            trace
            for trace in traces
            if start_episode < trace.episode_id <= target_checkpoint.episode
        ]

class CheckpointHistoryView(QWidget):
    inspect_episode_requested = Signal(object)
    checkpoint_import_requested = Signal(object)
    checkpoint_delete_requested = Signal(object)
    checkpoint_evaluation_requested = Signal(object)
    checkpoint_rename_requested = Signal(object)
    checkpoint_normalize_requested = Signal(object, int)
    curriculum_import_requested = Signal(object)
    training_run_config_selected = Signal(object)
    training_edge_live_edit_requested = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self._snapshot = TrainingHistorySnapshot([], [], {}, {})
        self._current_segment_episodes: list[EpisodeTrace] = []
        self._selected_start_node_id: str | None = None
        self._q_table_checkpoint: Checkpoint | None = None
        self._sb3_policy_checkpoint: Checkpoint | None = None
        self._current_segment_task_snapshot: TaskSnapshot | None = None

        root = QVBoxLayout(self)

        title = QLabel("Checkpoint History")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Browse checkpoint lineage top-down, inspect each training run, and jump directly to recorded episodes."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        left_panel = QWidget(splitter)
        left_layout = QVBoxLayout(left_panel)
        left_group = QGroupBox("Checkpoint Lineage", left_panel)
        left_group_layout = QVBoxLayout(left_group)
        self.graph_widget = CheckpointGraphWidget()
        graph_scroll = QScrollArea(left_group)
        graph_scroll.setWidgetResizable(False)
        graph_scroll.setWidget(self.graph_widget)
        left_group_layout.addWidget(graph_scroll)
        left_layout.addWidget(left_group)

        right_panel = QWidget(splitter)
        right_layout = QVBoxLayout(right_panel)

        self.training_source_label = QLabel("Training start checkpoint: scratch")
        self.training_source_label.setWordWrap(True)
        self.export_curriculum_button = QPushButton("Export Trace", right_panel)
        self.export_curriculum_button.setToolTip(
            "Export the ordered curriculum with recorded episode traces."
        )
        self.export_curriculum_button.setEnabled(False)
        self.export_curriculum_button.setVisible(False)
        self.export_curriculum_plan_button = QPushButton("Export Curriculum", right_panel)
        self.export_curriculum_plan_button.setToolTip(
            "Export only the executable curriculum structure: tasks and training steps."
        )
        self.export_curriculum_plan_button.setEnabled(False)
        self.show_training_report_button = QPushButton("Show Training Report", right_panel)
        self.show_training_report_button.setToolTip(
            "Show aggregate training information from the lineage start to the selected checkpoint(s)."
        )
        self.show_training_report_button.setEnabled(False)
        self.normalize_checkpoint_button = QPushButton("Normalize", right_panel)
        self.normalize_checkpoint_button.setToolTip(
            "Extend the selected checkpoint path to a fixed total episode count without updating the model."
        )
        self.normalize_checkpoint_button.setEnabled(False)
        self.export_checkpoint_button = QPushButton("Export Checkpoint", right_panel)
        self.export_checkpoint_button.setToolTip(
            "Export only the selected checkpoint JSON."
        )
        self.export_checkpoint_button.setEnabled(False)
        self.delete_checkpoint_button = QPushButton("Delete Checkpoint", right_panel)
        self.delete_checkpoint_button.setToolTip(
            "Delete the selected checkpoint(s) and all descendant checkpoints. Shortcut: Delete."
        )
        self.delete_checkpoint_button.setEnabled(False)
        self.delete_checkpoint_button.setVisible(False)
        self.evaluate_checkpoint_button = QPushButton("Run Evaluation", right_panel)
        self.evaluate_checkpoint_button.setToolTip(
            "Evaluate the selected checkpoint using the Evaluation tab settings. Double-click a checkpoint to open this action."
        )
        self.evaluate_checkpoint_button.setEnabled(False)
        self.evaluate_checkpoint_button.setVisible(False)
        self.live_edit_button = QPushButton("Live Edit", right_panel)
        self.live_edit_button.setToolTip(
            "Edit this training edge and replay the deepest descendant branch. Double-click a branch to start live edit."
        )
        self.live_edit_button.setEnabled(False)
        self.live_edit_button.setVisible(False)
        self.show_q_table_button = QPushButton("Show Q Table", right_panel)
        self.show_q_table_button.setToolTip(
            "Open the Q-learning table stored in the selected checkpoint."
        )
        self.show_q_table_button.setEnabled(False)
        self.show_q_table_button.setVisible(False)
        self.show_policy_button = QPushButton("Show Policy", right_panel)
        self.show_policy_button.setToolTip(
            "Open the deterministic greedy policy map for the selected SB3 checkpoint."
        )
        self.show_policy_button.setEnabled(False)
        self.show_policy_button.setVisible(False)
        self.import_checkpoint_button = QPushButton("Import Checkpoint", right_panel)
        self.import_checkpoint_button.setToolTip(
            "Import a checkpoint or curriculum JSON. The file content decides which importer is used."
        )
        self.import_checkpoint_button.setText("Import")
        self.import_curriculum_button = QPushButton("Import Curriculum", right_panel)
        self.import_curriculum_button.setToolTip(
            "Import a checkpoint or curriculum JSON. The file content decides which importer is used."
        )
        self.import_curriculum_button.setVisible(False)
        self.selection_label = QLabel("Select a training edge to inspect a run.")
        self.selection_label.setWordWrap(True)

        self.details_group = QGroupBox("Node Details", right_panel)
        checkpoint_layout = QFormLayout(self.details_group)
        self.checkpoint_details = QTextEdit(self.details_group)
        self.checkpoint_details.setReadOnly(True)
        self.checkpoint_details.setMinimumHeight(80)
        checkpoint_layout.addRow(self.checkpoint_details)

        self.segment_group = QGroupBox("Run Episodes", right_panel)
        segment_layout = QVBoxLayout(self.segment_group)
        self.segment_details = QTextEdit(self.segment_group)
        self.segment_details.setReadOnly(True)
        self.segment_details.setMinimumHeight(80)
        self.episode_list = QListWidget(self.segment_group)
        self.episode_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.inspect_episode_button = QPushButton("Inspect Selected Episode", self.segment_group)
        self.inspect_episode_button.setEnabled(False)
        self.state_visit_heatmap_button = QPushButton("Heatmap", self.segment_group)
        self.state_visit_heatmap_button.setToolTip(
            "Show a map heatmap of state visits across the listed episodes."
        )
        self.state_visit_heatmap_button.setEnabled(False)

        segment_layout.addWidget(self.segment_details)
        segment_layout.addWidget(self.episode_list, 1)
        episode_actions_layout = QHBoxLayout()
        episode_actions_layout.addWidget(self.inspect_episode_button, 4)
        episode_actions_layout.addWidget(self.state_visit_heatmap_button, 1)
        segment_layout.addLayout(episode_actions_layout)

        actions_layout = QVBoxLayout()
        import_export_buttons_layout = QHBoxLayout()
        import_export_buttons_layout.addWidget(self.import_checkpoint_button)
        import_export_buttons_layout.addWidget(self.export_checkpoint_button)
        import_export_buttons_layout.addWidget(self.export_curriculum_plan_button)
        import_export_buttons_layout.addStretch(1)

        secondary_buttons_layout = QHBoxLayout()
        secondary_buttons_layout.addWidget(self.show_training_report_button)
        secondary_buttons_layout.addWidget(self.normalize_checkpoint_button)
        secondary_buttons_layout.addWidget(self.delete_checkpoint_button)
        secondary_buttons_layout.addWidget(self.import_curriculum_button)
        secondary_buttons_layout.addWidget(self.live_edit_button)
        secondary_buttons_layout.addWidget(self.export_curriculum_button)
        secondary_buttons_layout.addWidget(self.evaluate_checkpoint_button)
        secondary_buttons_layout.addWidget(self.show_q_table_button)
        secondary_buttons_layout.addWidget(self.show_policy_button)
        secondary_buttons_layout.addStretch(1)

        actions_layout.addLayout(import_export_buttons_layout)
        actions_layout.addLayout(secondary_buttons_layout)

        right_layout.addWidget(self.training_source_label)
        right_layout.addLayout(actions_layout)
        right_layout.addWidget(self.selection_label)
        right_layout.addWidget(self.details_group)
        right_layout.addWidget(self.segment_group, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([620, 420])

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addWidget(splitter, 1)

        self.graph_widget.node_selected.connect(self._show_node_details)
        self.graph_widget.edge_selected.connect(self._on_edge_selected)
        self.graph_widget.node_double_clicked.connect(self._open_checkpoint_node_actions)
        self.graph_widget.edge_double_clicked.connect(self._request_live_edit_for_edge)
        self.graph_widget.delete_key_pressed.connect(self._emit_delete_selected_checkpoints)
        self.episode_list.currentRowChanged.connect(self._on_episode_selection_changed)
        self.inspect_episode_button.clicked.connect(self._emit_inspect_selected_episode)
        self.state_visit_heatmap_button.clicked.connect(self._show_state_visit_heatmap)
        self.export_curriculum_button.clicked.connect(
            lambda _checked=False: self._export_selected_curriculum(include_episode_traces=True)
        )
        self.export_curriculum_plan_button.clicked.connect(self._export_selected_curriculum_plan)
        self.show_training_report_button.clicked.connect(self._show_training_report_for_selected_checkpoint)
        self.normalize_checkpoint_button.clicked.connect(self._open_normalize_selected_checkpoint)
        self.export_checkpoint_button.clicked.connect(self._export_selected_checkpoint)
        self.delete_checkpoint_button.clicked.connect(self._emit_delete_selected_checkpoints)
        self.evaluate_checkpoint_button.clicked.connect(self._emit_evaluate_selected_checkpoint)
        self.live_edit_button.clicked.connect(self._request_live_edit_for_selected_edge)
        self.show_q_table_button.clicked.connect(self._show_selected_q_table)
        self.show_policy_button.clicked.connect(self._show_selected_policy)
        self.import_checkpoint_button.clicked.connect(self._import_from_file)
        self.import_curriculum_button.clicked.connect(self._import_from_file)

        self._render_empty_selection()

    def selected_checkpoint(self) -> Checkpoint | None:
        node_id = self._selected_start_node_id
        if node_id is None:
            return None
        node = self.graph_widget.node_for_id(node_id)
        if node is None:
            return None
        return node.checkpoint

    def selected_checkpoints(self) -> list[Checkpoint]:
        return self._selected_training_report_checkpoints()

    def start_from_scratch_selected(self) -> bool:
        node_id = self._selected_start_node_id
        if node_id is None:
            return not self._snapshot.checkpoints
        node = self.graph_widget.node_for_id(node_id)
        return node is not None and node.kind == "root"

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Delete:
            self._emit_delete_selected_checkpoints()
            event.accept()
            return
        super().keyPressEvent(event)

    def set_history(self, snapshot: TrainingHistorySnapshot) -> None:
        selected_node_ids = list(self.graph_widget.selected_node_ids)
        if not selected_node_ids and self._selected_start_node_id is not None:
            selected_node_ids = [self._selected_start_node_id]
        selected_edge_id = self.graph_widget.selected_edge_id
        self._snapshot = snapshot
        self.graph_widget.set_history(snapshot)

        resolved_nodes = [
            node
            for node_id in selected_node_ids
            if (node := self.graph_widget.node_for_id(node_id)) is not None
        ]
        if not resolved_nodes:
            latest_checkpoint = self._latest_checkpoint()
            if latest_checkpoint is not None:
                latest_node = self.graph_widget.node_for_id(latest_checkpoint.checkpoint_id)
                if latest_node is not None:
                    resolved_nodes = [latest_node]
        self._selected_start_node_id = resolved_nodes[-1].node_id if resolved_nodes else None

        if resolved_nodes:
            self.graph_widget.select_nodes([node.node_id for node in resolved_nodes])
            self._show_node_details(resolved_nodes[-1])
        else:
            self.graph_widget.select_nodes([])
            self._render_empty_selection()

        if selected_edge_id is not None:
            edge = self.graph_widget.edge_for_id(selected_edge_id)
            if edge is not None:
                self.graph_widget.select_edge(selected_edge_id)
                self._show_edge_details(edge)
                return

        if not resolved_nodes or resolved_nodes[-1].kind == "root":
            self._render_empty_segment_selection()

    def _render_empty_selection(self) -> None:
        self.details_group.setTitle("Node Details")
        if self._snapshot.checkpoints:
            self.training_source_label.setText("Training start checkpoint: pending selection")
        else:
            self.training_source_label.setText("Training start checkpoint: scratch")
        self.selection_label.setText(
            f"History contains {len(self._snapshot.checkpoints)} checkpoint(s) across {len(self._snapshot.runs)} run(s). "
            "Select a training edge to inspect a run."
        )
        self.checkpoint_details.setHtml(
            self._details_panel_html(
                heading="Node Details",
                rows=[],
                empty_message="Checkpoint or root-node details will appear here.",
            )
        )
        self.segment_group.setTitle("Run Episodes")
        self.segment_details.setPlainText("Training run details and recorded episodes will appear here.")
        self.episode_list.clear()
        self._current_segment_episodes = []
        self._current_segment_task_snapshot = None
        self.inspect_episode_button.setEnabled(False)
        self.state_visit_heatmap_button.setEnabled(False)
        self._set_export_buttons_enabled(False)
        self._set_live_edit_button_enabled(False)
        self._set_q_table_checkpoint(None)
        self._set_sb3_policy_checkpoint(None)

    def _render_empty_segment_selection(self) -> None:
        self.segment_group.setTitle("Run Episodes")
        self.selection_label.setText("Select a training edge to inspect a run.")
        self.segment_details.setPlainText("Training run details and recorded episodes will appear here.")
        self.episode_list.clear()
        self._current_segment_episodes = []
        self._current_segment_task_snapshot = None
        self.inspect_episode_button.setEnabled(False)
        self.state_visit_heatmap_button.setEnabled(False)
        self._set_live_edit_button_enabled(False)

    def _show_node_details(self, node: _LineageNode) -> None:
        self.details_group.setTitle("Node Details")
        self._selected_start_node_id = node.node_id
        self._set_q_table_checkpoint(node.checkpoint)
        self._set_sb3_policy_checkpoint(node.checkpoint)
        selected_nodes = self.graph_widget.selected_nodes()
        if node.node_id not in {selected_node.node_id for selected_node in selected_nodes}:
            selected_nodes = [node]
        selected_checkpoint_nodes = [
            selected_node
            for selected_node in selected_nodes
            if selected_node.checkpoint is not None
        ]
        if len(selected_checkpoint_nodes) > 1:
            self.training_source_label.setText(
                f"Training start checkpoint: {node.checkpoint.label if node.checkpoint is not None else 'pending selection'}"
            )
            self.selection_label.setText(f"Selected checkpoints: {len(selected_checkpoint_nodes)}")
            self._set_checkpoint_comparison_details(selected_checkpoint_nodes)
            self._set_export_buttons_enabled(node.checkpoint is not None)
            self._set_live_edit_button_enabled(False)
            if node.checkpoint is not None:
                self._set_checkpoint_evaluation_episodes(node.checkpoint)
            else:
                self._render_empty_segment_selection()
            return

        if node.kind == "root":
            self.training_source_label.setText("Training start checkpoint: scratch")
            self._set_root_details(node)
            self._render_empty_segment_selection()
            self._set_export_buttons_enabled(False)
            self._set_live_edit_button_enabled(False)
            self._set_q_table_checkpoint(None)
            self._set_sb3_policy_checkpoint(None)
            return

        checkpoint = node.checkpoint
        if checkpoint is None:
            self._render_empty_selection()
            return

        self.training_source_label.setText(f"Training start checkpoint: {checkpoint.label}")
        self._set_checkpoint_details(checkpoint, heading="Selected checkpoint")
        self._set_export_buttons_enabled(True)
        self._set_live_edit_button_enabled(False)
        self._set_checkpoint_evaluation_episodes(checkpoint)

    def _show_edge_details(self, edge: _LineageEdge) -> None:
        self.details_group.setTitle("Training Details")
        self.segment_group.setTitle("Training Run")
        self._set_q_table_checkpoint(None)
        self._set_sb3_policy_checkpoint(None)
        checkpoint = edge.target_checkpoint
        run = edge.run
        self._current_segment_task_snapshot = edge.task_snapshot or checkpoint.task_snapshot
        self._set_export_buttons_enabled(True)
        self._set_live_edit_button_enabled(self._live_edit_config_for_edge(edge) is not None)
        self.selection_label.setText(
            f"Selected training run: {run.run_id if run is not None else checkpoint.run_id or 'unknown'}"
        )
        self._set_training_run_details(edge)

        self.segment_details.setPlainText(f"Recorded training episodes: {len(edge.episodes)}")

        self.episode_list.clear()
        self._current_segment_episodes = list(edge.episodes)
        for trace in self._current_segment_episodes:
            self.episode_list.addItem(
                QListWidgetItem(
                    f"Episode {trace.episode_id} | reward={trace.total_reward:.2f} | success={trace.success}"
                )
            )

        if self._current_segment_episodes:
            self.episode_list.setCurrentRow(0)
            self.inspect_episode_button.setEnabled(True)
        else:
            self.inspect_episode_button.setEnabled(False)
        self._refresh_state_visit_heatmap_button()

    def _on_edge_selected(self, edge: _LineageEdge) -> None:
        self._show_edge_details(edge)
        config = self._training_config_for_edge(edge)
        if config is not None:
            self.training_run_config_selected.emit(config)

    def _training_config_for_edge(self, edge: _LineageEdge) -> RunConfig | None:
        run_config = self._run_config_payload(edge.run)
        if run_config is None:
            return None
        return RunConfig.from_dict(run_config)

    def _live_edit_config_for_edge(self, edge: _LineageEdge) -> RunConfig | None:
        config = self._training_config_for_edge(edge)
        if config is None:
            return None
        return self._config_limited_to_edge_segment(
            config,
            source_checkpoint=edge.source_checkpoint,
            target_checkpoint=edge.target_checkpoint,
        )

    def _config_limited_to_edge_segment(
        self,
        config: RunConfig,
        *,
        source_checkpoint: Checkpoint | None,
        target_checkpoint: Checkpoint,
    ) -> RunConfig:
        segment_config = RunConfig.from_dict(config.to_dict())
        segment_steps = target_checkpoint.step
        segment_episodes = target_checkpoint.episode
        if (
            source_checkpoint is not None
            and source_checkpoint.run_id is not None
            and source_checkpoint.run_id == target_checkpoint.run_id
        ):
            segment_steps = max(0, target_checkpoint.step - source_checkpoint.step)
            segment_episodes = max(0, target_checkpoint.episode - source_checkpoint.episode)

        if segment_episodes > 0:
            segment_config.max_episodes = segment_episodes
        if segment_config.max_steps is not None or segment_episodes <= 0:
            segment_config.max_steps = segment_steps if segment_steps > 0 else segment_config.max_steps
        return segment_config

    def _request_live_edit_for_selected_edge(self) -> None:
        selected_edge_id = self.graph_widget.selected_edge_id
        edge = self.graph_widget.edge_for_id(selected_edge_id) if selected_edge_id is not None else None
        if edge is None:
            QMessageBox.information(
                self,
                "Live Edit",
                "Select a training edge before starting live edit.",
            )
            return

        self._request_live_edit_for_edge(edge)

    def _request_live_edit_for_edge(self, edge: _LineageEdge) -> None:
        self.graph_widget.select_edge(edge.edge_id)
        self._show_edge_details(edge)

        config = self._live_edit_config_for_edge(edge)
        if config is None:
            QMessageBox.information(
                self,
                "Live Edit",
                "This training edge does not contain a replayable run configuration.",
            )
            return

        dialog = _TrainingEdgeEditDialog(config, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.training_edge_live_edit_requested.emit(edge, dialog.edited_config())

    def _set_checkpoint_evaluation_episodes(self, checkpoint: Checkpoint) -> None:
        self.segment_group.setTitle("Evaluation Episodes")
        evaluation = checkpoint.metadata.get("evaluation")
        evaluation_error = checkpoint.metadata.get("evaluation_error")
        episodes = self._evaluation_episodes_for_checkpoint(checkpoint)
        self._current_segment_task_snapshot = self._task_snapshot_for_heatmap(
            episodes,
            fallback=checkpoint.task_snapshot,
        )
        self.episode_list.clear()
        self._current_segment_episodes = list(episodes)

        if isinstance(evaluation, dict):
            lines = [
                f"Evaluation run ID: {evaluation.get('run_id', 'unknown')}",
                f"Task: {evaluation.get('task_name', 'unknown')}",
                f"Environment: {evaluation.get('environment_id', 'unknown')}",
                f"Episodes requested: {evaluation.get('episode_count', 'unknown')}",
                f"Max steps / episode: {evaluation.get('max_steps_per_episode') or 'no limit'}",
                f"Seed: {evaluation.get('seed') if evaluation.get('seed') is not None else 'random'}",
                f"Recorded evaluation episodes: {len(episodes)}",
            ]
        elif evaluation_error is not None:
            lines = [
                "Evaluation failed for this checkpoint.",
                f"Error: {evaluation_error}",
            ]
        else:
            lines = [
                "No evaluation is attached to this checkpoint.",
                "Select the training edge to inspect recorded training episodes.",
            ]
        self.segment_details.setPlainText("\n".join(lines))

        for trace in self._current_segment_episodes:
            self.episode_list.addItem(
                QListWidgetItem(
                    f"Evaluation episode {trace.episode_id} | reward={trace.total_reward:.2f} | success={trace.success}"
                )
            )

        if self._current_segment_episodes:
            self.episode_list.setCurrentRow(0)
            self.inspect_episode_button.setEnabled(True)
        else:
            self.inspect_episode_button.setEnabled(False)
        self._refresh_state_visit_heatmap_button()

    def _evaluation_episodes_for_checkpoint(self, checkpoint: Checkpoint) -> list[EpisodeTrace]:
        evaluation = checkpoint.metadata.get("evaluation")
        if not isinstance(evaluation, dict):
            return []
        run_id = evaluation.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return []
        return list(self._snapshot.episodes_by_run.get(run_id, []))

    def _on_episode_selection_changed(self, row: int) -> None:
        self.inspect_episode_button.setEnabled(0 <= row < len(self._current_segment_episodes))

    def _emit_inspect_selected_episode(self) -> None:
        row = self.episode_list.currentRow()
        if row < 0 or row >= len(self._current_segment_episodes):
            return
        self.inspect_episode_requested.emit(self._current_segment_episodes[row])

    def _refresh_state_visit_heatmap_button(self) -> None:
        self.state_visit_heatmap_button.setEnabled(self._state_visit_heatmap_context() is not None)

    def _show_state_visit_heatmap(self) -> None:
        context = self._state_visit_heatmap_context()
        if context is None:
            QMessageBox.information(
                self,
                "State Visit Heatmap",
                "Select a Frozen Lake node or branch with recorded episodes first.",
            )
            return

        map_rows, visit_counts, episodes = context
        self._build_state_visit_heatmap_dialog(
            map_rows=map_rows,
            visit_counts=visit_counts,
            episodes=episodes,
        ).exec()

    def _build_state_visit_heatmap_dialog(
        self,
        *,
        map_rows: list[str],
        visit_counts: dict[int, int],
        episodes: list[EpisodeTrace],
    ) -> _StateVisitHeatmapDialog:
        return _StateVisitHeatmapDialog(
            title="State Visit Heatmap",
            map_rows=map_rows,
            visit_counts=visit_counts,
            episode_count=len(episodes),
            parent=self,
        )

    def _state_visit_heatmap_context(self) -> tuple[list[str], dict[int, int], list[EpisodeTrace]] | None:
        episodes = list(self._current_segment_episodes)
        if not episodes:
            return None

        task_snapshot = self._task_snapshot_for_heatmap(
            episodes,
            fallback=self._current_segment_task_snapshot,
        )
        map_rows = _frozen_lake_map_rows_from_task_snapshot(task_snapshot)
        if map_rows is None:
            return None

        visit_counts = self._state_visit_counts(episodes, map_rows)
        return map_rows, visit_counts, episodes

    def _task_snapshot_for_heatmap(
        self,
        episodes: list[EpisodeTrace],
        *,
        fallback: TaskSnapshot | None,
    ) -> TaskSnapshot | None:
        for trace in episodes:
            task_snapshot = trace.task_snapshot
            if task_snapshot is not None and task_snapshot.environment_id == "frozen_lake":
                return task_snapshot
        if fallback is not None and fallback.environment_id == "frozen_lake":
            return fallback
        return None

    def _state_visit_counts(
        self,
        episodes: list[EpisodeTrace],
        map_rows: list[str],
    ) -> dict[int, int]:
        state_count = len(map_rows) * (len(map_rows[0]) if map_rows else 0)
        visit_counts: dict[int, int] = {}
        for trace in episodes:
            for state_index in self._state_visit_sequence(trace):
                if state_index is None or state_index < 0 or state_index >= state_count:
                    continue
                visit_counts[state_index] = visit_counts.get(state_index, 0) + 1
        return visit_counts

    def _state_visit_sequence(self, trace: EpisodeTrace) -> list[int | None]:
        if trace.moments:
            states: list[int | None] = []
            for moment in trace.moments:
                state_index = _coerce_state_index(moment.observation)
                if state_index is None:
                    state_index = _state_index_from_restorable_state(moment.restorable_env_state)
                states.append(state_index)
            return states

        initial_observation = trace.initial_observation
        if initial_observation is None and trace.steps:
            initial_observation = trace.steps[0].observation
        return [
            _coerce_state_index(initial_observation),
            *(_coerce_state_index(step.next_observation) for step in trace.steps),
        ]

    def _set_export_buttons_enabled(self, enabled: bool) -> None:
        self.export_curriculum_button.setEnabled(enabled)
        self.export_curriculum_plan_button.setEnabled(enabled)
        self.show_training_report_button.setEnabled(enabled)
        self.export_checkpoint_button.setEnabled(enabled)
        self.delete_checkpoint_button.setEnabled(enabled)
        self.evaluate_checkpoint_button.setEnabled(enabled)
        self.normalize_checkpoint_button.setEnabled(enabled and self.selected_checkpoint() is not None)

    def _set_live_edit_button_enabled(self, enabled: bool) -> None:
        self.live_edit_button.setEnabled(enabled)

    def _set_q_table_checkpoint(self, checkpoint: Checkpoint | None) -> None:
        enabled = _checkpoint_has_q_learning_state(checkpoint)
        self._q_table_checkpoint = checkpoint if enabled else None
        self.show_q_table_button.setVisible(False)
        self.show_q_table_button.setEnabled(enabled)

    def _set_sb3_policy_checkpoint(self, checkpoint: Checkpoint | None) -> None:
        enabled = _checkpoint_has_sb3_policy_state(checkpoint)
        self._sb3_policy_checkpoint = checkpoint if enabled else None
        self.show_policy_button.setVisible(False)
        self.show_policy_button.setEnabled(enabled)

    def _show_selected_q_table(self) -> None:
        if self._q_table_checkpoint is None:
            return
        self._build_q_table_dialog(self._q_table_checkpoint).exec()

    def _build_q_table_dialog(self, checkpoint: Checkpoint) -> _QTableDialog:
        return _QTableDialog(checkpoint, self)

    def _show_selected_policy(self) -> None:
        if self._sb3_policy_checkpoint is None:
            return
        self._build_policy_dialog(self._sb3_policy_checkpoint).exec()

    def _build_policy_dialog(self, checkpoint: Checkpoint) -> _SB3PolicyDialog:
        return _SB3PolicyDialog(checkpoint, self)

    def _show_training_report_for_selected_checkpoint(self) -> None:
        checkpoints = self._selected_training_report_checkpoints()
        if not checkpoints:
            QMessageBox.warning(
                self,
                "Training Report",
                "Select one or more checkpoints before showing a training report.",
            )
            return
        self._build_training_report_dialog(checkpoints).exec()

    def _open_normalize_selected_checkpoint(self) -> None:
        checkpoint = self.selected_checkpoint()
        if checkpoint is None:
            QMessageBox.warning(
                self,
                "Normalize Checkpoint",
                "Select a checkpoint before normalizing.",
            )
            return

        dialog = self._build_normalize_checkpoint_dialog(checkpoint)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        additional_episodes = dialog.additional_episodes()
        if additional_episodes <= 0:
            return
        self.checkpoint_normalize_requested.emit(checkpoint, additional_episodes)

    def _build_normalize_checkpoint_dialog(self, checkpoint: Checkpoint) -> _NormalizeCheckpointDialog:
        return _NormalizeCheckpointDialog(
            checkpoint=checkpoint,
            current_episodes=self._cumulative_episodes_to_checkpoint(checkpoint),
            parent=self,
        )

    def _cumulative_episodes_to_checkpoint(self, target_checkpoint: Checkpoint) -> int:
        total_episodes = 0
        previous_checkpoint: Checkpoint | None = None
        for checkpoint in self._checkpoint_lineage(target_checkpoint) or [target_checkpoint]:
            total_episodes += self._training_episode_delta(checkpoint, previous_checkpoint)
            previous_checkpoint = checkpoint
        return total_episodes

    def _build_training_report_dialog(
        self,
        checkpoint_or_checkpoints: Checkpoint | list[Checkpoint] | tuple[Checkpoint, ...],
    ) -> _TrainingReportDialog:
        if isinstance(checkpoint_or_checkpoints, Checkpoint):
            checkpoints = [checkpoint_or_checkpoints]
        else:
            checkpoints = list(checkpoint_or_checkpoints)

        reports = [self._training_report_payload(checkpoint) for checkpoint in checkpoints]
        if len(reports) == 1:
            return _TrainingReportDialog(reports[0], self)
        return _TrainingReportDialog({"reports": reports}, self)

    def _export_selected_curriculum(self, *, include_episode_traces: bool) -> None:
        checkpoint = self._selected_export_checkpoint()
        if checkpoint is None:
            QMessageBox.warning(
                self,
                "Export Trace",
                "Select a checkpoint before exporting a trace.",
            )
            return

        suffix = "" if include_episode_traces else "_without_traces"
        default_path = f"trace_{checkpoint.checkpoint_id}{suffix}.json"
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Trace",
            default_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not selected_path:
            return

        path = Path(selected_path).expanduser()
        if path.suffix == "":
            path = path.with_suffix(".json")

        payload = self._curriculum_export_payload(
            checkpoint,
            include_episode_traces=include_episode_traces,
        )
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Export Trace",
                f"Could not write trace export:\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Export Trace",
            f"Trace exported to:\n{path}",
        )

    def _export_selected_curriculum_plan(self) -> None:
        checkpoint = self._selected_export_checkpoint()
        if checkpoint is None:
            QMessageBox.warning(
                self,
                "Export Curriculum",
                "Select a checkpoint before exporting a curriculum.",
            )
            return

        default_path = f"curriculum_{checkpoint.checkpoint_id}.json"
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Curriculum",
            default_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not selected_path:
            return

        path = Path(selected_path).expanduser()
        if path.suffix == "":
            path = path.with_suffix(".json")

        payload = self._curriculum_plan_export_payload(checkpoint)
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Export Curriculum",
                f"Could not write curriculum export:\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Export Curriculum",
            f"Curriculum exported to:\n{path}",
        )

    def _export_selected_checkpoint(self) -> None:
        checkpoint = self._selected_export_checkpoint()
        if checkpoint is None:
            QMessageBox.warning(
                self,
                "Export Checkpoint",
                "Select a checkpoint before exporting.",
            )
            return

        default_path = f"{checkpoint.checkpoint_id}.json"
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Checkpoint",
            default_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not selected_path:
            return

        path = Path(selected_path).expanduser()
        if path.suffix == "":
            path = path.with_suffix(".json")

        payload = self._checkpoint_export_payload(checkpoint)
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Export Checkpoint",
                f"Could not write checkpoint export:\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Export Checkpoint",
            f"Checkpoint exported to:\n{path}",
        )

    def _emit_delete_selected_checkpoints(self) -> None:
        checkpoint_ids = self._selected_delete_checkpoint_ids()
        if not checkpoint_ids:
            QMessageBox.warning(
                self,
                "Delete Checkpoint",
                "Select one or more checkpoints before deleting.",
            )
            return
        self.checkpoint_delete_requested.emit(checkpoint_ids)

    def _import_from_file(self) -> None:
        selected_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Import",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not selected_paths:
            return

        for selected_path in selected_paths:
            self._import_path(Path(selected_path).expanduser())

    def _import_path(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if self._is_curriculum_plan_payload(payload):
                self._validate_curriculum_plan_payload(payload)
                self.curriculum_import_requested.emit(payload)
                return

            checkpoint = self._checkpoint_from_import_payload(payload)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                "Import",
                f"Could not import file {path}:\n{exc}",
            )
            return

        self.checkpoint_import_requested.emit(checkpoint)

    def _import_checkpoint_from_file(self) -> None:
        self._import_from_file()

    def _import_curriculum_from_file(self) -> None:
        self._import_from_file()

    def _emit_evaluate_selected_checkpoint(self) -> None:
        checkpoint = self._selected_export_checkpoint()
        if checkpoint is None:
            QMessageBox.warning(
                self,
                "Run Evaluation",
                "Select a checkpoint before running evaluation.",
            )
            return
        self.checkpoint_evaluation_requested.emit(checkpoint)

    def _emit_rename_checkpoint_for_node(self, node: _LineageNode) -> None:
        if node.checkpoint is None:
            return
        self.checkpoint_rename_requested.emit(node.checkpoint)

    def _open_checkpoint_node_actions(self, node: _LineageNode) -> None:
        checkpoint = node.checkpoint
        if checkpoint is None:
            return

        dialog = self._build_checkpoint_node_action_dialog(checkpoint)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        action = dialog.selected_action
        if action == "rename":
            self.checkpoint_rename_requested.emit(checkpoint)
        elif action == "evaluate":
            self.checkpoint_evaluation_requested.emit(checkpoint)
        elif action == "q_table" and _checkpoint_has_q_learning_state(checkpoint):
            self._build_q_table_dialog(checkpoint).exec()
        elif action == "policy" and _checkpoint_has_sb3_policy_state(checkpoint):
            self._build_policy_dialog(checkpoint).exec()

    def _build_checkpoint_node_action_dialog(self, checkpoint: Checkpoint) -> _CheckpointNodeActionDialog:
        return _CheckpointNodeActionDialog(
            checkpoint,
            can_show_q_table=_checkpoint_has_q_learning_state(checkpoint),
            can_show_policy=_checkpoint_has_sb3_policy_state(checkpoint),
            parent=self,
        )

    def _selected_export_checkpoint(self) -> Checkpoint | None:
        selected_edge_id = self.graph_widget.selected_edge_id
        if selected_edge_id is not None:
            edge = self.graph_widget.edge_for_id(selected_edge_id)
            if edge is not None:
                return edge.target_checkpoint
        return self.selected_checkpoint()

    def _selected_training_report_checkpoints(self) -> list[Checkpoint]:
        selected_edge_id = self.graph_widget.selected_edge_id
        if selected_edge_id is not None:
            edge = self.graph_widget.edge_for_id(selected_edge_id)
            if edge is not None:
                return [edge.target_checkpoint]

        selected_checkpoints = [
            node.checkpoint
            for node in self.graph_widget.selected_nodes()
            if node.checkpoint is not None
        ]
        if selected_checkpoints:
            return selected_checkpoints

        checkpoint = self.selected_checkpoint()
        return [] if checkpoint is None else [checkpoint]

    def _selected_delete_checkpoint_ids(self) -> list[str]:
        selected_edge_id = self.graph_widget.selected_edge_id
        if selected_edge_id is not None:
            edge = self.graph_widget.edge_for_id(selected_edge_id)
            if edge is not None:
                return [edge.target_checkpoint.checkpoint_id]

        checkpoint_ids = [
            node.checkpoint.checkpoint_id
            for node in self.graph_widget.selected_nodes()
            if node.checkpoint is not None
        ]
        if checkpoint_ids:
            return checkpoint_ids

        checkpoint = self.selected_checkpoint()
        return [] if checkpoint is None else [checkpoint.checkpoint_id]

    def _checkpoint_export_payload(self, checkpoint: Checkpoint) -> dict[str, object]:
        return checkpoint.to_dict()

    def _training_report_payload(self, target_checkpoint: Checkpoint) -> dict[str, object]:
        lineage = self._checkpoint_lineage(target_checkpoint)
        if not lineage:
            lineage = [target_checkpoint]

        cumulative_reward = 0.0
        cumulative_episode_count = 0
        recorded_episode_count = 0
        reward_points: list[tuple[int, float]] = []
        task_switches: list[tuple[int, str]] = []
        previous_checkpoint: Checkpoint | None = None
        previous_task_key: str | None = None

        for checkpoint in lineage:
            task_snapshot = (
                self._snapshot.run_task_snapshots.get(checkpoint.run_id or "")
                or checkpoint.task_snapshot
            )
            task_name = task_snapshot.task_name if task_snapshot is not None else (checkpoint.task_name or "unknown")
            task_key = self._training_report_task_key(task_snapshot, task_name)
            if previous_task_key is not None and task_key != previous_task_key:
                task_switches.append((cumulative_episode_count, task_name))

            segment_episode_count = self._training_episode_delta(checkpoint, previous_checkpoint)
            segment_start_episode = 0
            if (
                previous_checkpoint is not None
                and previous_checkpoint.run_id is not None
                and previous_checkpoint.run_id == checkpoint.run_id
            ):
                segment_start_episode = previous_checkpoint.episode

            for trace in sorted(
                self._episodes_for_checkpoint_segment(checkpoint, previous_checkpoint),
                key=lambda item: item.episode_id,
            ):
                local_episode = max(1, trace.episode_id - segment_start_episode)
                global_episode = cumulative_episode_count + local_episode
                cumulative_reward += trace.total_reward
                recorded_episode_count += 1
                reward_points.append((global_episode, cumulative_reward))

            cumulative_episode_count += segment_episode_count
            previous_task_key = task_key
            previous_checkpoint = checkpoint

        return {
            "target_checkpoint_id": target_checkpoint.checkpoint_id,
            "target_label": target_checkpoint.label or target_checkpoint.checkpoint_id,
            "total_episodes": cumulative_episode_count,
            "recorded_episode_count": recorded_episode_count,
            "cumulative_reward_points": reward_points,
            "task_switches": task_switches,
        }

    def _training_report_task_key(self, task_snapshot: TaskSnapshot | None, fallback_name: str) -> str:
        if task_snapshot is None:
            return fallback_name
        return json.dumps(
            {
                "environment_id": task_snapshot.environment_id,
                "task_id": task_snapshot.task_id,
                "task_name": task_snapshot.task_name,
                "task_config": task_snapshot.task_config,
                "reward_config": task_snapshot.reward_config,
                "termination_config": task_snapshot.termination_config,
            },
            sort_keys=True,
        )

    def _training_episode_delta(
        self,
        checkpoint: Checkpoint,
        previous_checkpoint: Checkpoint | None,
    ) -> int:
        if (
            previous_checkpoint is not None
            and previous_checkpoint.run_id is not None
            and previous_checkpoint.run_id == checkpoint.run_id
        ):
            return max(0, checkpoint.episode - previous_checkpoint.episode)
        return max(0, checkpoint.episode)

    def _checkpoint_from_import_payload(self, payload: object) -> Checkpoint:
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint import file must contain a JSON object.")
        checkpoint = Checkpoint.from_dict(payload)
        if not checkpoint.checkpoint_id:
            raise ValueError("Checkpoint import is missing checkpoint_id.")
        return checkpoint

    def _curriculum_export_payload(
        self,
        target_checkpoint: Checkpoint,
        *,
        include_episode_traces: bool = True,
    ) -> dict[str, object]:
        lineage = self._checkpoint_lineage(target_checkpoint)
        if not lineage:
            lineage = [target_checkpoint]

        runs_by_id = {run.run_id: run for run in self._snapshot.runs}
        exported_tasks: list[dict[str, object]] = []
        task_refs_by_key: dict[str, str] = {}
        training_runs: list[dict[str, object]] = []
        previous_checkpoint: Checkpoint | None = None
        previous_checkpoint_id = "checkpoint_000_untrained"
        for order, checkpoint in enumerate(lineage, start=1):
            run = runs_by_id.get(checkpoint.run_id or "")
            task_snapshot = (
                self._snapshot.run_task_snapshots.get(checkpoint.run_id or "")
                or checkpoint.task_snapshot
            )
            task_ref_id = self._task_ref_id(
                task_snapshot,
                exported_tasks=exported_tasks,
                task_refs_by_key=task_refs_by_key,
            )
            episodes = self._episodes_for_checkpoint_segment(checkpoint, previous_checkpoint)
            run_payload: dict[str, object] = {
                "order": order,
                "run_id": checkpoint.run_id,
                "source_checkpoint_id": previous_checkpoint_id,
                "target_checkpoint_id": checkpoint.checkpoint_id,
                "run": None if run is None else run.to_dict(),
                "parameters": self._run_config_payload(run),
                "task_ref_id": task_ref_id,
                "recorded_episode_trace_count": len(episodes),
            }
            if include_episode_traces:
                run_payload["recorded_episode_traces"] = [trace.to_dict() for trace in episodes]
            else:
                run_payload["recorded_episode_summaries"] = [
                    self._episode_summary_payload(trace)
                    for trace in episodes
                ]
            training_runs.append(run_payload)
            previous_checkpoint = checkpoint
            previous_checkpoint_id = checkpoint.checkpoint_id

        return {
            "meta": {
                "target_checkpoint_id": target_checkpoint.checkpoint_id,
                "includes_episode_traces": include_episode_traces,
                "training_run_count": len(training_runs),
                "task_count": len(exported_tasks),
            },
            "tasks": exported_tasks,
            "training_runs": training_runs,
        }

    def _curriculum_plan_export_payload(self, target_checkpoint: Checkpoint) -> dict[str, object]:
        lineage = self._checkpoint_lineage(target_checkpoint)
        if not lineage:
            lineage = [target_checkpoint]

        runs_by_id = {run.run_id: run for run in self._snapshot.runs}
        environments: list[dict[str, object]] = []
        environment_ids_by_key: dict[str, int] = {}
        steps: list[dict[str, object]] = []
        curriculum_seed: int | None = None

        for order, checkpoint in enumerate(lineage, start=1):
            run = runs_by_id.get(checkpoint.run_id or "")
            task_snapshot = (
                self._snapshot.run_task_snapshots.get(checkpoint.run_id or "")
                or checkpoint.task_snapshot
            )
            env_id = self._curriculum_environment_ref(
                task_snapshot,
                environments=environments,
                environment_ids_by_key=environment_ids_by_key,
            )
            run_config_payload = self._run_config_payload(run) or {}
            step_payload = self._curriculum_plan_step_payload(
                order=order,
                env_id=env_id,
                run_config_payload=run_config_payload,
                checkpoint=checkpoint,
            )
            if curriculum_seed is None and isinstance(step_payload.get("seed"), int):
                curriculum_seed = int(step_payload["seed"])
            steps.append(step_payload)

        curriculum: dict[str, object] = {
            "size": len(steps),
            "steps": steps,
        }
        if curriculum_seed is not None:
            curriculum["seed"] = curriculum_seed

        payload: dict[str, object] = {
            "curriculum": curriculum,
            "environments": environments,
        }
        evaluation_payload = self._curriculum_plan_evaluation_payload(
            target_checkpoint,
            environments=environments,
            environment_ids_by_key=environment_ids_by_key,
        )
        if evaluation_payload is not None:
            payload["evaluation"] = evaluation_payload
        return payload

    def _curriculum_plan_step_payload(
        self,
        *,
        order: int,
        env_id: int | None,
        run_config_payload: dict[str, object],
        checkpoint: Checkpoint,
    ) -> dict[str, object]:
        config = RunConfig.from_dict(run_config_payload) if run_config_payload else RunConfig(
            algorithm=str(checkpoint.metadata.get("algorithm", "q_learning")),
            max_steps=checkpoint.step if checkpoint.step > 0 else None,
        )
        hyperparameters = dict(config.hyperparameters)
        step_payload: dict[str, object] = {
            "step_id": order,
            "env_id": env_id,
            "steps": config.max_steps,
            "max_episode_length": config.max_steps_per_episode,
            "algorithm": config.algorithm,
            "learning_rate": config.learning_rate,
            "discount_factor": config.gamma,
            "epsilon_start": config.epsilon,
            "episode_trace_sample_rate": config.episode_trace_sample_rate,
        }
        if config.seed is not None:
            step_payload["seed"] = config.seed
        if config.max_episodes is not None:
            step_payload["max_episodes"] = config.max_episodes
        if config.max_duration_seconds is not None:
            step_payload["max_duration_seconds"] = config.max_duration_seconds
        if "epsilon_decay" in hyperparameters:
            step_payload["epsilon_decay"] = hyperparameters["epsilon_decay"]
        if "epsilon_min" in hyperparameters:
            step_payload["epsilon_min"] = hyperparameters["epsilon_min"]
        extra_hyperparameters = {
            key: value
            for key, value in hyperparameters.items()
            if key not in {"learning_rate", "lr", "gamma", "epsilon", "epsilon_decay", "epsilon_min"}
        }
        if extra_hyperparameters:
            step_payload["hyperparameters"] = extra_hyperparameters
        if config.breakpoints:
            step_payload["breakpoints"] = [breakpoint.to_dict() for breakpoint in config.breakpoints]
        return step_payload

    def _curriculum_plan_evaluation_payload(
        self,
        checkpoint: Checkpoint,
        *,
        environments: list[dict[str, object]],
        environment_ids_by_key: dict[str, int],
    ) -> dict[str, object] | None:
        evaluation = checkpoint.metadata.get("evaluation")
        if not isinstance(evaluation, dict):
            return None
        run_id = evaluation.get("run_id")
        task_snapshot = (
            self._snapshot.run_task_snapshots.get(run_id)
            if isinstance(run_id, str)
            else None
        )
        evaluation_env = self._curriculum_environment_ref(
            task_snapshot,
            environments=environments,
            environment_ids_by_key=environment_ids_by_key,
        )
        return {
            "evaluation_env": evaluation_env,
            "eval_episodes": evaluation.get("episode_count"),
            "max_episode_length": evaluation.get("max_steps_per_episode"),
            "eval_seed": evaluation.get("seed"),
        }

    def _curriculum_environment_ref(
        self,
        task_snapshot: TaskSnapshot | None,
        *,
        environments: list[dict[str, object]],
        environment_ids_by_key: dict[str, int],
    ) -> int | None:
        if task_snapshot is None:
            return None

        task_payload = task_snapshot.to_dict()
        task_key = json.dumps(task_payload, sort_keys=True)
        existing_ref = environment_ids_by_key.get(task_key)
        if existing_ref is not None:
            return existing_ref

        env_ref = len(environments)
        environment_ids_by_key[task_key] = env_ref
        environment_payload = {
            **task_payload,
            "task_id": env_ref,
        }
        environments.append(environment_payload)
        return env_ref

    def _validate_curriculum_plan_payload(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Curriculum file must contain a JSON object.")
        curriculum = payload.get("curriculum")
        if not isinstance(curriculum, dict):
            raise ValueError("Curriculum file is missing the curriculum object.")
        steps = curriculum.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Curriculum must contain at least one training step.")
        environments = payload.get("environments")
        if not isinstance(environments, list) or not environments:
            raise ValueError("Curriculum must contain environment/task definitions.")

    def _is_curriculum_plan_payload(self, payload: object) -> bool:
        return (
            isinstance(payload, dict)
            and "checkpoint_id" not in payload
            and ("curriculum" in payload or "environments" in payload)
        )

    def _checkpoint_lineage(self, target_checkpoint: Checkpoint) -> list[Checkpoint]:
        checkpoints_by_id = {
            checkpoint.checkpoint_id: checkpoint
            for checkpoint in self._snapshot.checkpoints
        }
        lineage_reversed: list[Checkpoint] = []
        seen: set[str] = set()
        checkpoint: Checkpoint | None = target_checkpoint

        while checkpoint is not None:
            if checkpoint.checkpoint_id in seen:
                break

            seen.add(checkpoint.checkpoint_id)
            lineage_reversed.append(checkpoint)

            parent_id = checkpoint.parent_checkpoint_id
            if parent_id is None:
                break

            parent = checkpoints_by_id.get(parent_id)
            if parent is None:
                break
            checkpoint = parent

        return list(reversed(lineage_reversed))

    def _episodes_for_checkpoint_segment(
        self,
        target_checkpoint: Checkpoint,
        source_checkpoint: Checkpoint | None,
    ) -> list[EpisodeTrace]:
        if target_checkpoint.run_id is None:
            return []

        traces = self._snapshot.episodes_by_run.get(target_checkpoint.run_id, [])
        start_episode = 0
        if source_checkpoint is not None and source_checkpoint.run_id == target_checkpoint.run_id:
            start_episode = source_checkpoint.episode

        return [
            trace
            for trace in traces
            if start_episode < trace.episode_id <= target_checkpoint.episode
        ]

    def _episode_summary_payload(self, trace: EpisodeTrace) -> dict[str, object]:
        return {
            "episode_id": trace.episode_id,
            "run_id": trace.run_id,
            "total_reward": trace.total_reward,
            "success": trace.success,
            "step_count": len(trace.steps),
            "moment_count": len(trace.moments),
        }

    def _task_ref_id(
        self,
        task_snapshot: TaskSnapshot | None,
        *,
        exported_tasks: list[dict[str, object]],
        task_refs_by_key: dict[str, str],
    ) -> str | None:
        if task_snapshot is None:
            return None

        task_payload = task_snapshot.to_dict()
        task_key = json.dumps(task_payload, sort_keys=True)
        task_ref_id = task_refs_by_key.get(task_key)
        if task_ref_id is not None:
            return task_ref_id

        task_ref_id = f"task_{len(exported_tasks) + 1:03d}"
        task_refs_by_key[task_key] = task_ref_id
        exported_tasks.append(
            {
                "task_ref_id": task_ref_id,
                **task_payload,
            }
        )
        return task_ref_id

    def _run_config_payload(self, run: TrainingRun | None) -> dict[str, object] | None:
        if run is None:
            return None
        run_config = run.metadata.get("run_config")
        return dict(run_config) if isinstance(run_config, dict) else None

    def _latest_checkpoint(self) -> Checkpoint | None:
        if not self._snapshot.checkpoints:
            return None
        return self._snapshot.checkpoints[-1]

    def _set_root_details(self, node: _LineageNode) -> None:
        self.checkpoint_details.setHtml(
            self._details_panel_html(
                heading="Selected training origin",
                rows=[
                    ("Node type", "Untrained agent root"),
                    ("Environment", node.environment_id or "unknown"),
                    ("Algorithm", node.algorithm or "unknown"),
                ],
                note="Training will start from scratch with no checkpoint restoration.",
            )
        )

    def _set_training_run_details(self, edge: _LineageEdge) -> None:
        checkpoint = edge.target_checkpoint
        task_snapshot = edge.task_snapshot or checkpoint.task_snapshot
        run = edge.run
        run_config = self._run_config_payload(run) or {}
        task_name = task_snapshot.task_name if task_snapshot is not None else (checkpoint.task_name or "unknown")
        setup_rows = [
            ("Task name", task_name),
            ("Recorded episodes", str(len(edge.episodes))),
            (
                "Max steps / episode",
                self._format_optional_value(run_config.get("max_steps_per_episode"), empty="no limit"),
            ),
            ("Episodes", self._format_optional_value(run_config.get("max_episodes"), empty="no limit")),
        ]

        metrics = checkpoint.metadata.get("training_metrics")
        if (not isinstance(metrics, dict) or not metrics) and run is not None:
            metrics = run.metadata.get("latest_metrics")

        self.checkpoint_details.setHtml(
            self._summary_panel_html(
                heading="Selected training run",
                columns=[
                    ("Training setup", setup_rows),
                    (
                        "Training results",
                        self._summary_metric_rows(
                            metrics,
                            fallback_episode=checkpoint.episode,
                            include_cumulative_reward=True,
                        ),
                    ),
                ],
            )
        )

    def _set_checkpoint_comparison_details(self, nodes: list[_LineageNode]) -> None:
        columns: list[tuple[str, list[tuple[str, str]]]] = []
        for node in nodes:
            checkpoint = node.checkpoint
            if checkpoint is None:
                continue
            columns.append(
                (
                    self._checkpoint_comparison_title(checkpoint),
                    self._checkpoint_result_rows(checkpoint),
                )
            )

        self.checkpoint_details.setHtml(
            self._comparison_panel_html(
                heading="Selected checkpoints",
                columns=columns,
            )
        )

    def _set_checkpoint_details(self, checkpoint: Checkpoint, *, heading: str) -> None:
        result_heading, rows = self._checkpoint_result_summary(checkpoint)
        self.checkpoint_details.setHtml(
            self._summary_panel_html(
                heading=heading,
                columns=[
                    (result_heading, rows),
                ],
            )
        )

    def _checkpoint_result_summary(self, checkpoint: Checkpoint) -> tuple[str, list[tuple[str, str]]]:
        evaluation = checkpoint.metadata.get("evaluation")
        metrics = checkpoint.metadata.get("evaluation_metrics")
        result_heading = "Evaluation results"
        fallback_episode = evaluation.get("episode_count") if isinstance(evaluation, dict) else checkpoint.episode
        if not isinstance(metrics, dict) or not metrics:
            metrics = checkpoint.metadata.get("training_metrics")
            result_heading = "Training results"
            fallback_episode = checkpoint.episode
        return result_heading, self._summary_metric_rows(metrics, fallback_episode=fallback_episode)

    def _checkpoint_result_rows(self, checkpoint: Checkpoint) -> list[tuple[str, str]]:
        _heading, rows = self._checkpoint_result_summary(checkpoint)
        return rows

    def _checkpoint_comparison_title(self, checkpoint: Checkpoint) -> str:
        label = checkpoint.label.strip() if checkpoint.label else checkpoint.checkpoint_id
        if len(label) > 24:
            return label[:21] + "..."
        return label

    def _details_panel_html(
        self,
        *,
        heading: str,
        rows: list[tuple[str, str]],
        note: str | None = None,
        metrics_heading: str | None = None,
        metric_rows: list[tuple[str, str]] | None = None,
        metrics_empty_message: str | None = None,
        empty_message: str | None = None,
    ) -> str:
        parts = [
            "<div style='font-family: Sans-Serif; color: #0f172a;'>",
            f"<div style='font-weight: 700; margin-bottom: 8px;'>{escape(heading)}</div>",
        ]

        if rows:
            parts.append(self._table_html(rows))
        elif empty_message is not None:
            parts.append(f"<div style='color: #64748b;'>{escape(empty_message)}</div>")

        if note is not None:
            parts.append(
                f"<div style='margin-top: 10px; color: #475569;'>{escape(note)}</div>"
            )

        if metrics_heading is not None:
            parts.append(
                f"<div style='font-weight: 700; margin: 12px 0 8px 0;'>{escape(metrics_heading)}</div>"
            )
            if metric_rows:
                parts.append(self._table_html(metric_rows))
            elif metrics_empty_message is not None:
                parts.append(f"<div style='color: #64748b;'>{escape(metrics_empty_message)}</div>")

        parts.append("</div>")
        return "".join(parts)

    def _table_html(self, rows: list[tuple[str, str]]) -> str:
        table_rows = []
        for label, value in rows:
            table_rows.append(
                "<tr>"
                f"<td style='padding: 6px 10px; border: 1px solid #dbe4f0; background: #f8fafc; "
                f"font-weight: 600; color: #334155; width: 42%;'>{escape(label)}</td>"
                f"<td style='padding: 6px 10px; border: 1px solid #dbe4f0; color: #0f172a;'>{escape(value)}</td>"
                "</tr>"
            )
        return (
            "<table cellspacing='0' cellpadding='0' "
            "style='width: 100%; border-collapse: collapse; background: #ffffff;'>"
            f"{''.join(table_rows)}"
            "</table>"
        )

    def _summary_panel_html(
        self,
        *,
        heading: str,
        columns: list[tuple[str, list[tuple[str, str]]]],
    ) -> str:
        parts = [
            "<div style='font-family: Sans-Serif; color: #0f172a;'>",
            f"<div style='font-weight: 700; margin-bottom: 8px;'>{escape(heading)}</div>",
        ]

        if len(columns) == 1:
            title, rows = columns[0]
            parts.append(self._summary_column_html(title, rows))
        else:
            cells = []
            width = 100.0 / max(len(columns), 1)
            for title, rows in columns:
                cells.append(
                    "<td style='vertical-align: top; padding-right: 8px; "
                    f"width: {width:.0f}%;'>"
                    f"{self._summary_column_html(title, rows)}"
                    "</td>"
                )
            parts.append(
                "<table cellspacing='0' cellpadding='0' style='width: 100%; border-collapse: collapse;'>"
                f"<tr>{''.join(cells)}</tr>"
                "</table>"
            )

        parts.append("</div>")
        return "".join(parts)

    def _summary_column_html(self, title: str, rows: list[tuple[str, str]]) -> str:
        return (
            f"<div style='font-weight: 700; margin-bottom: 6px;'>{escape(title)}</div>"
            f"{self._table_html(rows)}"
        )

    def _comparison_panel_html(
        self,
        *,
        heading: str,
        columns: list[tuple[str, list[tuple[str, str]]]],
    ) -> str:
        return (
            "<div style='font-family: Sans-Serif; color: #0f172a;'>"
            f"<div style='font-weight: 700; margin-bottom: 8px;'>{escape(heading)}</div>"
            f"{self._comparison_table_html(columns)}"
            "</div>"
        )

    def _comparison_table_html(self, columns: list[tuple[str, list[tuple[str, str]]]]) -> str:
        if not columns:
            return "<div style='color: #64748b;'>No checkpoint metrics are available.</div>"

        labels = ["Episode", "Success rate", "Mean reward", "Episode length"]
        header_cells = [
            "<td style='padding: 6px 10px; border: 1px solid #dbe4f0; "
            "background: #f8fafc; font-weight: 700; color: #334155;'>Metric</td>"
        ]
        for title, _rows in columns:
            header_cells.append(
                "<td style='padding: 6px 10px; border: 1px solid #dbe4f0; "
                "background: #f8fafc; font-weight: 700; color: #334155;'>"
                f"{escape(title)}</td>"
            )

        table_rows = [f"<tr>{''.join(header_cells)}</tr>"]
        row_maps = [dict(rows) for _title, rows in columns]
        for label in labels:
            cells = [
                "<td style='padding: 6px 10px; border: 1px solid #dbe4f0; "
                "background: #f8fafc; font-weight: 600; color: #334155;'>"
                f"{escape(label)}</td>"
            ]
            for row_map in row_maps:
                cells.append(
                    "<td style='padding: 6px 10px; border: 1px solid #dbe4f0; color: #0f172a;'>"
                    f"{escape(row_map.get(label, '--'))}</td>"
                )
            table_rows.append(f"<tr>{''.join(cells)}</tr>")

        return (
            "<table cellspacing='0' cellpadding='0' "
            "style='width: 100%; border-collapse: collapse; background: #ffffff;'>"
            f"{''.join(table_rows)}"
            "</table>"
        )

    def _summary_metric_rows(
        self,
        metrics: object,
        *,
        fallback_episode: object = None,
        include_cumulative_reward: bool = False,
    ) -> list[tuple[str, str]]:
        payload = metrics if isinstance(metrics, dict) else {}
        episode = payload.get("episode")
        if episode is None:
            episode = fallback_episode
        rows = [
            ("Episode", self._format_optional_value(episode)),
            ("Success rate", self._format_percent_metric(payload.get("success_rate"))),
            (
                "Mean reward",
                self._format_decimal_metric(payload.get("mean_reward", payload.get("episode_reward_mean"))),
            ),
            ("Episode length", self._format_decimal_metric(payload.get("episode_length_mean"))),
        ]
        if include_cumulative_reward:
            rows.append(("Cumulative reward", self._format_decimal_metric(payload.get("cumulative_reward"))))
        return rows

    def _format_optional_value(self, value: object, *, empty: str = "unknown") -> str:
        if value is None or value == "":
            return empty
        return str(value)

    def _format_decimal_metric(self, value: object) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    def _format_percent_metric(self, value: object) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value) * 100.0:.1f}%"
        except (TypeError, ValueError):
            return str(value)
