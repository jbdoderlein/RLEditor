from __future__ import annotations

from collections import deque
from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rleditor.core.models import Breakpoint, RunConfig, TrainingMetrics, TrainingStatus


BREAKPOINT_RULES: list[tuple[str, str, bool, float, float, int, float]] = [
    ("max_step", "Max Step >=", False, 1.0, 50_000_000.0, 0, 100.0),
    ("episode_count_gte", "Episode Count >=", False, 1.0, 1_000_000.0, 0, 1.0),
    ("mean_reward_gte", "Mean Reward >=", False, -5.0, 5.0, 3, 0.01),
    ("episode_reward_mean_gte", "Episode Reward Mean >=", False, -5.0, 5.0, 3, 0.01),
    ("success_rate_gte", "Success Rate >=", True, 0.0, 100.0, 2, 1.0),
]


def _breakpoint_spec(rule_kind: str) -> tuple[str, str, bool, float, float, int, float]:
    for spec in BREAKPOINT_RULES:
        if spec[0] == rule_kind:
            return spec
    return BREAKPOINT_RULES[0]


def _format_scalar(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _format_percent(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value * 100.0:.{digits}f}%"


METRIC_SPECS: list[tuple[str, str, str]] = [
    ("episode_reward_mean", "Episode Return (Mean)", "#0a9396"),
    ("success_rate", "Success Rate", "#386641"),
    ("episode_length_mean", "Episode Length (Mean)", "#bc6c25"),
    ("exploration_rate", "Exploration Rate", "#7c3aed"),
    ("value_loss", "TD Error", "#b45309"),
    ("fps", "FPS", "#577590"),
]


class SparklineWidget(QWidget):
    def __init__(
        self,
        *,
        color: str,
        max_points: int = 160,
    ) -> None:
        super().__init__()
        self._values: deque[float] = deque(maxlen=max_points)
        self._color = QColor(color)
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def add_point(self, value: float) -> None:
        self._values.append(float(value))
        self.update()

    def clear(self) -> None:
        self._values.clear()
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, QColor("#f9fbff"))

        if len(self._values) < 2:
            painter.setPen(QPen(QColor("#d0d7e4"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(rect.left(), rect.center().y(), rect.right(), rect.center().y())
            return

        min_value = min(self._values)
        max_value = max(self._values)
        span = max(max_value - min_value, 1e-6)

        points: list[tuple[float, float]] = []
        width = max(1, rect.width())
        height = max(1, rect.height())
        count = len(self._values)
        for idx, value in enumerate(self._values):
            x = rect.left() + (idx / (count - 1)) * width
            y_ratio = (value - min_value) / span
            y = rect.bottom() - y_ratio * height
            points.append((x, y))

        path = QPainterPath()
        first_x, first_y = points[0]
        path.moveTo(first_x, first_y)
        for x, y in points[1:]:
            path.lineTo(x, y)

        painter.setPen(QPen(self._color, 2.0))
        painter.drawPath(path)


class MetricCard(QFrame):
    def __init__(
        self,
        *,
        title: str,
        color: str,
    ) -> None:
        super().__init__()
        self.setObjectName("MetricCardFrame")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricTitleLabel")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("MetricValueLabel")
        self.sparkline = SparklineWidget(color=color)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.sparkline, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_value_text(self, text: str) -> None:
        self.value_label.setText(text)

    def add_point(self, value: float) -> None:
        self.sparkline.add_point(value)

    def clear(self) -> None:
        self.value_label.setText("--")
        self.sparkline.clear()


class RunMetricPanel(QGroupBox):
    def __init__(
        self,
        *,
        title: str,
        formatters: dict[str, Callable[[float | None], str]],
    ) -> None:
        super().__init__(title)
        self.metric_cards: dict[str, MetricCard] = {}
        self._formatters = formatters
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        grid = QGridLayout(self)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for column in range(3):
            grid.setColumnStretch(column, 1)
        for row in range(2):
            grid.setRowStretch(row, 1)

        for index, (key, card_title, color) in enumerate(METRIC_SPECS):
            card = MetricCard(title=card_title, color=color)
            row = index // 3
            col = index % 3
            grid.addWidget(card, row, col)
            self.metric_cards[key] = card

    def set_metrics(self, metrics: TrainingMetrics) -> None:
        self._update_metric_card("episode_reward_mean", metrics.episode_reward_mean)
        self._update_metric_card("success_rate", metrics.success_rate)
        self._update_metric_card("episode_length_mean", metrics.episode_length_mean)
        self._update_metric_card("exploration_rate", metrics.exploration_rate)
        self._update_metric_card("value_loss", metrics.value_loss)
        self._update_metric_card("fps", metrics.fps)

    def clear(self) -> None:
        for card in self.metric_cards.values():
            card.clear()

    def _update_metric_card(self, key: str, value: float | None) -> None:
        card = self.metric_cards[key]
        card.set_value_text(self._formatters[key](value))
        if value is not None:
            card.add_point(value)


class TrainingMonitorView(QWidget):
    start_requested = Signal(object)
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._metric_cards: dict[str, MetricCard] = {}
        self._run_metric_panels: dict[str, RunMetricPanel] = {}
        self._breakpoint_rules: list[Breakpoint] = []
        self._formatters: dict[str, Callable[[float | None], str]] = {
            "episode_reward_mean": lambda value: _format_scalar(value),
            "success_rate": lambda value: _format_percent(value),
            "episode_length_mean": lambda value: _format_scalar(value, digits=2),
            "exploration_rate": lambda value: _format_percent(value),
            "value_loss": lambda value: _format_scalar(value),
            "fps": lambda value: "--" if value is None else f"{value:.1f}",
        }

        root = QVBoxLayout(self)

        title = QLabel("Training Monitor")
        title.setObjectName("TitleLabel")
        subtitle = QLabel("Track return, success, exploration, TD error, and throughput during training.")
        subtitle.setObjectName("SubtitleLabel")
        root.addWidget(title)
        root.addWidget(subtitle)

        config_group = QGroupBox("Run Config")
        config_form = QFormLayout(config_group)

        self.total_steps_spin = QSpinBox(config_group)
        self.total_steps_spin.setRange(-1, 50_000_000)
        self.total_steps_spin.setSpecialValueText("No limit")
        self.total_steps_spin.setToolTip("Set to -1 to train until stopped, paused by breakpoint, or another limit is reached.")
        self.total_steps_spin.setValue(100_000)

        self.max_steps_per_episode_spin = QSpinBox(config_group)
        self.max_steps_per_episode_spin.setRange(0, 100_000)
        self.max_steps_per_episode_spin.setSpecialValueText("No limit")
        self.max_steps_per_episode_spin.setValue(0)

        self.trace_sample_rate_spin = QDoubleSpinBox(config_group)
        self.trace_sample_rate_spin.setRange(0.0, 100.0)
        self.trace_sample_rate_spin.setDecimals(1)
        self.trace_sample_rate_spin.setSingleStep(5.0)
        self.trace_sample_rate_spin.setSuffix(" %")
        self.trace_sample_rate_spin.setValue(100.0)

        config_form.addRow("Max steps", self.total_steps_spin)
        config_form.addRow("Max steps / episode", self.max_steps_per_episode_spin)
        config_form.addRow("Recorded episodes", self.trace_sample_rate_spin)

        root.addWidget(config_group)
        root.addWidget(self._build_breakpoints_group())

        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")

        self.start_btn.clicked.connect(self._emit_start)
        self.pause_btn.clicked.connect(self.pause_requested.emit)
        self.resume_btn.clicked.connect(self.resume_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        controls.addWidget(self.start_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.resume_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        root.addLayout(controls)

        root.addWidget(self._build_metrics_group(), 1)

        self.status_label = QLabel("Status: idle")
        self.metrics_label = QLabel(
            "steps=0 | episodes=0 | return_mean=0.000 | success=0.0% | epsilon=0.0% | td_error=--"
        )
        self.breakpoint_label = QLabel("Breakpoint: -")

        root.addWidget(self.status_label)
        root.addWidget(self.metrics_label)
        root.addWidget(self.breakpoint_label)

    def build_config(self) -> RunConfig:
        return RunConfig(
            algorithm="q_learning",
            episode_trace_sample_rate=self.trace_sample_rate_spin.value() / 100.0,
            max_steps=(
                self.total_steps_spin.value()
                if self.total_steps_spin.value() > 0
                else None
            ),
            max_steps_per_episode=(
                self.max_steps_per_episode_spin.value()
                if self.max_steps_per_episode_spin.value() > 0
                else None
            ),
            breakpoints=[
                Breakpoint(
                    kind=rule.kind,
                    value=rule.value,
                    window=rule.window,
                    actions=list(rule.actions),
                )
                for rule in self._breakpoint_rules
            ],
        )

    def set_status(self, status: TrainingStatus) -> None:
        self.status_label.setText(f"Status: {status.value}")

    def set_metrics(self, metrics: TrainingMetrics) -> None:
        self.metrics_label.setText(
            "steps="
            f"{metrics.step} | episodes={metrics.episode} | "
            f"return_mean={metrics.episode_reward_mean:.3f} | "
            f"success={_format_percent(metrics.success_rate)} | "
            f"epsilon={_format_percent(metrics.exploration_rate)} | "
            f"td_error={_format_scalar(metrics.value_loss)}"
        )

        if not self._run_metric_panels:
            self._aggregate_metric_panel.set_metrics(metrics)

    def set_run_metrics(self, run_id: str, task_name: str, metrics: TrainingMetrics) -> None:
        panel = self._run_metric_panels.get(run_id)
        if panel is None:
            self._ensure_run_panel_mode()
            panel = RunMetricPanel(
                title=f"{task_name} | {run_id}",
                formatters=self._formatters,
            )
            self._run_metric_panels[run_id] = panel
            self.metrics_splitter.addWidget(panel)
            self.metrics_splitter.setStretchFactor(self.metrics_splitter.indexOf(panel), 1)
            self._balance_metric_panel_sizes()
        panel.set_metrics(metrics)

    def set_breakpoint_event(self, message: str) -> None:
        self.breakpoint_label.setText(f"Breakpoint: {message}")

    def _emit_start(self) -> None:
        self._reset_metric_cards()
        self.breakpoint_label.setText("Breakpoint: -")
        self.start_requested.emit(self.build_config())

    def _build_breakpoints_group(self) -> QGroupBox:
        group = QGroupBox("Training Breakpoints")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.breakpoint_kind_combo = QComboBox(group)
        for kind, label, _pct, _min_v, _max_v, _decimals, _step in BREAKPOINT_RULES:
            self.breakpoint_kind_combo.addItem(label, kind)

        self.breakpoint_value_spin = QDoubleSpinBox(group)
        self.breakpoint_value_spin.setDecimals(3)
        self.breakpoint_value_spin.setRange(-1000.0, 1000.0)

        self.breakpoint_window_spin = QSpinBox(group)
        self.breakpoint_window_spin.setRange(0, 5000)
        self.breakpoint_window_spin.setValue(0)
        self.breakpoint_window_spin.setToolTip("Optional moving window in steps (0 = disabled)")

        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove Selected")
        clear_btn = QPushButton("Clear")

        add_btn.clicked.connect(self._add_breakpoint_rule)
        remove_btn.clicked.connect(self._remove_selected_breakpoint)
        clear_btn.clicked.connect(self._clear_breakpoint_rules)
        self.breakpoint_kind_combo.currentIndexChanged.connect(self._sync_breakpoint_inputs)

        controls.addWidget(QLabel("Rule"))
        controls.addWidget(self.breakpoint_kind_combo, 2)
        controls.addWidget(QLabel("Value"))
        controls.addWidget(self.breakpoint_value_spin, 1)
        controls.addWidget(QLabel("Window"))
        controls.addWidget(self.breakpoint_window_spin, 1)
        controls.addWidget(add_btn)
        controls.addWidget(remove_btn)
        controls.addWidget(clear_btn)

        self.breakpoint_list = QListWidget(group)
        self.breakpoint_list.setMinimumHeight(92)

        layout.addLayout(controls)
        layout.addWidget(self.breakpoint_list)

        self._sync_breakpoint_inputs()
        return group

    def _build_metrics_group(self) -> QGroupBox:
        group = QGroupBox("Live Scalars")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(group)

        self.metrics_splitter = QSplitter(Qt.Orientation.Vertical, group)
        self.metrics_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.metrics_splitter, 1)

        self._aggregate_metric_panel = RunMetricPanel(
            title="Aggregate training metrics",
            formatters=self._formatters,
        )
        self._metric_cards = self._aggregate_metric_panel.metric_cards
        self.metrics_splitter.addWidget(self._aggregate_metric_panel)
        self.metrics_splitter.setStretchFactor(0, 1)

        return group

    def _reset_metric_cards(self) -> None:
        self.metrics_label.setText(
            "steps=0 | episodes=0 | return_mean=0.000 | success=0.0% | epsilon=0.0% | td_error=--"
        )
        for panel in self._run_metric_panels.values():
            panel.setParent(None)
            panel.deleteLater()
        self._run_metric_panels.clear()
        self._ensure_aggregate_panel_visible()
        self._aggregate_metric_panel.clear()

    def _ensure_run_panel_mode(self) -> None:
        if self.metrics_splitter.indexOf(self._aggregate_metric_panel) >= 0:
            self._aggregate_metric_panel.setParent(None)

    def _ensure_aggregate_panel_visible(self) -> None:
        if self.metrics_splitter.indexOf(self._aggregate_metric_panel) >= 0:
            return
        self.metrics_splitter.addWidget(self._aggregate_metric_panel)
        self.metrics_splitter.setStretchFactor(self.metrics_splitter.indexOf(self._aggregate_metric_panel), 1)

    def _balance_metric_panel_sizes(self) -> None:
        count = self.metrics_splitter.count()
        if count <= 0:
            return
        available_height = max(self.metrics_splitter.height(), 220 * count)
        self.metrics_splitter.setSizes([available_height // count for _ in range(count)])

    def _sync_breakpoint_inputs(self) -> None:
        kind = self.breakpoint_kind_combo.currentData()
        spec = _breakpoint_spec(str(kind) if kind else "max_step")
        _kind, _label, is_percentage, min_value, max_value, decimals, step = spec

        self.breakpoint_value_spin.setSuffix(" %" if is_percentage else "")
        self.breakpoint_value_spin.setDecimals(decimals)
        self.breakpoint_value_spin.setRange(min_value, max_value)
        self.breakpoint_value_spin.setSingleStep(step)
        default_value = min_value if min_value > 0 else 0.0
        self.breakpoint_value_spin.setValue(default_value)

    def _add_breakpoint_rule(self) -> None:
        kind_data = self.breakpoint_kind_combo.currentData()
        if not kind_data:
            return

        kind = str(kind_data)
        _kind, label, is_percentage, _min_v, _max_v, _decimals, _step = _breakpoint_spec(kind)
        raw_value = self.breakpoint_value_spin.value()
        value = raw_value / 100.0 if is_percentage else raw_value
        window = self.breakpoint_window_spin.value()

        self._breakpoint_rules.append(
            Breakpoint(
                kind=kind,
                value=value,
                window=window if window > 0 else None,
                actions=["pause", "checkpoint"],
            )
        )
        self._refresh_breakpoint_list()
        self.breakpoint_label.setText(f"Breakpoint: added {label} {raw_value:g} (pause + checkpoint)")

    def _remove_selected_breakpoint(self) -> None:
        row = self.breakpoint_list.currentRow()
        if row < 0 or row >= len(self._breakpoint_rules):
            return
        self._breakpoint_rules.pop(row)
        self._refresh_breakpoint_list()

    def _clear_breakpoint_rules(self) -> None:
        self._breakpoint_rules.clear()
        self._refresh_breakpoint_list()

    def _refresh_breakpoint_list(self) -> None:
        self.breakpoint_list.clear()
        for rule in self._breakpoint_rules:
            _kind, label, is_percentage, _min_v, _max_v, _decimals, _step = _breakpoint_spec(rule.kind)
            displayed_value = rule.value * 100.0 if is_percentage else rule.value
            window_text = f", window={rule.window}" if rule.window else ""
            suffix = "%" if is_percentage else ""
            actions_text = f", actions={'+'.join(rule.actions)}" if rule.actions else ""
            self.breakpoint_list.addItem(
                f"{label} {displayed_value:g}{suffix}{window_text}{actions_text}"
            )
