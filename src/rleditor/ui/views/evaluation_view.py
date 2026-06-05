from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rleditor.core.models import TaskDefinition


EVALUATION_RESULT_COLUMNS = [
    ("task_name", "Task"),
    ("environment_id", "Environment"),
    ("episode_count", "Episodes"),
    ("success_rate", "Success Rate"),
    ("mean_reward", "Mean Reward"),
    ("cumulative_reward", "Cumulative Reward"),
    ("episode_length_mean", "Episode Length"),
    ("step", "Steps"),
    ("error", "Error"),
]


class EvaluationResultsDialog(QDialog):
    def __init__(self, rows: list[dict[str, object]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Multiple Evaluation Results")
        self.resize(860, 420)
        self.rows = list(rows)

        root = QVBoxLayout(self)
        self.table = QTableWidget(len(self.rows), len(EVALUATION_RESULT_COLUMNS), self)
        self.table.setHorizontalHeaderLabels([label for _key, label in EVALUATION_RESULT_COLUMNS])
        for row_index, row in enumerate(self.rows):
            for column_index, (key, _label) in enumerate(EVALUATION_RESULT_COLUMNS):
                self.table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(self._display_value(key, row.get(key))),
                )
        self.table.resizeColumnsToContents()
        root.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.save_button = buttons.addButton("Save", QDialogButtonBox.ButtonRole.ActionRole)
        self.save_button.clicked.connect(self._save_csv_requested)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _save_csv_requested(self) -> None:
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Evaluation Results",
            "evaluation_results.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not selected_path:
            return
        path = Path(selected_path).expanduser()
        if path.suffix == "":
            path = path.with_suffix(".csv")
        self.save_csv(path)

    def save_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([label for _key, label in EVALUATION_RESULT_COLUMNS])
            for row in self.rows:
                writer.writerow(
                    [
                        self._display_value(key, row.get(key))
                        for key, _label in EVALUATION_RESULT_COLUMNS
                    ]
                )

    def _display_value(self, key: str, value: object) -> str:
        if value is None:
            return ""
        if key == "success_rate":
            try:
                return f"{float(value) * 100.0:.1f}%"
            except (TypeError, ValueError):
                return str(value)
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)


class EvaluationView(QWidget):
    """Configuration for checkpoint evaluation runs."""

    import_task_requested = Signal()
    evaluate_multiple_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._tasks: list[TaskDefinition] = []
        self._multi_task_checkboxes: list[tuple[QCheckBox, int]] = []
        self._multi_selection_initialized = False

        root = QVBoxLayout(self)
        title = QLabel("Evaluation", self)
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Choose the task used to evaluate each checkpoint. "
            "Evaluation does not learn and records every episode trace.",
            self,
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)

        form = QFormLayout()
        self.task_combo = QComboBox(self)
        self.import_task_button = QPushButton("Import Task", self)
        self.import_task_button.setToolTip("Import a task JSON or a generated curriculum task file.")
        self.import_task_button.clicked.connect(self.import_task_requested.emit)
        task_row = QHBoxLayout()
        task_row.addWidget(self.task_combo, 1)
        task_row.addWidget(self.import_task_button)
        self.episode_count_spin = QSpinBox(self)
        self.episode_count_spin.setRange(1, 100_000)
        self.episode_count_spin.setValue(10)
        self.max_steps_per_episode_spin = QSpinBox(self)
        self.max_steps_per_episode_spin.setRange(0, 10_000_000)
        self.max_steps_per_episode_spin.setSpecialValueText("No limit")
        self.max_steps_per_episode_spin.setValue(100)
        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setSpecialValueText("Use training seed")
        self.seed_spin.setValue(-1)

        form.addRow("Task", task_row)
        form.addRow("Episodes", self.episode_count_spin)
        form.addRow("Max steps / episode", self.max_steps_per_episode_spin)
        form.addRow("Seed", self.seed_spin)

        self.multi_eval_group = QGroupBox("Multiple Evaluation", self)
        multi_eval_layout = QVBoxLayout(self.multi_eval_group)
        self.multi_task_list_widget = QWidget(self.multi_eval_group)
        self.multi_task_list_layout = QVBoxLayout(self.multi_task_list_widget)
        self.multi_task_list_layout.setContentsMargins(0, 0, 0, 0)
        self.multi_task_scroll = QScrollArea(self.multi_eval_group)
        self.multi_task_scroll.setWidgetResizable(True)
        self.multi_task_scroll.setMinimumHeight(120)
        self.multi_task_scroll.setWidget(self.multi_task_list_widget)
        self.evaluate_multiple_button = QPushButton("Evaluate Multiple", self.multi_eval_group)
        self.evaluate_multiple_button.setEnabled(False)
        self.evaluate_multiple_button.clicked.connect(self.evaluate_multiple_requested.emit)
        multi_action_row = QHBoxLayout()
        multi_action_row.addWidget(self.evaluate_multiple_button)
        multi_action_row.addStretch(1)
        multi_eval_layout.addWidget(self.multi_task_scroll)
        multi_eval_layout.addLayout(multi_action_row)

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addLayout(form)
        root.addWidget(self.multi_eval_group)
        root.addStretch(1)

    def set_tasks(self, tasks: list[TaskDefinition]) -> None:
        selected_key = self._selected_task_key()
        checked_keys = self._selected_multi_task_keys()
        had_multi_selection = self._multi_selection_initialized
        self._tasks = [deepcopy(task) for task in tasks]
        self.task_combo.clear()
        self._clear_multi_task_checkboxes()

        selected_index = 0
        for index, task in enumerate(self._tasks):
            if selected_key is not None and self._task_key(task) == selected_key:
                selected_index = index
            label = f"{task.name} ({task.task_id or 'no id'})"
            self.task_combo.addItem(label, index)
            checkbox = QCheckBox(label, self.multi_task_list_widget)
            task_key = self._task_key(task)
            if had_multi_selection:
                checked = task_key in checked_keys
            elif selected_key is None:
                checked = index == 0
            else:
                checked = task_key == selected_key
            checkbox.setChecked(checked)
            checkbox.stateChanged.connect(self._update_evaluate_multiple_enabled)
            self.multi_task_list_layout.addWidget(checkbox)
            self._multi_task_checkboxes.append((checkbox, index))

        if self._tasks:
            self.task_combo.setCurrentIndex(min(selected_index, len(self._tasks) - 1))
        self.multi_task_list_layout.addStretch(1)
        self._multi_selection_initialized = True
        self._update_evaluate_multiple_enabled()

    def set_selected_task_index(self, index: int) -> None:
        if 0 <= index < len(self._tasks):
            self.task_combo.setCurrentIndex(index)

    def selected_task(self) -> TaskDefinition | None:
        index = self.task_combo.currentData()
        if index is None:
            return None
        try:
            task_index = int(index)
        except (TypeError, ValueError):
            return None
        if task_index < 0 or task_index >= len(self._tasks):
            return None
        return deepcopy(self._tasks[task_index])

    def build_evaluation_policy(self) -> dict[str, object]:
        task = self.selected_task()
        if task is None:
            return {}

        return self._evaluation_policy_for_task(task)

    def selected_evaluation_tasks(self) -> list[TaskDefinition]:
        selected_tasks: list[TaskDefinition] = []
        for checkbox, task_index in self._multi_task_checkboxes:
            if not checkbox.isChecked():
                continue
            if 0 <= task_index < len(self._tasks):
                selected_tasks.append(deepcopy(self._tasks[task_index]))
        return selected_tasks

    def build_multiple_evaluation_policies(self) -> list[dict[str, object]]:
        return [
            self._evaluation_policy_for_task(task)
            for task in self.selected_evaluation_tasks()
        ]

    def _evaluation_policy_for_task(self, task: TaskDefinition) -> dict[str, object]:
        max_steps = self.max_steps_per_episode_spin.value()
        seed = self.seed_spin.value()
        return {
            "task": task.to_dict(),
            "episode_count": self.episode_count_spin.value(),
            "max_steps_per_episode": max_steps if max_steps > 0 else None,
            "seed": seed if seed >= 0 else None,
            "trace_sample_rate": 1.0,
        }

    def _selected_task_key(self) -> tuple[str | None, str] | None:
        task = self.selected_task()
        if task is None:
            return None
        return self._task_key(task)

    def _selected_multi_task_keys(self) -> set[tuple[str | None, str]]:
        keys: set[tuple[str | None, str]] = set()
        for checkbox, task_index in self._multi_task_checkboxes:
            if checkbox.isChecked() and 0 <= task_index < len(self._tasks):
                keys.add(self._task_key(self._tasks[task_index]))
        return keys

    def _task_key(self, task: TaskDefinition) -> tuple[str | None, str]:
        return task.task_id, task.name

    def _clear_multi_task_checkboxes(self) -> None:
        self._multi_task_checkboxes = []
        while self.multi_task_list_layout.count():
            item = self.multi_task_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _update_evaluate_multiple_enabled(self) -> None:
        self.evaluate_multiple_button.setEnabled(
            any(checkbox.isChecked() for checkbox, _task_index in self._multi_task_checkboxes)
        )
