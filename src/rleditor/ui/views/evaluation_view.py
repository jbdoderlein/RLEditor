from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rleditor.core.models import TaskDefinition


class EvaluationView(QWidget):
    """Configuration for checkpoint evaluation runs."""

    import_task_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._tasks: list[TaskDefinition] = []

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

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addLayout(form)
        root.addStretch(1)

    def set_tasks(self, tasks: list[TaskDefinition]) -> None:
        selected_key = self._selected_task_key()
        self._tasks = [deepcopy(task) for task in tasks]
        self.task_combo.clear()

        selected_index = 0
        for index, task in enumerate(self._tasks):
            if selected_key is not None and self._task_key(task) == selected_key:
                selected_index = index
            label = f"{task.name} ({task.task_id or 'no id'})"
            self.task_combo.addItem(label, index)

        if self._tasks:
            self.task_combo.setCurrentIndex(min(selected_index, len(self._tasks) - 1))

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

    def _task_key(self, task: TaskDefinition) -> tuple[str | None, str]:
        return task.task_id, task.name
