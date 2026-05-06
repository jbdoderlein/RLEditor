from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QLineEdit, QVBoxLayout, QWidget

from rleditor.core.models import TaskDefinition
from rleditor.plugins.base import EnvironmentPlugin


class TaskEditorView(QWidget):
    task_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._active_editor: QWidget | None = None
        self._task: TaskDefinition | None = None
        self._loading_task = False

        self.root_layout = QVBoxLayout(self)
        self.title = QLabel("Task Editor")
        self.title.setObjectName("TitleLabel")
        self.helper = QLabel(
            "This panel is delegated to the selected environment plugin."
        )
        self.helper.setObjectName("SubtitleLabel")
        self.name_group = QGroupBox("Task Identity", self)
        self.name_layout = QFormLayout(self.name_group)
        self.name_input = QLineEdit(self.name_group)
        self.name_input.setPlaceholderText("Task name")
        self.name_input.editingFinished.connect(self._on_name_edited)
        self.name_layout.addRow("Name", self.name_input)
        self.root_layout.addWidget(self.title)
        self.root_layout.addWidget(self.helper)
        self.root_layout.addWidget(self.name_group)

    def set_plugin_task(self, plugin: EnvironmentPlugin, task: TaskDefinition) -> None:
        self._task = task
        self.name_input.blockSignals(True)
        self.name_input.setText(task.name)
        self.name_input.blockSignals(False)
        self._clear_editor()

        if plugin.gui_extension is None:
            placeholder = QLabel(
                "No environment-specific editor registered. "
                "Install or implement a GUI extension for this plugin."
            )
            self._active_editor = placeholder
            self.root_layout.addWidget(placeholder)
            return

        self._loading_task = True
        try:
            self._active_editor = plugin.gui_extension.create_task_editor_widget(
                task=task,
                on_task_changed=self._on_task_changed,
            )
        finally:
            self._loading_task = False
        self.root_layout.addWidget(self._active_editor)
        self.root_layout.addStretch(1)

    def _clear_editor(self) -> None:
        while self.root_layout.count() > 3:
            item = self.root_layout.takeAt(3)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _on_name_edited(self) -> None:
        if self._task is None:
            return

        candidate_name = self.name_input.text().strip()
        if not candidate_name:
            self.name_input.blockSignals(True)
            self.name_input.setText(self._task.name)
            self.name_input.blockSignals(False)
            return

        if candidate_name == self._task.name:
            return

        self._task.name = candidate_name
        self.task_changed.emit(self._task)

    def _on_task_changed(self, task: TaskDefinition) -> None:
        self._task = task
        if self._loading_task:
            return
        self.task_changed.emit(task)
