from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from rleditor.core.models import TaskDefinition
from rleditor.plugins.base import EnvironmentPlugin
from rleditor.ui.views.task_editor_view import TaskEditorView


class _DummyGuiExtension:
    def create_task_editor_widget(self, task: TaskDefinition, on_task_changed):
        _ = task, on_task_changed
        return QWidget()

    def create_episode_replay_widget(self, parent: QWidget | None = None):
        _ = parent
        return None


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_task_editor_view_allows_editing_task_name() -> None:
    _app()
    task = TaskDefinition(environment_id="dummy_env", name="Original Name", task_id="task_1")
    plugin = EnvironmentPlugin(
        plugin_id="dummy",
        display_name="Dummy",
        description="Test plugin",
        backend=object(),
        gui_extension=_DummyGuiExtension(),
    )
    view = TaskEditorView()
    changed: list[TaskDefinition] = []
    view.task_changed.connect(changed.append)

    view.set_plugin_task(plugin, task)
    view.name_input.setText("Renamed Task")
    view._on_name_edited()

    assert task.name == "Renamed Task"
    assert changed[-1].name == "Renamed Task"


def test_task_editor_view_rejects_empty_task_name() -> None:
    _app()
    task = TaskDefinition(environment_id="dummy_env", name="Original Name", task_id="task_1")
    plugin = EnvironmentPlugin(
        plugin_id="dummy",
        display_name="Dummy",
        description="Test plugin",
        backend=object(),
        gui_extension=_DummyGuiExtension(),
    )
    view = TaskEditorView()

    view.set_plugin_task(plugin, task)
    view.name_input.setText("   ")
    view._on_name_edited()

    assert task.name == "Original Name"
    assert view.name_input.text() == "Original Name"
