from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from rleditor.application.persistence import ProjectStore
from rleditor.application.services import TaskService, TrainingService
from rleditor.core.models import DerivedTaskDefinition, RunConfig, TaskDefinition
from rleditor.plugins.base import EnvironmentPlugin
from rleditor.plugins.registry import PluginRegistry
from rleditor.ui.shell.main_window import MainWindow


class _DummyBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="dummy_env",
            name="Dummy Main Task",
            task_id="task_main",
            config={"difficulty": 1, "layout": "base"},
            reward_config={"goal": 1.0},
            termination_config={"max_steps": 20},
            metadata={"source": "default"},
        )

    def create_env(self, task: TaskDefinition):
        _ = task
        return object()


class _EmittingGuiExtension:
    def create_task_editor_widget(self, task, on_task_changed):
        task.metadata["editor_initialized"] = True
        on_task_changed(task)
        return QLabel("Editor")

    def create_episode_replay_widget(self, parent=None):
        _ = parent
        return None


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_add_new_task_clones_first_workspace_task_state() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="dummy",
            display_name="Dummy",
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=TrainingService(registry),
        initial_plugin_id="dummy",
    )

    base_task = window._task_workspace[0]
    base_task.name = "Edited Main Task"
    base_task.config["difficulty"] = 7
    base_task.reward_config["goal"] = 2.5
    base_task.termination_config["max_steps"] = 42
    base_task.metadata["tag"] = "edited"

    window._create_new_task()

    new_task = window._task_workspace[-1]

    assert new_task is not base_task
    assert new_task.name == "Edited Main Task 2"
    assert new_task.task_id is None
    assert new_task.config == base_task.config
    assert new_task.reward_config == base_task.reward_config
    assert new_task.termination_config == base_task.termination_config
    assert new_task.metadata == base_task.metadata

    new_task.config["difficulty"] = 99
    assert base_task.config["difficulty"] == 7


def test_task_history_selection_updates_current_task_and_combo() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="dummy",
            display_name="Dummy",
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=TrainingService(registry),
        initial_plugin_id="dummy",
    )

    second_task = TaskDefinition(
        environment_id="dummy_env",
        name="Independent Task",
        task_id="task_other",
        config={"difficulty": 3},
    )
    window._add_task_to_workspace(second_task, select=False)
    window.task_history_view.set_primary_workspace_index(
        1,
        preserve_multi_selection=False,
        emit_signal=True,
    )

    assert window._current_task is second_task
    assert window.task_history_view.selected_task() is second_task


def test_task_selection_ignores_editor_initialization_change_signal() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="dummy",
            display_name="Dummy",
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=_EmittingGuiExtension(),
        )
    )
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=TrainingService(registry),
        initial_plugin_id="dummy",
    )
    second_task = TaskDefinition(
        environment_id="dummy_env",
        name="Independent Task",
        task_id="task_other",
        config={"difficulty": 3},
    )
    window._add_task_to_workspace(second_task, select=False)

    refresh_count = 0
    original_set_tasks = window.task_history_view.set_tasks

    def _counting_set_tasks(tasks):
        nonlocal refresh_count
        refresh_count += 1
        original_set_tasks(tasks)

    window.task_history_view.set_tasks = _counting_set_tasks  # type: ignore[method-assign]
    window.task_history_view.set_primary_workspace_index(
        1,
        preserve_multi_selection=False,
        emit_signal=True,
    )

    assert refresh_count == 0
    assert window._current_task is second_task
    assert second_task.metadata["editor_initialized"] is True


def test_task_history_add_new_task_button_creates_from_workspace_root() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="dummy",
            display_name="Dummy",
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=TrainingService(registry),
        initial_plugin_id="dummy",
    )

    window._task_workspace[0].name = "Edited Main Task"
    window.task_history_view.create_task_requested.emit()

    assert len(window._task_workspace) == 2
    assert window._task_workspace[-1].name == "Edited Main Task 2"
    assert window.task_history_view.selected_task() is window._task_workspace[-1]


def test_task_history_edit_button_opens_selected_task_editor() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="dummy",
            display_name="Dummy",
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=TrainingService(registry),
        initial_plugin_id="dummy",
    )
    second_task = TaskDefinition(
        environment_id="dummy_env",
        name="Independent Task",
        task_id="task_other",
    )
    window._add_task_to_workspace(second_task, select=False)
    window.task_history_view.set_primary_workspace_index(1, preserve_multi_selection=False, emit_signal=False)

    window.task_history_view.edit_task_button.click()

    assert window._current_task is second_task
    assert window.tabs.currentWidget() is window.task_editor_tab


def test_task_history_copy_duplicates_selected_task() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="dummy",
            display_name="Dummy",
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=TrainingService(registry),
        initial_plugin_id="dummy",
    )
    child_task = DerivedTaskDefinition(
        environment_id="dummy_env",
        name="Child Task",
        task_id="task_child",
        parent_task_id="task_main",
    )
    window._add_task_to_workspace(child_task, select=False)

    window.task_history_view.set_primary_workspace_index(1, preserve_multi_selection=False, emit_signal=False)
    window.task_history_view.copy_task_button.click()

    copied_task = window._task_workspace[-1]
    assert copied_task.name == "Child Task Copy"
    assert copied_task.task_id is None
    assert isinstance(copied_task, DerivedTaskDefinition)
    assert copied_task.parent_task_id == "task_main"


def test_project_is_saved_only_when_save_button_is_clicked(tmp_path) -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="dummy",
            display_name="Dummy",
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )
    project_store = ProjectStore(tmp_path / "project.json")
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=TrainingService(registry),
        initial_plugin_id="dummy",
        project_store=project_store,
    )

    task = window._task_workspace[0]
    task.name = "Unsaved Task Name"
    window._on_task_changed(task)

    assert not project_store.project_path.exists()

    window.save_project_btn.click()
    restored = project_store.load()

    assert restored is not None
    assert restored.task_workspace[0].name == "Unsaved Task Name"


def test_start_training_uses_parallel_launch_when_multiple_tasks_are_selected() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="dummy",
            display_name="Dummy",
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )
    training_service = TrainingService(registry)
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=training_service,
        initial_plugin_id="dummy",
    )

    second_task = TaskDefinition(
        environment_id="dummy_env",
        name="Second Task",
        task_id="task_second",
        config={"difficulty": 2},
    )
    third_task = TaskDefinition(
        environment_id="dummy_env",
        name="Third Task",
        task_id="task_third",
        config={"difficulty": 3},
    )
    window._add_task_to_workspace(second_task, select=False)
    window._add_task_to_workspace(third_task, select=False)
    window.evaluation_view.set_tasks(window._task_workspace)
    window.evaluation_view.task_combo.setCurrentIndex(2)
    window.evaluation_view.episode_count_spin.setValue(4)
    window.evaluation_view.max_steps_per_episode_spin.setValue(77)
    window.evaluation_view.seed_spin.setValue(314)

    window.task_history_view.set_primary_workspace_index(1, preserve_multi_selection=False, emit_signal=True)
    window.task_history_view.toggle_workspace_index_selection(2, emit_signal=True)
    window.task_history_view.set_primary_workspace_index(1, preserve_multi_selection=True, emit_signal=True)

    captured: dict[str, object] = {}

    def _capture_start_many(tasks, config, **kwargs):
        captured["tasks"] = tasks
        captured["config"] = config
        captured["kwargs"] = kwargs

    training_service.start_many = _capture_start_many  # type: ignore[method-assign]
    window._start_training(RunConfig(max_steps=123))

    assert captured["tasks"] == [second_task, third_task]
    assert isinstance(captured["config"], RunConfig)
    config = captured["config"]
    assert config.evaluation_policy["task"]["task_id"] == "task_third"
    assert config.evaluation_policy["episode_count"] == 4
    assert config.evaluation_policy["max_steps_per_episode"] == 77
    assert config.evaluation_policy["seed"] == 314
