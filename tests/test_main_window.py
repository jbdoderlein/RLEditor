from __future__ import annotations

import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from gymnasium.spaces import Discrete
from PySide6.QtWidgets import QApplication, QLabel

from rleditor.application.persistence import ProjectStore
from rleditor.application.services import TaskService, TrainingHistorySnapshot, TrainingService
from rleditor.core.models import (
    Breakpoint,
    Checkpoint,
    DerivedTaskDefinition,
    EpisodeTrace,
    RunConfig,
    TaskDefinition,
    TaskSnapshot,
    TrainingRun,
    TrainingStatus,
)
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


class _TinyEnv:
    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(3)
        self._state = 0

    def reset(self, *, seed: int | None = None):
        _ = seed
        self._state = 0
        return self._state, {}

    def step(self, action: int):
        _ = action
        self._state = min(2, self._state + 1)
        terminated = self._state >= 2
        reward = 1.0 if terminated else 0.0
        return self._state, reward, terminated, False, {"is_success": terminated}

    def close(self) -> None:
        return


class _TinyBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(
            environment_id="tiny_env",
            name="Tiny Main Task",
            task_id="task_tiny",
        )

    def create_env(self, task: TaskDefinition):
        _ = task
        return _TinyEnv()


class _EmittingGuiExtension:
    def create_task_editor_widget(self, task, on_task_changed):
        task.metadata["editor_initialized"] = True
        on_task_changed(task)
        return QLabel("Editor")

    def create_episode_replay_widget(self, parent=None):
        _ = parent
        return None


class _FakeInteractionLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, object]]] = []

    def log(self, event: str, **payload: object) -> None:
        self.records.append((event, payload))


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_for(predicate, *, timeout_seconds: float = 1.0) -> None:
    app = _app()
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.001)
    assert predicate()


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


def test_task_history_delete_removes_multiple_selected_tasks() -> None:
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
    interaction_logger = _FakeInteractionLogger()
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=TrainingService(registry),
        initial_plugin_id="dummy",
        interaction_logger=interaction_logger,  # type: ignore[arg-type]
    )
    second_task = TaskDefinition(
        environment_id="dummy_env",
        name="Second Task",
        task_id="task_second",
    )
    third_task = TaskDefinition(
        environment_id="dummy_env",
        name="Third Task",
        task_id="task_third",
    )
    window._add_task_to_workspace(second_task, select=False)
    window._add_task_to_workspace(third_task, select=False)
    window.task_history_view.set_primary_workspace_index(1, preserve_multi_selection=False, emit_signal=False)
    window.task_history_view.toggle_workspace_index_selection(2, emit_signal=False)

    window.task_history_view.delete_task_button.click()

    assert [task.name for task in window._task_workspace] == ["Dummy Main Task"]
    assert window._current_task is window._task_workspace[0]
    assert window.task_history_view.selected_task() is window._task_workspace[0]
    assert window.evaluation_view.selected_task().name == "Dummy Main Task"
    assert "Deleted 2 task(s)" in window.statusBar().currentMessage()
    assert ("task_deleted", {"deleted_count": 2}) in interaction_logger.records


def test_task_history_delete_all_tasks_clears_selection() -> None:
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

    window.task_history_view.delete_task_button.click()

    assert window._task_workspace == []
    assert window._current_task is None
    assert window.task_history_view.selected_task() is None
    assert window.evaluation_view.selected_task() is None
    assert window.task_editor_view.name_input.text() == ""


def test_task_import_from_generated_curriculum_adds_workspace_task_without_training() -> None:
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
    payload = {
        "curriculum": {
            "steps": [{"env_id": "0", "algorithm": "q_learning"}],
        },
        "environments": [
            {
                "environment_id": "dummy_env",
                "task_id": 0,
                "task_name": "Generated Eval Task",
                "task_config": {"difficulty": 4},
                "reward_config": {"goal": 2.0},
                "termination_config": {"max_steps": 33},
            }
        ],
    }

    imported_count, selected_index = window._import_tasks_from_payload(payload)

    assert imported_count == 1
    assert selected_index == 1
    assert len(window._task_workspace) == 2
    imported_task = window._task_workspace[1]
    assert imported_task.name == "Generated Eval Task"
    assert imported_task.task_id is None
    assert imported_task.config == {"difficulty": 4}
    assert imported_task.reward_config == {"goal": 2.0}
    assert imported_task.termination_config == {"max_steps": 33}
    assert window.task_history_view.selected_task() is imported_task
    assert window.evaluation_view.selected_task().name == "Generated Eval Task"
    assert training_service.status == TrainingStatus.IDLE


def test_task_import_dialog_accepts_multiple_selected_files(tmp_path, monkeypatch) -> None:
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
    first_path = tmp_path / "first_task.json"
    second_path = tmp_path / "second_task.json"
    first_path.write_text(
        json.dumps(
            {
                "environment_id": "dummy_env",
                "name": "Imported First",
                "config": {"difficulty": 11},
            }
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(
            {
                "environment_id": "dummy_env",
                "name": "Imported Second",
                "config": {"difficulty": 12},
            }
        ),
        encoding="utf-8",
    )

    def _select_files(*_args, **_kwargs):
        return ([str(first_path), str(second_path)], "JSON Files (*.json)")

    monkeypatch.setattr(
        "rleditor.ui.shell.main_window.QFileDialog.getOpenFileNames",
        _select_files,
    )

    window._on_task_import_requested()

    assert [task.name for task in window._task_workspace[-2:]] == [
        "Imported First",
        "Imported Second",
    ]
    assert window.task_history_view.selected_task().name == "Imported First"
    assert window.evaluation_view.selected_task().name == "Imported First"
    assert "Imported 2 task(s)" in window.statusBar().currentMessage()


def test_task_import_reuses_existing_matching_workspace_task() -> None:
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
    existing_task = window._task_workspace[0]
    payload = {
        "environment_id": "dummy_env",
        "name": existing_task.name,
        "config": dict(existing_task.config),
        "reward_config": dict(existing_task.reward_config),
        "termination_config": dict(existing_task.termination_config),
    }

    imported_count, selected_index = window._import_tasks_from_payload(payload)

    assert imported_count == 0
    assert selected_index == 0
    assert window._task_workspace == [existing_task]
    assert window.task_history_view.selected_task() is existing_task


def test_curriculum_import_adds_tasks_and_starts_first_step() -> None:
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
    interaction_logger = _FakeInteractionLogger()
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=training_service,
        initial_plugin_id="dummy",
        interaction_logger=interaction_logger,  # type: ignore[arg-type]
    )
    payload = {
        "curriculum": {
            "size": 2,
            "seed": 42,
            "steps": [
                {
                    "env_id": 0,
                    "steps": 100,
                    "max_episode_length": 12,
                    "algorithm": "Q-learning",
                    "learning_rate": 0.2,
                    "discount_factor": 0.95,
                    "epsilon_start": 0.3,
                },
                {
                    "env_id": 1,
                    "steps": 50,
                    "algorithm": "q_learning",
                    "episode_trace_sample_rate": 0.0,
                },
            ],
        },
        "evaluation": {
            "evaluation_env": 0,
            "eval_episodes": 5,
            "eval_seed": 7,
        },
        "environments": [
            {
                "task_id": 0,
                "environment_id": "dummy_env",
                "task_name": "Imported Main",
                "task_config": {"difficulty": 1},
                "reward_config": {"goal": 1.0},
                "metadata": {},
            },
            {
                "task_id": 1,
                "environment_id": "dummy_env",
                "task_name": "Imported Sub",
                "task_config": {"difficulty": 2},
                "metadata": {},
            },
        ],
    }
    captured: dict[str, object] = {}

    def _capture_start(task, config, **kwargs):
        captured["task"] = task
        captured["config"] = config
        captured["kwargs"] = kwargs

    training_service.start = _capture_start  # type: ignore[method-assign]

    window._on_curriculum_import_requested(payload)

    assert [task.name for task in window._task_workspace[-2:]] == ["Imported Main", "Imported Sub"]
    assert captured["task"] is window._task_workspace[-2]
    assert isinstance(captured["config"], RunConfig)
    config = captured["config"]
    assert config.algorithm == "q_learning"
    assert config.seed == 42
    assert config.max_steps == 100
    assert config.max_steps_per_episode == 12
    assert config.learning_rate == 0.2
    assert config.gamma == 0.95
    assert config.epsilon == 1.0
    assert config.hyperparameters["epsilon"] == 1.0
    assert config.episode_trace_sample_rate == 1.0
    assert config.evaluation_policy["episode_count"] == 5
    assert config.evaluation_policy["seed"] == 7
    assert captured["kwargs"]["start_from_scratch"] is True
    assert len(window._imported_curriculum_queue) == 1
    queued_task, queued_config = window._imported_curriculum_queue[0]
    assert queued_task is window._task_workspace[-1]
    assert queued_config.max_steps_per_episode == 100
    assert queued_config.epsilon == 1.0
    assert queued_config.hyperparameters["epsilon"] == 1.0
    assert queued_config.episode_trace_sample_rate == 1.0
    assert queued_config.evaluation_policy["episode_count"] == 5
    assert queued_config.evaluation_policy["seed"] == 7


def test_curriculum_import_without_evaluation_uses_final_step_as_default_eval_task() -> None:
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
    window.evaluation_view.episode_count_spin.setValue(4)
    window.evaluation_view.max_steps_per_episode_spin.setValue(77)
    window.evaluation_view.seed_spin.setValue(314)
    payload = {
        "curriculum": {
            "steps": [
                {"env_id": "easy", "algorithm": "q_learning"},
                {"env_id": "target", "algorithm": "q_learning"},
            ],
        },
        "environments": [
            {
                "task_id": "easy",
                "environment_id": "dummy_env",
                "task_name": "Easy Step",
                "task_config": {"difficulty": 1},
            },
            {
                "task_id": "target",
                "environment_id": "dummy_env",
                "task_name": "Target Step",
                "task_config": {"difficulty": 2},
            },
        ],
    }
    captured: dict[str, object] = {}

    def _capture_start(task, config, **kwargs):
        captured["task"] = task
        captured["config"] = config
        captured["kwargs"] = kwargs

    training_service.start = _capture_start  # type: ignore[method-assign]

    window._on_curriculum_import_requested(payload)

    assert isinstance(captured["config"], RunConfig)
    config = captured["config"]
    assert config.evaluation_policy["task"]["name"] == "Target Step"
    assert config.evaluation_policy["episode_count"] == 4
    assert config.evaluation_policy["max_steps_per_episode"] == 77
    assert config.evaluation_policy["seed"] == 314
    _queued_task, queued_config = window._imported_curriculum_queue[0]
    assert queued_config.evaluation_policy["task"]["name"] == "Target Step"


def test_curriculum_import_records_all_training_episodes_by_default() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="tiny_env",
            display_name="Tiny",
            description="Tiny curriculum trace plugin",
            backend=_TinyBackend(),
            gui_extension=None,
        )
    )
    training_service = TrainingService(registry)
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=training_service,
        initial_plugin_id="tiny_env",
    )
    payload = {
        "curriculum": {
            "steps": [
                {
                    "env_id": 0,
                    "algorithm": "q_learning",
                    "max_episodes": 2,
                    "max_episode_length": 5,
                    "episode_trace_sample_rate": 0.0,
                }
            ],
        },
        "evaluation": False,
        "environments": [
            {
                "task_id": 0,
                "environment_id": "tiny_env",
                "task_name": "Imported Tiny Trace Task",
            }
        ],
    }

    window._on_curriculum_import_requested(payload)

    _wait_for(
        lambda: training_service.status == TrainingStatus.FINISHED
        and window._imported_curriculum_active is False,
        timeout_seconds=2.0,
    )

    snapshot = training_service.history_snapshot()
    assert len(snapshot.runs) == 1
    training_run_id = snapshot.runs[0].run_id
    assert len(snapshot.episodes_by_run[training_run_id]) == 2
    assert all(trace.steps for trace in snapshot.episodes_by_run[training_run_id])


def test_curriculum_import_reuses_matching_workspace_task() -> None:
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
        initial_tasks=[
            TaskDefinition(
                environment_id="dummy_env",
                name="Imported Main",
                task_id="task_existing",
                config={"difficulty": 1},
                reward_config={"goal": 1.0},
                termination_config={"max_steps": 20},
                metadata={"local_note": "keep"},
            )
        ],
    )
    existing_task = window._task_workspace[0]
    payload = {
        "curriculum": {
            "size": 2,
            "steps": [
                {"env_id": "same", "steps": 100},
                {"env_id": "changed", "steps": 50},
            ],
        },
        "environments": [
            {
                "task_id": "same",
                "environment_id": "dummy_env",
                "task_name": "Imported Main",
                "task_config": {"difficulty": 1},
                "reward_config": {"goal": 1.0},
                "termination_config": {"max_steps": 20},
                "metadata": {"export_note": "ignored for reuse"},
            },
            {
                "task_id": "changed",
                "environment_id": "dummy_env",
                "task_name": "Imported Main",
                "task_config": {"difficulty": 2},
                "reward_config": {"goal": 1.0},
                "termination_config": {"max_steps": 20},
            },
        ],
    }
    captured: dict[str, object] = {}

    def _capture_start(task, config, **kwargs):
        captured["task"] = task
        captured["config"] = config
        captured["kwargs"] = kwargs

    training_service.start = _capture_start  # type: ignore[method-assign]

    window._on_curriculum_import_requested(payload)

    assert len(window._task_workspace) == 2
    assert window._task_workspace[0] is existing_task
    assert window._task_workspace[1].name == "Imported Main 2"
    assert window._task_workspace[1].config == {"difficulty": 2}
    assert captured["task"] is existing_task
    queued_task, _queued_config = window._imported_curriculum_queue[0]
    assert queued_task is window._task_workspace[1]


@pytest.mark.parametrize("breakpoint_actions", [["checkpoint"], ["pause", "checkpoint"]])
def test_curriculum_import_ignores_exported_breakpoints_and_runs_episode_budget(
    breakpoint_actions: list[str],
) -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="tiny_env",
            display_name="Tiny",
            description="Tiny curriculum test plugin",
            backend=_TinyBackend(),
            gui_extension=None,
        )
    )
    training_service = TrainingService(registry)
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=training_service,
        initial_plugin_id="tiny_env",
    )
    payload = {
        "curriculum": {
            "size": 1,
            "seed": 42,
            "steps": [
                {
                    "env_id": 0,
                    "steps": 100,
                    "max_episodes": 2,
                    "algorithm": "q_learning",
                    "breakpoints": [
                        {
                            "kind": "max_step",
                            "value": 1,
                            "actions": breakpoint_actions,
                        }
                    ],
                }
            ],
        },
        "environments": [
            {
                "task_id": 0,
                "environment_id": "tiny_env",
                "task_name": "Imported Tiny",
            }
        ],
    }

    window._on_curriculum_import_requested(payload)

    _wait_for(lambda: not window._imported_curriculum_active, timeout_seconds=2.0)

    assert window._imported_curriculum_active is False
    assert window._imported_curriculum_waiting_for_step is False
    assert not window._imported_curriculum_queue
    snapshot = training_service.history_snapshot()
    assert len(snapshot.checkpoints) == 1
    assert snapshot.checkpoints[-1].episode == 2
    assert snapshot.runs[-1].metadata["run_config"]["breakpoints"] == []
    assert snapshot.runs[-1].metadata["run_config"]["metadata"]["ignored_imported_breakpoint_count"] == 1
    assert "Curriculum execution completed" in window.statusBar().currentMessage()

    training_service.start(
        TaskDefinition(environment_id="tiny_env", name="Manual Tiny", task_id="task_manual"),
        RunConfig(max_steps=2, max_episodes=1, seed=99),
        start_from_scratch=True,
    )

    assert training_service.status == TrainingStatus.RUNNING
    training_service.stop()


def test_status_bar_busy_indicator_tracks_active_work() -> None:
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

    assert "font-size: 14px" in window.statusBar().styleSheet()
    assert window.status_busy_indicator.isHidden()

    window._set_status_busy("training", True)

    assert not window.status_busy_indicator.isHidden()
    assert window._status_busy_timer.isActive()
    first_frame = window.status_busy_indicator.text()

    window._advance_status_busy_indicator()

    assert window.status_busy_indicator.text() != first_frame

    window._set_status_busy("training", False)

    assert window.status_busy_indicator.isHidden()
    assert not window._status_busy_timer.isActive()

    window._set_status_busy("curriculum", True)
    window._imported_curriculum_active = True
    window._imported_curriculum_waiting_for_step = True
    window._on_status_changed(TrainingStatus.PAUSED)

    assert window.status_busy_indicator.isHidden()


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


def test_checkpoint_history_edge_selection_applies_config_to_training_tab() -> None:
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
    interaction_logger = _FakeInteractionLogger()
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=TrainingService(registry),
        initial_plugin_id="dummy",
        interaction_logger=interaction_logger,  # type: ignore[arg-type]
    )
    selected_config = RunConfig(
        algorithm="sb3_ppo",
        max_steps=None,
        max_episodes=24,
        max_steps_per_episode=90,
        episode_trace_sample_rate=0.35,
        learning_rate=0.12,
        gamma=0.94,
        breakpoints=[
            Breakpoint(
                kind="success_rate_gte",
                value=0.7,
                window=10,
                actions=["pause", "checkpoint"],
            )
        ],
    )
    run = TrainingRun(
        run_id="run_selected",
        task_id="task_selected",
        status=TrainingStatus.FINISHED,
        metadata={"run_config": selected_config.to_dict()},
    )
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_selected",
        label="Selected checkpoint",
        created_at="2026-05-17 10:00:00",
        reason="run_finished",
        run_id="run_selected",
        task_id="task_selected",
        task_name="Selected Task",
        step=120,
        episode=24,
        task_snapshot=TaskSnapshot(environment_id="dummy_env", task_name="Selected Task"),
    )
    window.history_view.set_history(
        TrainingHistorySnapshot(
            runs=[run],
            checkpoints=[checkpoint],
            episodes_by_run={},
            run_task_snapshots={},
        )
    )

    edge = window.history_view.graph_widget.edge_for_id("edge:checkpoint_selected")
    assert edge is not None
    window.history_view.graph_widget.select_edge(edge.edge_id)
    window.history_view.graph_widget.edge_selected.emit(edge)

    applied_config = window.training_view.build_config()
    assert applied_config.algorithm == "sb3_ppo"
    assert applied_config.max_steps is None
    assert applied_config.max_episodes == 24
    assert applied_config.max_steps_per_episode == 90
    assert applied_config.episode_trace_sample_rate == pytest.approx(0.35)
    assert applied_config.learning_rate == pytest.approx(0.12)
    assert applied_config.gamma == pytest.approx(0.94)
    assert len(applied_config.breakpoints) == 1
    assert applied_config.breakpoints[0].kind == "success_rate_gte"
    assert "Training config loaded from selected run" in window.statusBar().currentMessage()
    assert any(
        event == "training_config_loaded_from_history"
        and payload["algorithm"] == "sb3_ppo"
        and payload["max_episodes"] == 24
        for event, payload in interaction_logger.records
    )


def test_checkpoint_history_live_edit_replays_deepest_descendant_branch() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="tiny_env",
            display_name="Tiny",
            description="Tiny live edit plugin",
            backend=_TinyBackend(),
            gui_extension=None,
        )
    )
    training_service = TrainingService(registry)
    interaction_logger = _FakeInteractionLogger()
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=training_service,
        initial_plugin_id="tiny_env",
        interaction_logger=interaction_logger,  # type: ignore[arg-type]
    )
    task_snapshot = TaskSnapshot(
        environment_id="tiny_env",
        task_name="Tiny Task",
        task_id="task_tiny",
    )
    source_config = RunConfig(algorithm="q_learning", max_episodes=1, max_steps_per_episode=5)
    selected_config = RunConfig(algorithm="q_learning", max_episodes=1, max_steps_per_episode=5)
    child_config = RunConfig(algorithm="q_learning", max_episodes=1, max_steps_per_episode=5)
    runs = [
        TrainingRun(
            run_id="run_001",
            task_id="task_tiny",
            status=TrainingStatus.FINISHED,
            metadata={"algorithm": "q_learning", "run_config": source_config.to_dict()},
        ),
        TrainingRun(
            run_id="run_002",
            task_id="task_tiny",
            status=TrainingStatus.FINISHED,
            parent_checkpoint_id="checkpoint_001",
            metadata={"algorithm": "q_learning", "run_config": selected_config.to_dict()},
        ),
        TrainingRun(
            run_id="run_003",
            task_id="task_tiny",
            status=TrainingStatus.FINISHED,
            parent_checkpoint_id="checkpoint_002",
            metadata={"algorithm": "q_learning", "run_config": child_config.to_dict()},
        ),
        TrainingRun(
            run_id="run_004",
            task_id="task_tiny",
            status=TrainingStatus.FINISHED,
            parent_checkpoint_id="checkpoint_002",
            metadata={"algorithm": "q_learning", "run_config": child_config.to_dict()},
        ),
        TrainingRun(
            run_id="run_005",
            task_id="task_tiny",
            status=TrainingStatus.FINISHED,
            parent_checkpoint_id="checkpoint_003",
            metadata={"algorithm": "q_learning", "run_config": child_config.to_dict()},
        ),
    ]
    learner_state = {"algorithm": "q_learning", "q_values": []}
    checkpoints = [
        Checkpoint(
            checkpoint_id="checkpoint_001",
            label="Checkpoint 001",
            created_at="2026-05-17 09:00:00",
            reason="run_finished",
            run_id="run_001",
            task_id="task_tiny",
            task_name="Tiny Task",
            step=2,
            episode=1,
            task_snapshot=task_snapshot,
            metadata={"algorithm": "q_learning", "learner_state": learner_state},
        ),
        Checkpoint(
            checkpoint_id="checkpoint_002",
            label="Checkpoint 002",
            created_at="2026-05-17 09:01:00",
            reason="run_finished",
            parent_checkpoint_id="checkpoint_001",
            run_id="run_002",
            task_id="task_tiny",
            task_name="Tiny Task",
            step=2,
            episode=1,
            task_snapshot=task_snapshot,
            metadata={"algorithm": "q_learning", "learner_state": learner_state},
        ),
        Checkpoint(
            checkpoint_id="checkpoint_003",
            label="Checkpoint 003",
            created_at="2026-05-17 09:02:00",
            reason="run_finished",
            parent_checkpoint_id="checkpoint_002",
            run_id="run_003",
            task_id="task_tiny",
            task_name="Tiny Task",
            step=2,
            episode=1,
            task_snapshot=task_snapshot,
            metadata={"algorithm": "q_learning", "learner_state": learner_state},
        ),
        Checkpoint(
            checkpoint_id="checkpoint_004",
            label="Checkpoint 004",
            created_at="2026-05-17 09:03:00",
            reason="run_finished",
            parent_checkpoint_id="checkpoint_002",
            run_id="run_004",
            task_id="task_tiny",
            task_name="Tiny Task",
            step=2,
            episode=1,
            task_snapshot=task_snapshot,
            metadata={"algorithm": "q_learning", "learner_state": learner_state},
        ),
        Checkpoint(
            checkpoint_id="checkpoint_005",
            label="Checkpoint 005",
            created_at="2026-05-17 09:04:00",
            reason="run_finished",
            parent_checkpoint_id="checkpoint_003",
            run_id="run_005",
            task_id="task_tiny",
            task_name="Tiny Task",
            step=2,
            episode=1,
            task_snapshot=task_snapshot,
            metadata={"algorithm": "q_learning", "learner_state": learner_state},
        ),
    ]
    snapshot = TrainingHistorySnapshot(
        runs=runs,
        checkpoints=checkpoints,
        episodes_by_run={},
        run_task_snapshots={run.run_id: task_snapshot for run in runs},
    )
    training_service.load_history(snapshot)
    window.history_view.set_history(training_service.history_snapshot(deep=False))
    edge = window.history_view.graph_widget.edge_for_id("edge:checkpoint_002")
    assert edge is not None

    edited_config = RunConfig(
        algorithm="q_learning",
        max_episodes=1,
        max_steps_per_episode=5,
        learning_rate=0.22,
        gamma=0.91,
    )
    window._on_training_edge_live_edit_requested(edge, edited_config)

    _wait_for(lambda: not window._live_edit_active, timeout_seconds=2.0)

    replay_snapshot = training_service.history_snapshot()
    new_checkpoints = replay_snapshot.checkpoints[5:]
    assert len(new_checkpoints) == 3
    assert new_checkpoints[0].parent_checkpoint_id == "checkpoint_001"
    assert new_checkpoints[1].parent_checkpoint_id == new_checkpoints[0].checkpoint_id
    assert new_checkpoints[2].parent_checkpoint_id == new_checkpoints[1].checkpoint_id

    replay_run_configs = [
        run.metadata["run_config"]
        for run in replay_snapshot.runs[5:]
    ]
    assert [config["metadata"]["live_edit_role"] for config in replay_run_configs] == [
        "edited_edge",
        "replayed_descendant_edge",
        "replayed_descendant_edge",
    ]
    assert [config["metadata"]["live_edit_original_target_checkpoint_id"] for config in replay_run_configs] == [
        "checkpoint_002",
        "checkpoint_003",
        "checkpoint_005",
    ]
    assert replay_run_configs[0]["learning_rate"] == pytest.approx(0.22)
    assert any(event == "live_edit_replay_completed" for event, _payload in interaction_logger.records)


def test_checkpoint_history_live_edit_planning_avoids_deep_history_snapshot() -> None:
    _app()
    registry = PluginRegistry()
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id="tiny_env",
            display_name="Tiny",
            description="Tiny live edit planning plugin",
            backend=_TinyBackend(),
            gui_extension=None,
        )
    )
    training_service = TrainingService(registry)
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=training_service,
        initial_plugin_id="tiny_env",
    )
    task_snapshot = TaskSnapshot(
        environment_id="tiny_env",
        task_name="Tiny Task",
        task_id="task_tiny",
    )
    run_config = RunConfig(algorithm="q_learning", max_episodes=1, max_steps_per_episode=5)
    runs = [
        TrainingRun(
            run_id="run_001",
            task_id="task_tiny",
            status=TrainingStatus.FINISHED,
            metadata={"algorithm": "q_learning", "run_config": run_config.to_dict()},
        ),
        TrainingRun(
            run_id="run_002",
            task_id="task_tiny",
            status=TrainingStatus.FINISHED,
            parent_checkpoint_id="checkpoint_001",
            metadata={"algorithm": "q_learning", "run_config": run_config.to_dict()},
        ),
    ]
    learner_state = {
        "algorithm": "q_learning",
        "q_values": [
            {"state_key": str(index), "action": 0, "value": float(index)}
            for index in range(500)
        ],
    }
    checkpoints = [
        Checkpoint(
            checkpoint_id="checkpoint_001",
            label="Checkpoint 001",
            created_at="2026-05-17 09:00:00",
            reason="run_finished",
            run_id="run_001",
            task_id="task_tiny",
            task_name="Tiny Task",
            step=2,
            episode=1,
            task_snapshot=task_snapshot,
            metadata={"algorithm": "q_learning", "learner_state": learner_state},
        ),
        Checkpoint(
            checkpoint_id="checkpoint_002",
            label="Checkpoint 002",
            created_at="2026-05-17 09:01:00",
            reason="run_finished",
            parent_checkpoint_id="checkpoint_001",
            run_id="run_002",
            task_id="task_tiny",
            task_name="Tiny Task",
            step=2,
            episode=1,
            task_snapshot=task_snapshot,
            metadata={"algorithm": "q_learning", "learner_state": learner_state},
        ),
    ]
    training_service.load_history(
        TrainingHistorySnapshot(
            runs=runs,
            checkpoints=checkpoints,
            episodes_by_run={},
            run_task_snapshots={run.run_id: task_snapshot for run in runs},
        )
    )
    window.history_view.set_history(training_service.history_snapshot(deep=False))
    edge = window.history_view.graph_widget.edge_for_id("edge:checkpoint_002")
    assert edge is not None

    original_history_snapshot = training_service.history_snapshot
    history_snapshot_deep_args: list[bool] = []

    def _capture_history_snapshot(*, deep: bool = True):
        history_snapshot_deep_args.append(deep)
        return original_history_snapshot(deep=deep)

    captured_start: dict[str, object] = {}

    def _capture_start(task, config, **kwargs):
        captured_start["task"] = task
        captured_start["config"] = config
        captured_start["kwargs"] = kwargs

    training_service.history_snapshot = _capture_history_snapshot  # type: ignore[method-assign]
    training_service.start = _capture_start  # type: ignore[method-assign]

    window._on_training_edge_live_edit_requested(
        edge,
        RunConfig(algorithm="q_learning", max_episodes=1, max_steps_per_episode=5),
    )

    assert history_snapshot_deep_args == [False]
    assert captured_start["kwargs"]["initial_checkpoint"] is edge.source_checkpoint


def test_multiple_evaluation_button_starts_training_on_checked_tasks_with_self_evaluation(monkeypatch) -> None:
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
    interaction_logger = _FakeInteractionLogger()
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=training_service,
        initial_plugin_id="dummy",
        interaction_logger=interaction_logger,  # type: ignore[arg-type]
    )
    second_task = TaskDefinition(
        environment_id="dummy_env",
        name="Second Eval",
        task_id="task_second_eval",
        config={"difficulty": 2},
    )
    window._add_task_to_workspace(second_task, select=False)
    for checkbox, _task_index in window.evaluation_view._multi_task_checkboxes:
        checkbox.setChecked(True)
    window.evaluation_view.episode_count_spin.setValue(6)
    window.evaluation_view.max_steps_per_episode_spin.setValue(55)
    window.evaluation_view.seed_spin.setValue(42)
    window.evaluation_view.multi_training_episode_count_spin.setValue(123)
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_eval",
        label="Checkpoint Eval",
        created_at="2026-06-05 10:00:00",
        reason="test",
        metadata={"learner_state": {"q_values": []}},
    )
    monkeypatch.setattr(window.history_view, "selected_checkpoint", lambda: checkpoint)

    captured: dict[str, object] = {}

    def _start_many_with_configs(tasks, configs, **kwargs):
        captured["tasks"] = tasks
        captured["configs"] = configs
        captured["kwargs"] = kwargs

    training_service.start_many_with_configs = _start_many_with_configs  # type: ignore[method-assign]

    window.evaluation_view.evaluate_multiple_button.click()

    assert [task.name for task in captured["tasks"]] == ["Dummy Main Task", "Second Eval"]
    configs = captured["configs"]
    assert [config.max_episodes for config in configs] == [123, 123]
    assert [config.max_steps_per_episode for config in configs] == [100, 100]
    assert [config.evaluation_policy["task"]["name"] for config in configs] == [
        "Dummy Main Task",
        "Second Eval",
    ]
    assert [config.evaluation_policy["episode_count"] for config in configs] == [6, 6]
    assert [config.evaluation_policy["max_steps_per_episode"] for config in configs] == [55, 55]
    assert [config.evaluation_policy["seed"] for config in configs] == [42, 42]
    assert captured["kwargs"]["initial_checkpoint"] is checkpoint
    assert captured["kwargs"]["run_in_background"] is True
    assert "Batch training started on 2 task(s)" in window.statusBar().currentMessage()
    logged = [
        payload
        for event, payload in interaction_logger.records
        if event == "multiple_task_adaptation_started"
    ]
    assert logged[-1]["task_count"] == 2
    assert logged[-1]["training_episodes"] == 123


def test_checkpoint_history_delete_removes_selected_checkpoint_subtree() -> None:
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
    interaction_logger = _FakeInteractionLogger()
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=training_service,
        initial_plugin_id="dummy",
        interaction_logger=interaction_logger,  # type: ignore[arg-type]
    )
    task_snapshot = TaskSnapshot(environment_id="dummy_env", task_name="Dummy Task", task_id="task_main")
    training_service.load_history(
        TrainingHistorySnapshot(
            runs=[
                TrainingRun(run_id="run_001", task_id="task_main", status=TrainingStatus.FINISHED),
                TrainingRun(
                    run_id="run_002",
                    task_id="task_main",
                    status=TrainingStatus.FINISHED,
                    parent_checkpoint_id="checkpoint_001",
                ),
            ],
            checkpoints=[
                Checkpoint(
                    checkpoint_id="checkpoint_001",
                    label="Checkpoint 001",
                    created_at="2026-06-11 10:00:00",
                    reason="run_finished",
                    run_id="run_001",
                    task_snapshot=task_snapshot,
                    metadata={"learner_state": {"q_values": []}},
                ),
                Checkpoint(
                    checkpoint_id="checkpoint_002",
                    label="Checkpoint 002",
                    created_at="2026-06-11 10:01:00",
                    reason="run_finished",
                    parent_checkpoint_id="checkpoint_001",
                    run_id="run_002",
                    task_snapshot=task_snapshot,
                    metadata={"learner_state": {"q_values": []}},
                ),
            ],
            episodes_by_run={
                "run_001": [EpisodeTrace(episode_id=1, run_id="run_001", total_reward=1.0, success=True)],
                "run_002": [EpisodeTrace(episode_id=1, run_id="run_002", total_reward=2.0, success=True)],
            },
            run_task_snapshots={
                "run_001": task_snapshot,
                "run_002": task_snapshot,
            },
        )
    )
    window.history_view.graph_widget.select_node("checkpoint_001")
    node = window.history_view.graph_widget.node_for_id("checkpoint_001")
    assert node is not None
    window.history_view._show_node_details(node)

    window.history_view.delete_checkpoint_button.click()
    snapshot = training_service.history_snapshot()

    assert snapshot.checkpoints == []
    assert snapshot.runs == []
    assert snapshot.episodes_by_run == {}
    assert window.history_view.graph_widget.node_for_id("checkpoint_001") is None
    assert "Deleted 2 checkpoint(s)" in window.statusBar().currentMessage()
    logged = [
        payload
        for event, payload in interaction_logger.records
        if event == "checkpoint_deleted"
    ]
    assert logged[-1]["deleted_count"] == 2


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
    interaction_logger = _FakeInteractionLogger()
    window = MainWindow(
        registry=registry,
        task_service=TaskService(registry),
        training_service=training_service,
        initial_plugin_id="dummy",
        interaction_logger=interaction_logger,  # type: ignore[arg-type]
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
    training_started = [
        payload
        for event, payload in interaction_logger.records
        if event == "training_started"
    ]
    assert training_started
    assert training_started[-1]["mode"] == "parallel"
    assert training_started[-1]["algorithm"] == "q_learning"
    assert training_started[-1]["max_steps"] == 123
    assert training_started[-1]["seed"] is None
    assert training_started[-1]["tasks"] == [
        {"task_id": "task_second", "name": "Second Task", "environment_id": "dummy_env"},
        {"task_id": "task_third", "name": "Third Task", "environment_id": "dummy_env"},
    ]
