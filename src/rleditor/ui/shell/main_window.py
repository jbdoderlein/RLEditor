from __future__ import annotations

from copy import deepcopy

from PySide6.QtWidgets import (
    QFrame,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rleditor.application.persistence import ProjectState, ProjectStore
from rleditor.application.services import TaskService, TrainingService
from rleditor.core.models import (
    Checkpoint,
    EpisodeTrace,
    RunConfig,
    TaskDefinition,
    TaskDerivationOptions,
    TrainingMetrics,
    TrainingStatus,
)
from rleditor.plugins.base import EnvironmentPlugin
from rleditor.plugins.registry import PluginRegistry
from rleditor.ui.views.checkpoint_history_view import CheckpointHistoryView
from rleditor.ui.views.evaluation_view import EvaluationView
from rleditor.ui.views.episode_inspector_view import EpisodeInspectorView
from rleditor.ui.views.task_editor_view import TaskEditorView
from rleditor.ui.views.task_history_view import TaskHistoryView
from rleditor.ui.views.training_monitor_view import TrainingMonitorView


class MainWindow(QMainWindow):
    def __init__(
        self,
        registry: PluginRegistry,
        task_service: TaskService,
        training_service: TrainingService,
        initial_plugin_id: str,
        initial_tasks: list[TaskDefinition] | None = None,
        project_store: ProjectStore | None = None,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._task_service = task_service
        self._training_service = training_service
        self._project_store = project_store
        self._loading_project = True

        self._current_plugin: EnvironmentPlugin | None = None
        self._current_task: TaskDefinition | None = None
        self._task_workspace: list[TaskDefinition] = []

        self.setWindowTitle("RL Debug Studio")
        self._build_ui()
        self._wire_signals()
        self._load_initial_plugin(initial_plugin_id, initial_tasks=initial_tasks)
        self._loading_project = False
        self._refresh_history_view()

    def _build_ui(self) -> None:
        surface = QFrame(self)
        surface.setObjectName("MainSurface")
        root = QVBoxLayout(surface)

        self.tabs = QTabWidget(surface)
        self.history_view = CheckpointHistoryView()
        self.task_history_view = TaskHistoryView()
        self.task_editor_view = TaskEditorView()
        self.training_view = TrainingMonitorView()
        self.evaluation_view = EvaluationView()
        self.episode_view = EpisodeInspectorView()
        self.history_tab = self.history_view
        self.task_history_tab = self.task_history_view
        self.task_editor_tab = self._wrap_tab(self.task_editor_view)
        self.training_tab = self._wrap_tab(self.training_view)
        self.evaluation_tab = self._wrap_tab(self.evaluation_view)
        self.episode_tab = self._wrap_tab(self.episode_view)

        self.tabs.addTab(self.history_tab, "Checkpoint History")
        self.tabs.addTab(self.task_history_tab, "Task History")
        self.tabs.addTab(self.task_editor_tab, "Task Editor")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.evaluation_tab, "Evaluation")
        self.tabs.addTab(self.episode_tab, "Episode Inspector")

        root.addWidget(self.tabs, 1)
        self.setCentralWidget(surface)
        self.save_project_btn = QPushButton("Save Project", self)
        self.save_project_btn.setEnabled(self._project_store is not None)
        self.save_project_btn.setToolTip("Write the current task workspace and training history to disk.")
        self.statusBar().addPermanentWidget(self.save_project_btn)
        self.statusBar().showMessage("Ready")

    def _wire_signals(self) -> None:
        self.task_history_view.selection_changed.connect(self._on_task_history_selection_changed)
        self.task_history_view.create_task_requested.connect(self._create_new_task)
        self.task_editor_view.task_changed.connect(self._on_task_changed)
        self.episode_view.create_task_from_moment_requested.connect(self._on_derive_task_from_episode_moment)
        self.history_view.inspect_episode_requested.connect(self._inspect_episode_from_history)
        self.history_view.checkpoint_import_requested.connect(self._on_checkpoint_import_requested)

        self.training_view.start_requested.connect(self._start_training)
        self.training_view.pause_requested.connect(self._training_service.pause)
        self.training_view.resume_requested.connect(self._training_service.resume)
        self.training_view.stop_requested.connect(self._training_service.stop)

        self._training_service.status_changed.connect(self._on_status_changed)
        self._training_service.metrics_updated.connect(self._on_metrics_updated)
        self._training_service.run_metrics_updated.connect(self._on_run_metrics_updated)
        self._training_service.episode_captured.connect(self._on_episode_captured)
        self._training_service.breakpoint_triggered.connect(self._on_breakpoint_triggered)
        self._training_service.history_changed.connect(self._on_history_changed)
        self.save_project_btn.clicked.connect(self._on_save_project_requested)

    def _load_initial_plugin(
        self,
        plugin_id: str,
        *,
        initial_tasks: list[TaskDefinition] | None = None,
    ) -> None:
        plugin = self._registry.get_environment_plugin(plugin_id)
        tasks = [deepcopy(task) for task in initial_tasks or []]
        if not tasks:
            tasks = [self._task_service.create_default_task(plugin.plugin_id)]

        self._current_plugin = plugin
        self._task_workspace = tasks
        self.evaluation_view.set_tasks(self._task_workspace)
        self._refresh_task_history_view(selected_workspace_index=0, preserve_multi_selection=False)
        self._select_task_index(0, sync_task_history=True, preserve_graph_multi_selection=False)
        self.episode_view.clear_episodes()
        self.episode_view.set_context(plugin)

        self.statusBar().showMessage(f"Loaded environment: {plugin.display_name}")

    def _on_task_changed(self, task: TaskDefinition) -> None:
        self._current_task = task
        selected_workspace_index = self._workspace_index_for_task(task)
        self._refresh_task_history_view(
            selected_workspace_index=selected_workspace_index,
            preserve_multi_selection=True,
        )
        self.statusBar().showMessage("Task updated")

    def _on_task_history_selection_changed(
        self,
        primary_task: TaskDefinition | None,
        selected_tasks: list[TaskDefinition],
    ) -> None:
        if primary_task is None:
            return

        workspace_index = self._workspace_index_for_task(primary_task)
        if workspace_index is None:
            return

        self._select_task_index(workspace_index, sync_task_history=False)
        if len(selected_tasks) > 1:
            self.statusBar().showMessage(
                f"Selected training task: {primary_task.name} ({len(selected_tasks)} tasks selected)"
            )
        else:
            self.statusBar().showMessage(f"Selected training task: {primary_task.name}")

    def _on_derive_task_from_episode_moment(self, trace: EpisodeTrace, moment_index: int) -> None:
        plugin = self._current_plugin
        if plugin is None:
            return

        snapshot = trace.task_snapshot
        if snapshot is not None:
            source_task = self._task_service.task_from_snapshot(snapshot)
        elif self._current_task is not None:
            source_task = self._current_task
        else:
            return

        derivation_options = TaskDerivationOptions(
            derivation_reason="start_from_episode_moment",
            source_episode_id=trace.episode_id,
            source_moment_index=moment_index,
            source_run_id=trace.run_id,
        )

        derive_from_episode = getattr(plugin.backend, "derive_task_from_episode", None)
        if not callable(derive_from_episode):
            self.statusBar().showMessage("Current environment does not support task derivation from episode moments")
            return

        resolved_options = derive_from_episode(source_task, trace, moment_index)
        if not isinstance(resolved_options, TaskDerivationOptions):
            self.statusBar().showMessage("Could not derive a task from the selected episode moment")
            return

        derivation_options = resolved_options
        derivation_options.derivation_reason = (
            derivation_options.derivation_reason or "start_from_episode_moment"
        )
        derivation_options.source_episode_id = trace.episode_id
        derivation_options.source_moment_index = moment_index
        derivation_options.source_run_id = trace.run_id

        derived_name = (
            f"{source_task.name} - From Episode {trace.episode_id} "
            f"Step {moment_index}"
        )
        derived_task = self._task_service.derive_task(
            source_task,
            name=derived_name,
            options=derivation_options,
            parent_task_id=source_task.task_id or source_task.name,
        )
        self._add_task_to_workspace(derived_task, select=True)
        self.tabs.setCurrentWidget(self.task_editor_tab)
        self.statusBar().showMessage(f"Derived task created from episode moment: {derived_task.name}")

    def _start_training(self, config: RunConfig) -> None:
        primary_task = self.task_history_view.selected_task() or self._current_task
        selected_tasks = self.task_history_view.selected_tasks()
        if primary_task is None:
            self.statusBar().showMessage("Cannot start training: no task selected")
            return
        if not selected_tasks:
            selected_tasks = [primary_task]
        elif primary_task in selected_tasks:
            selected_tasks = [primary_task] + [task for task in selected_tasks if task is not primary_task]

        config.evaluation_policy = self.evaluation_view.build_evaluation_policy()
        self.episode_view.clear_episodes()
        try:
            if len(selected_tasks) > 1:
                self._training_service.start_many(
                    selected_tasks,
                    config,
                    initial_checkpoint=self.history_view.selected_checkpoint(),
                    start_from_scratch=self.history_view.start_from_scratch_selected(),
                    run_in_background=True,
                )
                self.statusBar().showMessage(f"Parallel training started on {len(selected_tasks)} tasks")
            else:
                self._training_service.start(
                    primary_task,
                    config,
                    initial_checkpoint=self.history_view.selected_checkpoint(),
                    start_from_scratch=self.history_view.start_from_scratch_selected(),
                    run_in_background=True,
                )
                self.statusBar().showMessage("Training started")
        except RuntimeError as exc:
            self.statusBar().showMessage(str(exc))

    def _on_status_changed(self, status: TrainingStatus) -> None:
        self.training_view.set_status(status)

    def _on_metrics_updated(self, metrics: TrainingMetrics) -> None:
        self.training_view.set_metrics(metrics)

    def _on_run_metrics_updated(self, run_id: str, metrics: TrainingMetrics, task_name: str) -> None:
        self.training_view.set_run_metrics(run_id, task_name, metrics)

    def _on_breakpoint_triggered(self, event) -> None:
        self.training_view.set_breakpoint_event(event.message)
        self.statusBar().showMessage(event.message)

    def _on_episode_captured(self, trace: EpisodeTrace) -> None:
        self.episode_view.set_episode(trace, focus=self.tabs.currentWidget() is self.episode_tab)

    def _inspect_episode_from_history(self, trace: EpisodeTrace) -> None:
        plugin = self._current_plugin
        task_snapshot = trace.task_snapshot
        if task_snapshot is not None:
            try:
                plugin = self._registry.get_environment_plugin(task_snapshot.environment_id)
            except KeyError:
                plugin = self._current_plugin
        self.episode_view.set_context(plugin)
        self.episode_view.focus_episode(trace)
        self.tabs.setCurrentWidget(self.episode_tab)
        self.statusBar().showMessage(f"Inspecting episode {trace.episode_id} from {trace.run_id or 'unknown run'}")

    def _on_checkpoint_import_requested(self, checkpoint: Checkpoint) -> None:
        try:
            self._training_service.import_checkpoint(checkpoint)
        except RuntimeError as exc:
            self.statusBar().showMessage(str(exc))
            return
        self.statusBar().showMessage(f"Imported checkpoint: {checkpoint.checkpoint_id}")

    def _refresh_history_view(self) -> None:
        self.history_view.set_history(self._training_service.history_snapshot(deep=False))

    def _on_history_changed(self) -> None:
        self._refresh_history_view()

    def _add_task_to_workspace(self, task: TaskDefinition, *, select: bool) -> None:
        self._task_workspace.append(task)
        selected_workspace_index = len(self._task_workspace) - 1 if select else self._current_workspace_index()
        self._refresh_task_history_view(
            selected_workspace_index=selected_workspace_index,
            preserve_multi_selection=not select,
        )
        if select and selected_workspace_index is not None:
            self._select_task_index(
                selected_workspace_index,
                sync_task_history=True,
                preserve_graph_multi_selection=False,
            )

    def _select_task_index(
        self,
        index: int,
        *,
        sync_task_history: bool,
        preserve_graph_multi_selection: bool = False,
    ) -> None:
        if self._current_plugin is None:
            return
        if index < 0 or index >= len(self._task_workspace):
            return

        task = self._task_workspace[index]
        self._current_task = task
        self.task_editor_view.set_plugin_task(self._current_plugin, task)
        if sync_task_history:
            self.task_history_view.set_primary_workspace_index(
                index,
                preserve_multi_selection=preserve_graph_multi_selection,
                emit_signal=False,
            )

    def _current_workspace_index(self) -> int | None:
        if self._current_task is None:
            return None
        return self._workspace_index_for_task(self._current_task)

    def _refresh_task_history_view(
        self,
        *,
        selected_workspace_index: int | None,
        preserve_multi_selection: bool,
    ) -> None:
        self.task_history_view.set_tasks(self._task_workspace)
        self.evaluation_view.set_tasks(self._task_workspace)
        if selected_workspace_index is not None:
            self.task_history_view.set_primary_workspace_index(
                selected_workspace_index,
                preserve_multi_selection=preserve_multi_selection,
                emit_signal=False,
            )

    def _create_new_task(self) -> None:
        plugin = self._current_plugin
        if plugin is None:
            return
        if self._task_workspace:
            task = deepcopy(self._task_workspace[0])
            task.task_id = None
        else:
            task = self._task_service.create_default_task(plugin.plugin_id)
        task.name = self._unique_task_name(task.name)
        self._add_task_to_workspace(task, select=True)
        self.tabs.setCurrentWidget(self.task_editor_tab)
        self.statusBar().showMessage(f"Created task: {task.name}")

    def _on_save_project_requested(self) -> None:
        if self._save_project():
            assert self._project_store is not None
            self.statusBar().showMessage(f"Project saved: {self._project_store.project_path}")

    def _save_project(self) -> bool:
        if self._loading_project or self._project_store is None or self._current_plugin is None:
            return False

        state = ProjectState(
            environment_id=self._current_plugin.plugin_id,
            task_workspace=deepcopy(self._task_workspace),
            history=self._training_service.history_snapshot(),
        )
        try:
            self._project_store.save(state)
        except (OSError, TypeError, ValueError) as exc:
            self.statusBar().showMessage(f"Could not save project state: {exc}")
            return False
        return True

    def _workspace_index_for_task(self, task: TaskDefinition) -> int | None:
        for index, workspace_task in enumerate(self._task_workspace):
            if workspace_task is task:
                return index
        return None

    def _unique_task_name(self, base_name: str) -> str:
        existing = {task.name for task in self._task_workspace}
        if base_name not in existing:
            return base_name
        suffix = 2
        while True:
            candidate = f"{base_name} {suffix}"
            if candidate not in existing:
                return candidate
            suffix += 1

    def _wrap_tab(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll
