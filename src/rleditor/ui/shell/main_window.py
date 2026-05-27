from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
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
    DerivedTaskDefinition,
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
from rleditor.ui.interaction_logging import InteractionLogger
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
        interaction_logger: InteractionLogger | None = None,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._task_service = task_service
        self._training_service = training_service
        self._project_store = project_store
        self._interaction_logger = interaction_logger
        self._loading_project = True

        self._current_plugin: EnvironmentPlugin | None = None
        self._current_task: TaskDefinition | None = None
        self._task_workspace: list[TaskDefinition] = []
        self._imported_curriculum_queue: deque[tuple[TaskDefinition, RunConfig]] = deque()
        self._imported_curriculum_active = False
        self._imported_curriculum_waiting_for_step = False
        self._imported_curriculum_checkpoint_stop_pending = False
        self._imported_curriculum_completed_steps = 0
        self._status_busy_sources: set[str] = set()
        self._status_busy_frames = ("|", "/", "-", "\\")
        self._status_busy_frame_index = 0

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
        self.statusBar().setStyleSheet(
            """
            QStatusBar {
                font-size: 14px;
            }

            QLabel#StatusBusyIndicator {
                color: #0f766e;
                font-size: 15px;
                font-weight: 700;
            }
            """
        )
        self.status_busy_indicator = QLabel("", self)
        self.status_busy_indicator.setObjectName("StatusBusyIndicator")
        self.status_busy_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_busy_indicator.setFixedWidth(18)
        self.status_busy_indicator.setVisible(False)
        self._status_busy_timer = QTimer(self)
        self._status_busy_timer.setInterval(120)
        self._status_busy_timer.timeout.connect(self._advance_status_busy_indicator)
        self.statusBar().addWidget(self.status_busy_indicator)
        self.save_project_btn = QPushButton("Save Project", self)
        self.save_project_btn.setEnabled(self._project_store is not None)
        self.save_project_btn.setToolTip("Write the current task workspace and training history to disk.")
        self.statusBar().addPermanentWidget(self.save_project_btn)
        self.statusBar().showMessage("Ready")

    def _wire_signals(self) -> None:
        self.task_history_view.selection_changed.connect(self._on_task_history_selection_changed)
        self.task_history_view.create_task_requested.connect(self._create_new_task)
        self.task_history_view.edit_task_requested.connect(self._edit_task_from_history)
        self.task_history_view.copy_task_requested.connect(self._copy_task_from_history)
        self.task_editor_view.task_changed.connect(self._on_task_changed)
        self.episode_view.create_task_from_moment_requested.connect(self._on_derive_task_from_episode_moment)
        self.history_view.inspect_episode_requested.connect(self._inspect_episode_from_history)
        self.history_view.checkpoint_import_requested.connect(self._on_checkpoint_import_requested)
        self.history_view.checkpoint_evaluation_requested.connect(self._on_checkpoint_evaluation_requested)
        self.history_view.curriculum_import_requested.connect(self._on_curriculum_import_requested)
        self.history_view.training_run_config_selected.connect(self._on_training_run_config_selected)

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
        selected_checkpoint = self.history_view.selected_checkpoint()
        start_from_scratch = self.history_view.start_from_scratch_selected()
        try:
            if len(selected_tasks) > 1:
                self._training_service.start_many(
                    selected_tasks,
                    config,
                    initial_checkpoint=selected_checkpoint,
                    start_from_scratch=start_from_scratch,
                    run_in_background=True,
                )
                self.statusBar().showMessage(f"Parallel training started on {len(selected_tasks)} tasks")
                mode = "parallel"
            else:
                self._training_service.start(
                    primary_task,
                    config,
                    initial_checkpoint=selected_checkpoint,
                    start_from_scratch=start_from_scratch,
                    run_in_background=True,
                )
                self.statusBar().showMessage("Training started")
                mode = "single"
            self._log_interaction(
                "training_started",
                mode=mode,
                tasks=[
                    {
                        "task_id": task.task_id,
                        "name": task.name,
                        "environment_id": task.environment_id,
                    }
                    for task in selected_tasks
                ],
                algorithm=config.algorithm,
                max_steps=config.max_steps,
                max_episodes=config.max_episodes,
                max_steps_per_episode=config.max_steps_per_episode,
                seed=config.seed,
                initial_checkpoint_id=(
                    selected_checkpoint.checkpoint_id if selected_checkpoint is not None else None
                ),
                start_from_scratch=start_from_scratch,
            )
            self._set_status_busy("training", True)
        except RuntimeError as exc:
            self.statusBar().showMessage(str(exc))

    def _on_training_run_config_selected(self, config: RunConfig) -> None:
        self.training_view.set_config(config)
        self.statusBar().showMessage("Training config loaded from selected run")
        self._log_interaction(
            "training_config_loaded_from_history",
            algorithm=config.algorithm,
            max_steps=config.max_steps,
            max_episodes=config.max_episodes,
            max_steps_per_episode=config.max_steps_per_episode,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
        )

    def _on_status_changed(self, status: TrainingStatus) -> None:
        self.training_view.set_status(status)
        self._set_status_busy("training", status == TrainingStatus.RUNNING)
        if not self._imported_curriculum_active or not self._imported_curriculum_waiting_for_step:
            return
        if status == TrainingStatus.FINISHED:
            self._imported_curriculum_checkpoint_stop_pending = False
            self._imported_curriculum_waiting_for_step = False
            QTimer.singleShot(0, self._start_next_imported_curriculum_step)
        elif status == TrainingStatus.STOPPED:
            self._imported_curriculum_queue.clear()
            self._imported_curriculum_active = False
            self._imported_curriculum_waiting_for_step = False
            self._imported_curriculum_checkpoint_stop_pending = False
            self._set_status_busy("curriculum", False)
            self.statusBar().showMessage("Curriculum execution stopped")
        elif status == TrainingStatus.PAUSED:
            self._set_status_busy("curriculum", False)
            if self._imported_curriculum_checkpoint_stop_pending:
                self._imported_curriculum_checkpoint_stop_pending = False
                self.statusBar().showMessage("Curriculum checkpoint reached; stopping imported run")
                self._training_service.stop()

    def _on_metrics_updated(self, metrics: TrainingMetrics) -> None:
        self.training_view.set_metrics(metrics)

    def _on_run_metrics_updated(self, run_id: str, metrics: TrainingMetrics, task_name: str) -> None:
        self.training_view.set_run_metrics(run_id, task_name, metrics)

    def _on_breakpoint_triggered(self, event) -> None:
        self.training_view.set_breakpoint_event(event.message)
        self.statusBar().showMessage(event.message)
        self._log_interaction(
            "breakpoint_triggered",
            message=getattr(event, "message", ""),
            step=getattr(event, "step", None),
            episode=getattr(event, "episode", None),
        )
        actions = set(getattr(event.breakpoint, "actions", []))
        if (
            self._imported_curriculum_active
            and self._imported_curriculum_waiting_for_step
            and "checkpoint" in actions
            and "stop" not in actions
        ):
            if "pause" in actions:
                self._imported_curriculum_checkpoint_stop_pending = True
            else:
                self.statusBar().showMessage("Curriculum checkpoint reached; stopping imported run")
                self._training_service.stop()

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
            imported_checkpoint = self._training_service.import_checkpoint(checkpoint)
        except RuntimeError as exc:
            self.statusBar().showMessage(str(exc))
            return
        self.statusBar().showMessage(f"Imported checkpoint: {imported_checkpoint.checkpoint_id}")

    def _on_checkpoint_evaluation_requested(self, checkpoint: Checkpoint) -> None:
        policy = self.evaluation_view.build_evaluation_policy()
        if not policy:
            self.statusBar().showMessage("Cannot evaluate checkpoint: no evaluation task selected")
            return
        try:
            self._set_status_busy("evaluation", True)
            evaluated_checkpoint = self._training_service.evaluate_checkpoint(
                checkpoint.checkpoint_id,
                policy,
            )
        except RuntimeError as exc:
            self.statusBar().showMessage(str(exc))
            return
        finally:
            self._set_status_busy("evaluation", False)
        self.statusBar().showMessage(f"Evaluation completed for {evaluated_checkpoint.checkpoint_id}")
        self._log_interaction(
            "checkpoint_evaluated",
            checkpoint_id=evaluated_checkpoint.checkpoint_id,
            policy=policy,
        )

    def _on_curriculum_import_requested(self, payload: object) -> None:
        try:
            imported_tasks, curriculum_steps = self._curriculum_from_import_payload(payload)
        except ValueError as exc:
            self.statusBar().showMessage(f"Cannot import curriculum: {exc}")
            return
        if self._training_service.status in {TrainingStatus.RUNNING, TrainingStatus.PAUSED}:
            self.statusBar().showMessage("Cannot import curriculum while training is running")
            return

        self._task_workspace.extend(imported_tasks)
        selected_index = len(self._task_workspace) - len(imported_tasks) if imported_tasks else None
        self._refresh_task_history_view(
            selected_workspace_index=selected_index,
            preserve_multi_selection=False,
        )
        if selected_index is not None:
            self._select_task_index(
                selected_index,
                sync_task_history=True,
                preserve_graph_multi_selection=False,
            )

        self._imported_curriculum_queue = deque(curriculum_steps)
        self._imported_curriculum_active = True
        self._imported_curriculum_waiting_for_step = False
        self._imported_curriculum_checkpoint_stop_pending = False
        self._imported_curriculum_completed_steps = 0
        self._set_status_busy("curriculum", True)
        self.episode_view.clear_episodes()
        self._log_interaction(
            "curriculum_import_started",
            task_count=len(imported_tasks),
            step_count=len(curriculum_steps),
        )
        self._start_next_imported_curriculum_step()

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

    def _edit_task_from_history(self, workspace_index: int) -> None:
        if workspace_index < 0 or workspace_index >= len(self._task_workspace):
            return
        self._select_task_index(
            workspace_index,
            sync_task_history=True,
            preserve_graph_multi_selection=True,
        )
        self.tabs.setCurrentWidget(self.task_editor_tab)
        self.statusBar().showMessage(f"Editing task: {self._task_workspace[workspace_index].name}")

    def _copy_task_from_history(self, workspace_index: int) -> None:
        if workspace_index < 0 or workspace_index >= len(self._task_workspace):
            return
        task = deepcopy(self._task_workspace[workspace_index])
        task.name = self._unique_task_name(f"{task.name} Copy")
        task.task_id = None
        if isinstance(task, DerivedTaskDefinition):
            task.derived_task_id = None
        self._add_task_to_workspace(task, select=True)
        self.statusBar().showMessage(f"Copied task: {task.name}")

    def _curriculum_from_import_payload(
        self,
        payload: object,
    ) -> tuple[list[TaskDefinition], list[tuple[TaskDefinition, RunConfig]]]:
        if not isinstance(payload, dict):
            raise ValueError("Curriculum import must be a JSON object.")
        curriculum = payload.get("curriculum")
        if not isinstance(curriculum, dict):
            raise ValueError("Missing curriculum object.")
        raw_steps = curriculum.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("Curriculum must contain at least one step.")
        raw_environments = payload.get("environments")
        if not isinstance(raw_environments, list) or not raw_environments:
            raise ValueError("Curriculum must contain environments.")

        imported_tasks, tasks_by_ref = self._tasks_from_curriculum_environments(raw_environments)
        global_seed = self._optional_int(curriculum.get("seed"))
        evaluation_policy = self._evaluation_policy_from_curriculum_payload(payload, tasks_by_ref)

        curriculum_steps: list[tuple[TaskDefinition, RunConfig]] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict):
                raise ValueError(f"Curriculum step {index + 1} must be an object.")
            env_ref = raw_step.get("env_id", raw_step.get("environment_id"))
            task = tasks_by_ref.get(str(env_ref))
            if task is None:
                raise ValueError(f"Curriculum step {index + 1} references unknown env_id: {env_ref}")
            config = self._run_config_from_curriculum_step(raw_step, fallback_seed=global_seed)
            if evaluation_policy:
                config.evaluation_policy = deepcopy(evaluation_policy)
            curriculum_steps.append((task, config))

        return imported_tasks, curriculum_steps

    def _tasks_from_curriculum_environments(
        self,
        environments: list[object],
    ) -> tuple[list[TaskDefinition], dict[str, TaskDefinition]]:
        imported_tasks: list[TaskDefinition] = []
        tasks_by_ref: dict[str, TaskDefinition] = {}
        existing_names = {task.name for task in self._task_workspace}

        for index, raw_environment in enumerate(environments):
            if not isinstance(raw_environment, dict):
                raise ValueError(f"Environment entry {index + 1} must be an object.")
            task = self._task_from_curriculum_environment(raw_environment)
            task.name = self._unique_task_name_from_set(task.name, existing_names)
            existing_names.add(task.name)
            imported_tasks.append(task)

            raw_task_id = raw_environment.get("task_id", index)
            tasks_by_ref[str(raw_task_id)] = task
            tasks_by_ref[str(index)] = task

        return imported_tasks, tasks_by_ref

    def _task_from_curriculum_environment(self, payload: dict[str, Any]) -> TaskDefinition:
        metadata = dict(payload.get("metadata")) if isinstance(payload.get("metadata"), dict) else {}
        raw_task_id = payload.get("task_id")
        if raw_task_id is not None:
            metadata.setdefault("curriculum_env_id", raw_task_id)
        task_kwargs = {
            "environment_id": str(payload.get("environment_id", "")),
            "name": str(payload.get("task_name", payload.get("name", "Imported Task"))),
            "task_id": None,
            "config": self._dict_payload(payload.get("task_config", payload.get("config", {}))),
            "reward_config": self._dict_payload(payload.get("reward_config", {})),
            "termination_config": self._dict_payload(payload.get("termination_config", {})),
            "metadata": metadata,
        }
        if any(
            key in payload
            for key in {
                "derived_task_id",
                "parent_task_id",
                "derivation_reason",
                "source_episode_id",
                "source_moment_index",
                "source_run_id",
                "start_state",
                "goal_state",
            }
        ):
            return DerivedTaskDefinition(
                **task_kwargs,
                derived_task_id=payload.get("derived_task_id"),
                parent_task_id=payload.get("parent_task_id"),
                derivation_reason=payload.get("derivation_reason"),
                source_episode_id=self._optional_int(payload.get("source_episode_id")),
                source_moment_index=self._optional_int(payload.get("source_moment_index")),
                source_run_id=payload.get("source_run_id"),
                start_state=payload.get("start_state"),
                goal_state=payload.get("goal_state"),
            )
        return TaskDefinition(**task_kwargs)

    def _run_config_from_curriculum_step(
        self,
        step: dict[str, Any],
        *,
        fallback_seed: int | None,
    ) -> RunConfig:
        run_config_payload = step.get("run_config")
        if isinstance(run_config_payload, dict):
            config = RunConfig.from_dict(run_config_payload)
        else:
            hyperparameters = dict(step.get("hyperparameters")) if isinstance(step.get("hyperparameters"), dict) else {}
            if "epsilon_decay" in step:
                hyperparameters["epsilon_decay"] = step["epsilon_decay"]
            if "epsilon_min" in step:
                hyperparameters["epsilon_min"] = step["epsilon_min"]
            config = RunConfig.from_dict(
                {
                    "algorithm": self._normalize_curriculum_algorithm(step.get("algorithm", "q_learning")),
                    "seed": self._optional_int(step.get("seed"), fallback=fallback_seed),
                    "episode_trace_sample_rate": float(step.get("episode_trace_sample_rate", 0.0)),
                    "max_steps": self._optional_int(step.get("steps", step.get("max_steps"))),
                    "max_episodes": self._optional_int(step.get("max_episodes")),
                    "max_steps_per_episode": self._optional_int(
                        step.get("max_episode_length", step.get("max_steps_per_episode"))
                    ),
                    "max_duration_seconds": step.get("max_duration_seconds"),
                    "learning_rate": float(step.get("learning_rate", step.get("lr", 0.1))),
                    "gamma": float(step.get("discount_factor", step.get("gamma", 0.99))),
                    "epsilon": float(step.get("epsilon_start", step.get("epsilon", 0.1))),
                    "hyperparameters": hyperparameters,
                    "breakpoints": step.get("breakpoints", []),
                }
            )
        config.metadata["imported_curriculum_step"] = True
        for rule in config.breakpoints:
            actions = set(rule.actions)
            if (
                "checkpoint" in actions
                and "pause" not in actions
                and "stop" not in actions
            ):
                rule.actions.append("pause")
        return config

    def _evaluation_policy_from_curriculum_payload(
        self,
        payload: dict[str, Any],
        tasks_by_ref: dict[str, TaskDefinition],
    ) -> dict[str, object]:
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, dict):
            return {}
        env_ref = evaluation.get("evaluation_env", evaluation.get("evaluation_env_id"))
        task = tasks_by_ref.get(str(env_ref))
        if task is None:
            return {}
        episode_count = self._optional_int(evaluation.get("eval_episodes", evaluation.get("episode_count")))
        if episode_count is None or episode_count <= 0:
            return {}
        return {
            "task": task.to_dict(),
            "episode_count": episode_count,
            "max_steps_per_episode": self._optional_int(
                evaluation.get("max_episode_length", evaluation.get("max_steps_per_episode"))
            ),
            "seed": self._optional_int(evaluation.get("eval_seed", evaluation.get("seed"))),
            "trace_sample_rate": 1.0,
        }

    def _start_next_imported_curriculum_step(self) -> None:
        if not self._imported_curriculum_active:
            return
        if not self._imported_curriculum_queue:
            completed = self._imported_curriculum_completed_steps
            self._imported_curriculum_active = False
            self._imported_curriculum_waiting_for_step = False
            self._imported_curriculum_checkpoint_stop_pending = False
            self._set_status_busy("curriculum", False)
            self.statusBar().showMessage(f"Curriculum execution completed: {completed} step(s)")
            return

        task, config = self._imported_curriculum_queue.popleft()
        step_number = self._imported_curriculum_completed_steps + 1
        start_from_scratch = step_number == 1
        initial_checkpoint = None if start_from_scratch else self._latest_checkpoint()
        self._imported_curriculum_waiting_for_step = True
        try:
            self._training_service.start(
                task,
                config,
                initial_checkpoint=initial_checkpoint,
                start_from_scratch=start_from_scratch,
                run_in_background=True,
            )
        except RuntimeError as exc:
            self._imported_curriculum_queue.clear()
            self._imported_curriculum_active = False
            self._imported_curriculum_waiting_for_step = False
            self._imported_curriculum_checkpoint_stop_pending = False
            self._set_status_busy("curriculum", False)
            self.statusBar().showMessage(f"Curriculum execution failed: {exc}")
            return

        self._imported_curriculum_completed_steps = step_number
        workspace_index = self._workspace_index_for_task(task)
        if workspace_index is not None:
            self.task_history_view.set_primary_workspace_index(
                workspace_index,
                preserve_multi_selection=False,
                emit_signal=False,
            )
        self.statusBar().showMessage(
            f"Curriculum step {step_number} started on task: {task.name}"
        )

    def _set_status_busy(self, source: str, busy: bool) -> None:
        if busy:
            self._status_busy_sources.add(source)
        else:
            self._status_busy_sources.discard(source)

        if self._status_busy_sources:
            if self.status_busy_indicator.isHidden():
                self.status_busy_indicator.setText(self._status_busy_frames[self._status_busy_frame_index])
                self.status_busy_indicator.setVisible(True)
            if not self._status_busy_timer.isActive():
                self._status_busy_timer.start()
            return

        self._status_busy_timer.stop()
        self.status_busy_indicator.setVisible(False)
        self.status_busy_indicator.setText("")

    def _advance_status_busy_indicator(self) -> None:
        if not self._status_busy_sources:
            return
        self._status_busy_frame_index = (self._status_busy_frame_index + 1) % len(self._status_busy_frames)
        self.status_busy_indicator.setText(self._status_busy_frames[self._status_busy_frame_index])

    def _latest_checkpoint(self) -> Checkpoint | None:
        snapshot = self._training_service.history_snapshot(deep=False)
        return snapshot.checkpoints[-1] if snapshot.checkpoints else None

    def _normalize_curriculum_algorithm(self, value: object) -> str:
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "q_learning": "q_learning",
            "qlearning": "q_learning",
            "stable_baselines3_dqn": "sb3_dqn",
            "sb3_dqn": "sb3_dqn",
            "stable_baselines3_ppo": "sb3_ppo",
            "sb3_ppo": "sb3_ppo",
        }
        return aliases.get(normalized, str(value))

    def _optional_int(self, value: object, *, fallback: int | None = None) -> int | None:
        if value is None or value == "":
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _dict_payload(self, value: object) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _on_save_project_requested(self) -> None:
        if self._save_project():
            assert self._project_store is not None
            self.statusBar().showMessage(f"Project saved: {self._project_store.project_path}")
            self._log_interaction("project_saved", path=str(self._project_store.project_path))

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

    def _log_interaction(self, event: str, **payload: object) -> None:
        if self._interaction_logger is None:
            return
        self._interaction_logger.log(event, **payload)

    def _workspace_index_for_task(self, task: TaskDefinition) -> int | None:
        for index, workspace_task in enumerate(self._task_workspace):
            if workspace_task is task:
                return index
        return None

    def _unique_task_name(self, base_name: str) -> str:
        existing = {task.name for task in self._task_workspace}
        return self._unique_task_name_from_set(base_name, existing)

    def _unique_task_name_from_set(self, base_name: str, existing: set[str]) -> str:
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
