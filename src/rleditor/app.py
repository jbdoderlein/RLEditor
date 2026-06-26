from __future__ import annotations

import argparse
from pathlib import Path
import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from rleditor.application.persistence import ProjectStore
from rleditor.application.services import TaskService, TrainingService
from rleditor.plugins.registry import PluginRegistry, register_builtin_plugins
from rleditor.ui.app_icon import application_icon
from rleditor.ui.interaction_logging import InteractionLogger
from rleditor.ui.shell.main_window import MainWindow
from rleditor.ui.styles.theme import apply_theme


def _build_parser(plugin_ids: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rleditor")
    parser.add_argument(
        "--env",
        dest="environment_id",
        choices=plugin_ids,
        help="Environment plugin id to load at startup.",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List available environment plugin ids and exit.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        help="Project JSON path. Defaults to an environment-specific file under the user data directory.",
    )
    parser.add_argument(
        "--interaction-log",
        type=Path,
        help="Write UI interaction events to this JSON Lines log file.",
    )
    return parser


def _resolve_initial_plugin_id(
    registry: PluginRegistry,
    requested_plugin_id: str | None,
) -> str:
    plugins = registry.list_environment_plugins()
    if requested_plugin_id is not None:
        return requested_plugin_id
    if len(plugins) == 1:
        return plugins[0].plugin_id
    msg = "Multiple environment plugins are available; pass --env to choose one."
    raise ValueError(msg)


def run(argv: Sequence[str] | None = None) -> int:
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    plugin_ids = [plugin.plugin_id for plugin in registry.list_environment_plugins()]

    parser = _build_parser(plugin_ids)
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    if args.list_envs:
        for plugin in registry.list_environment_plugins():
            print(f"{plugin.plugin_id}\t{plugin.display_name}")
        return 0

    try:
        initial_plugin_id = _resolve_initial_plugin_id(registry, args.environment_id)
    except ValueError as exc:
        parser.error(str(exc))

    project_store = (
        ProjectStore(args.project)
        if args.project is not None
        else ProjectStore.default_for_environment(initial_plugin_id)
    )
    try:
        project_state = project_store.load()
    except (OSError, ValueError) as exc:
        parser.error(f"Could not load project state: {exc}")

    if project_state is not None and project_state.environment_id not in {"", initial_plugin_id}:
        parser.error(
            "Project environment mismatch: "
            f"project is for '{project_state.environment_id}', but --env selected '{initial_plugin_id}'."
        )

    QApplication.setApplicationName("RL Debug Studio")
    QApplication.setApplicationDisplayName("RL Debug Studio")
    QApplication.setDesktopFileName("rleditor")
    app = QApplication([sys.argv[0]])
    icon = application_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    apply_theme(app)
    try:
        interaction_logger = (
            InteractionLogger(args.interaction_log)
            if args.interaction_log is not None
            else None
        )
    except OSError as exc:
        parser.error(f"Could not open interaction log: {exc}")

    task_service = TaskService(registry)
    training_service = TrainingService(registry)
    if project_state is not None:
        training_service.load_history(project_state.history)

    window = MainWindow(
        registry=registry,
        task_service=task_service,
        training_service=training_service,
        initial_plugin_id=initial_plugin_id,
        initial_tasks=None if project_state is None else project_state.task_workspace,
        project_store=project_store,
        interaction_logger=interaction_logger,
    )
    if interaction_logger is not None:
        interaction_logger.attach(app, root_widget=window, training_service=training_service)
        interaction_logger.log(
            "application_started",
            environment_id=initial_plugin_id,
            project_path=str(project_store.project_path),
        )
    window.resize(1280, 820)
    window.show()

    try:
        return app.exec()
    finally:
        if interaction_logger is not None:
            interaction_logger.close()
