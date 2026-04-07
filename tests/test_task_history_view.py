from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.core.models import DerivedTaskDefinition, TaskDefinition
from rleditor.ui.views.task_history_view import TaskHistoryView


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_task_history_view_builds_lineage_and_defaults_to_first_task() -> None:
    _app()
    base_task = TaskDefinition(
        environment_id="dummy_env",
        name="Main Task",
        task_id="task_main",
    )
    derived_task = DerivedTaskDefinition(
        environment_id="dummy_env",
        name="Derived Task",
        task_id="task_derived",
        parent_task_id="task_main",
        derivation_reason="focus_region",
    )

    view = TaskHistoryView()
    view.set_tasks([base_task, derived_task])

    assert view.selected_task() is base_task
    assert view.selected_tasks() == [base_task]
    assert len(view.graph_widget._edges) == 1


def test_task_history_view_supports_multi_selection_with_primary_task() -> None:
    _app()
    base_task = TaskDefinition(
        environment_id="dummy_env",
        name="Main Task",
        task_id="task_main",
    )
    second_task = TaskDefinition(
        environment_id="dummy_env",
        name="Independent Task",
        task_id="task_other",
    )

    view = TaskHistoryView()
    view.set_tasks([base_task, second_task])
    view.toggle_workspace_index_selection(1, emit_signal=False)

    assert view.selected_task() is second_task
    assert view.selected_tasks() == [base_task, second_task]
    assert view.primary_workspace_index() == 1
