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


def test_task_history_view_builds_tree_and_defaults_to_first_task() -> None:
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
    root_item = view.tree_widget.topLevelItem(0)
    assert root_item.text(0) == "Main Task"
    assert root_item.childCount() == 1
    assert root_item.child(0).text(0) == "Derived Task"


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


def test_task_history_view_emits_edit_and_copy_requests() -> None:
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
    )

    view = TaskHistoryView()
    view.set_tasks([base_task, derived_task])
    view.set_primary_workspace_index(1, preserve_multi_selection=False, emit_signal=False)

    edited: list[int] = []
    copied: list[int] = []
    view.edit_task_requested.connect(edited.append)
    view.copy_task_requested.connect(copied.append)

    view.edit_task_button.click()
    view.copy_task_button.click()

    assert edited == [1]
    assert copied == [1]


def test_task_history_view_emits_import_request() -> None:
    _app()
    view = TaskHistoryView()
    emitted: list[bool] = []
    view.import_task_requested.connect(lambda: emitted.append(True))

    view.import_task_button.click()

    assert emitted == [True]
