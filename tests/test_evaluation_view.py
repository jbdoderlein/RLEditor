from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.core.models import TaskDefinition
from rleditor.ui.views.evaluation_view import EvaluationView


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_evaluation_view_builds_policy_from_selected_task() -> None:
    _app()
    view = EvaluationView()
    main_task = TaskDefinition(environment_id="dummy_env", name="Main", task_id="task_main")
    eval_task = TaskDefinition(
        environment_id="dummy_env",
        name="Eval",
        task_id="task_eval",
        config={"difficulty": 3},
    )

    view.set_tasks([main_task, eval_task])
    view.task_combo.setCurrentIndex(1)
    view.episode_count_spin.setValue(7)
    view.max_steps_per_episode_spin.setValue(250)
    view.seed_spin.setValue(123)

    policy = view.build_evaluation_policy()

    assert policy["episode_count"] == 7
    assert policy["max_steps_per_episode"] == 250
    assert policy["seed"] == 123
    assert policy["trace_sample_rate"] == 1.0
    assert policy["task"]["task_id"] == "task_eval"
    assert policy["task"]["config"] == {"difficulty": 3}


def test_evaluation_view_defaults_to_100_steps_per_episode() -> None:
    _app()
    view = EvaluationView()
    task = TaskDefinition(environment_id="dummy_env", name="Eval", task_id="task_eval")

    view.set_tasks([task])

    assert view.build_evaluation_policy()["max_steps_per_episode"] == 100


def test_evaluation_view_import_task_button_emits_request() -> None:
    _app()
    view = EvaluationView()
    emitted: list[bool] = []
    view.import_task_requested.connect(lambda: emitted.append(True))

    view.import_task_button.click()

    assert emitted == [True]
