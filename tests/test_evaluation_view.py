from __future__ import annotations

import csv
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from rleditor.core.models import TaskDefinition
from rleditor.ui.views.evaluation_view import EvaluationResultsDialog, EvaluationView


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


def test_evaluation_view_builds_multiple_policies_from_checked_tasks() -> None:
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
    view._multi_task_checkboxes[0][0].setChecked(False)
    view._multi_task_checkboxes[1][0].setChecked(True)
    view.episode_count_spin.setValue(9)
    view.max_steps_per_episode_spin.setValue(45)
    view.seed_spin.setValue(77)

    policies = view.build_multiple_evaluation_policies()

    assert len(policies) == 1
    assert policies[0]["task"]["task_id"] == "task_eval"
    assert policies[0]["episode_count"] == 9
    assert policies[0]["max_steps_per_episode"] == 45
    assert policies[0]["seed"] == 77
    assert view.evaluate_multiple_button.isEnabled()


def test_evaluation_view_multiple_button_emits_request() -> None:
    _app()
    view = EvaluationView()
    view.set_tasks([TaskDefinition(environment_id="dummy_env", name="Eval", task_id="task_eval")])
    emitted: list[bool] = []
    view.evaluate_multiple_requested.connect(lambda: emitted.append(True))

    view.evaluate_multiple_button.click()

    assert emitted == [True]


def test_evaluation_results_dialog_saves_csv(tmp_path) -> None:
    _app()
    dialog = EvaluationResultsDialog(
        [
            {
                "task_name": "Eval",
                "environment_id": "dummy_env",
                "episode_count": 2,
                "success_rate": 0.5,
                "mean_reward": 1.25,
                "cumulative_reward": 2.5,
                "episode_length_mean": 3.0,
                "step": 6,
                "error": "",
            }
        ]
    )
    path = tmp_path / "results.csv"

    try:
        dialog.save_csv(path)
    finally:
        dialog.close()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == [
        "Task",
        "Environment",
        "Episodes",
        "Success Rate",
        "Mean Reward",
        "Cumulative Reward",
        "Episode Length",
        "Steps",
        "Error",
    ]
    assert rows[1] == ["Eval", "dummy_env", "2", "50.0%", "1.250", "2.500", "3.000", "6", ""]
