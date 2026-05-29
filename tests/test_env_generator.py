from __future__ import annotations

import json

import pytest

from eval.env_generator import (
    export_rq1_curricula,
    hole_density,
    make_progressive_rq1_curriculum_dict,
    progressive_hole_curriculum_maps,
)


def _target_map() -> list[str]:
    return [
        "SFFF",
        "FHHF",
        "FFHF",
        "FFFG",
    ]


def test_progressive_hole_curriculum_maps_start_empty_and_end_at_target() -> None:
    target = _target_map()

    maps = progressive_hole_curriculum_maps(target, steps=4)

    assert maps[0] == [
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ]
    assert maps[-1] == target
    assert [sum(tile == "H" for row in map_desc for tile in row) for map_desc in maps] == [0, 1, 2, 3]
    target_holes = {
        (row_index, col_index)
        for row_index, row in enumerate(target)
        for col_index, tile in enumerate(row)
        if tile == "H"
    }
    for map_desc in maps:
        map_holes = {
            (row_index, col_index)
            for row_index, row in enumerate(map_desc)
            for col_index, tile in enumerate(row)
            if tile == "H"
        }
        assert map_holes <= target_holes


def test_progressive_hole_curriculum_requires_at_least_two_steps() -> None:
    with pytest.raises(ValueError, match="at least 2 steps"):
        progressive_hole_curriculum_maps(_target_map(), steps=1)


def test_make_progressive_rq1_curriculum_sets_episode_budget_per_step() -> None:
    payload = make_progressive_rq1_curriculum_dict(
        _target_map(),
        "RQ1 Task",
        curriculum_steps=4,
        episodes_per_step=123,
        max_steps_per_episode=44,
        evaluation_episodes=5,
        evaluation_max_steps_per_episode=99,
    )

    steps = payload["curriculum"]["steps"]
    environments = payload["environments"]

    assert payload["curriculum"]["size"] == 4
    assert [step["env_id"] for step in steps] == [0, 1, 2, 3]
    assert all(step["algorithm"] == "q_learning" for step in steps)
    assert all(step["max_episodes"] == 123 for step in steps)
    assert all(step["max_episode_length"] == 44 for step in steps)
    assert len(environments) == 4
    assert payload["evaluation"] == {
        "evaluation_env": 3,
        "eval_episodes": 5,
        "max_episode_length": 99,
    }
    assert environments[0]["task_config"]["map_desc"] == [
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ]
    assert environments[-1]["task_config"]["map_desc"] == _target_map()
    assert environments[-1]["metadata"]["target_hole_count"] == 3
    assert environments[-1]["task_config"]["hole_probability"] == pytest.approx(hole_density(_target_map()))


def test_export_rq1_curricula_writes_single_task_and_progressive_curricula(tmp_path) -> None:
    export_rq1_curricula(
        [_target_map()],
        output_dir=str(tmp_path),
        curriculum_steps=3,
        episodes_per_step=77,
        max_steps_per_episode=55,
        evaluation_episodes=6,
        evaluation_max_steps_per_episode=88,
    )

    task_payload = json.loads((tmp_path / "task_001.json").read_text(encoding="utf-8"))
    curriculum_payload = json.loads((tmp_path / "curriculum_001.json").read_text(encoding="utf-8"))

    assert task_payload["curriculum"]["steps"] == [{"env_id": "0", "algorithm": "q_learning"}]
    assert len(task_payload["environments"]) == 1
    assert task_payload["environments"][0]["task_config"]["map_desc"] == _target_map()

    assert curriculum_payload["curriculum"]["size"] == 3
    assert curriculum_payload["evaluation"] == {
        "evaluation_env": 2,
        "eval_episodes": 6,
        "max_episode_length": 88,
    }
    assert [step["max_episodes"] for step in curriculum_payload["curriculum"]["steps"]] == [77, 77, 77]
    assert [step["max_episode_length"] for step in curriculum_payload["curriculum"]["steps"]] == [55, 55, 55]
    assert curriculum_payload["environments"][0]["task_config"]["map_desc"] == [
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ]
    assert curriculum_payload["environments"][-1]["task_config"]["map_desc"] == _target_map()
