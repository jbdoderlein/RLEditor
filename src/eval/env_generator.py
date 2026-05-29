"""
eval/env_generator.py

Environment generation utilities for RQ1 and RQ2 evaluation.

RQ1: 30 target tasks at fixed difficulty (~30% hole density, 12x12).
     Each target task is exported as a progressive curriculum: start on an
     empty map, then add the target holes gradually until the final map.
RQ2: 1 fixed target task + 15 test environments (3 density levels x 5 maps).
     Structural dissimilarity d(A,B) is reported to verify test env diversity.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from eval.frozen_lake import FrozenLakeEnv, generate_random_map

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

SIZE        = 12                  # grid side length
MAX_DENSITY = 43 / (SIZE * SIZE)  # 0.2986 — largest discrete density < 30%
RQ1_N       = 30                  # number of target tasks for RQ1
RQ1_CURRICULUM_STEPS = 4          # maps per RQ1 curriculum, including empty + target
RQ1_EPISODES_PER_STEP = 1000      # training episodes for each RQ1 curriculum step
RQ1_MAX_STEPS_PER_EPISODE = 100   # max steps for each training episode
RQ1_EVAL_EPISODES = 100            # checkpoint evaluation episodes for each RQ1 curriculum
RQ1_EVAL_MAX_STEPS_PER_EPISODE = 100

RQ2_DENSITY_LEVELS = [0.10, 0.20, MAX_DENSITY]  # hole densities for RQ2 test envs
RQ2_N_PER_LEVEL    = 5                           # test maps per density level


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def hole_density(map_desc: List[str]) -> float:
    """Fraction of all n×n cells that are holes."""
    n = len(map_desc)
    holes = sum(c == "H" for row in map_desc for c in row)
    return holes / (n * n)


def map_dissimilarity(map_a: List[str], map_b: List[str]) -> float:
    """
    Normalized Hamming distance over non-S/G cells.

        d(A, B) = |{(i,j) : A[i,j] != B[i,j], (i,j) not in {S, G}}|
                  -------------------------------------------------------
                                        n^2 - 2

    Each cell is binary: H=1, F=0.  S at (0,0) and G at (n-1,n-1) excluded.
    Returns float in [0.0, 1.0].
    """
    n = len(map_a)
    assert len(map_b) == n, "Maps must be the same size."
    diffs = 0
    for i in range(n):
        for j in range(n):
            if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
                continue
            if (map_a[i][j] == "H") != (map_b[i][j] == "H"):
                diffs += 1
    return diffs / (n * n - 2)


# ---------------------------------------------------------------------------
# RQ1 — 30 target tasks at fixed difficulty
# ---------------------------------------------------------------------------

def generate_rq1_tasks(
    N: int = RQ1_N,
    size: int = SIZE,
    max_density: float = MAX_DENSITY,
    seed: int = 42,
) -> List[List[str]]:
    """
    Generate N unique target tasks for RQ1.

    All maps: hole density in [0.28, max_density), guaranteed valid path S→G,
    guaranteed mutually unique (no two identical maps).

    Args:
        N:           number of tasks (default 30).
        size:        grid side length (default 12).
        max_density: strict upper bound on hole density (default 43/144 ≈ 0.299).
        seed:        master seed for reproducibility.

    Returns:
        List of N maps, each a List[str] of length `size`.
    """
    rng   = np.random.default_rng(seed)
    tasks: List[List[str]] = []
    seen:  set = set()

    while len(tasks) < N:
        map_seed = int(rng.integers(0, 2**31))
        m   = generate_random_map(size=size, p=1.0 - max_density, seed=map_seed)
        key = tuple(m)
        if 0.28 <= hole_density(m) < max_density and key not in seen:
            seen.add(key)
            tasks.append(m)

    return tasks


# ---------------------------------------------------------------------------
# RQ2 — 1 target task + test environments at three hole density levels
# ---------------------------------------------------------------------------

def generate_rq2_target_task(
    size: int = SIZE,
    max_density: float = MAX_DENSITY,
    seed: int = 0,
) -> List[str]:
    """
    Generate the single fixed target task for RQ2 training.

    Hole density is in [0.28, max_density), guaranteed valid path, fixed seed.
    """
    rng = np.random.default_rng(seed)
    while True:
        map_seed = int(rng.integers(0, 2**31))
        m = generate_random_map(size=size, p=1.0 - max_density, seed=map_seed)
        if 0.28 <= hole_density(m) < max_density:
            return m


def generate_rq2_test_envs(
    density_levels: List[float] = RQ2_DENSITY_LEVELS,
    n_per_level: int = RQ2_N_PER_LEVEL,
    size: int = SIZE,
    seed: int = 100,
) -> Dict[float, List[List[str]]]:
    """
    Generate test environments at three hole density levels for RQ2.

    Each level produces n_per_level unique valid maps (uniqueness guaranteed
    within each level). These are unseen environments for frozen-policy eval.

    Args:
        density_levels: hole density targets (default [0.10, 0.20, ~0.30]).
        n_per_level:    maps per level (default 5).
        size:           grid side length (default 12).
        seed:           master seed for reproducibility.

    Returns:
        {density_level: [map_1, ..., map_n_per_level], ...}
    """
    rng       = np.random.default_rng(seed)
    test_envs: Dict[float, List[List[str]]] = {}

    for density in density_levels:
        maps: List[List[str]] = []
        seen: set = set()
        while len(maps) < n_per_level:
            map_seed = int(rng.integers(0, 2**31))
            m   = generate_random_map(size=size, p=1.0 - density, seed=map_seed)
            key = tuple(m)
            if key not in seen:
                seen.add(key)
                maps.append(m)
        test_envs[density] = maps

    return test_envs


# ---------------------------------------------------------------------------
# Dissimilarity report
# ---------------------------------------------------------------------------

def dissimilarity_report(
    target_map: List[str],
    test_envs: Dict[float, List[List[str]]],
) -> Dict:
    """
    For each density level compute:
      - d(test_i, target)   for each test map
      - d(test_i, test_j)   pairwise among the n_per_level maps

    Returns nested dict with scores, mean, std per level.
    """
    report: Dict = {}
    for density, maps in test_envs.items():
        vs_target = [map_dissimilarity(m, target_map) for m in maps]
        pairs     = list(combinations(range(len(maps)), 2))
        pairwise  = [map_dissimilarity(maps[i], maps[j]) for i, j in pairs]
        report[density] = {
            "vs_target": {"scores": vs_target,
                          "mean": float(np.mean(vs_target)),
                          "std":  float(np.std(vs_target))},
            "pairwise":  {"scores": pairwise,
                          "mean": float(np.mean(pairwise)),
                          "std":  float(np.std(pairwise))},
        }
    return report


def print_dissimilarity_report(report: Dict) -> None:
    print("\n=== Structural Dissimilarity Report (RQ2) ===")
    for density in sorted(report.keys()):
        vt = report[density]["vs_target"]
        pw = report[density]["pairwise"]
        print(f"\n  Density level {density*100:.0f}%")
        print(f"    vs target — mean: {vt['mean']:.3f}  std: {vt['std']:.3f}"
              f"  scores: {[round(s, 3) for s in vt['scores']]}")
        print(f"    pairwise  — mean: {pw['mean']:.3f}  std: {pw['std']:.3f}"
              f"  scores: {[round(s, 3) for s in pw['scores']]}")
    print()


# ---------------------------------------------------------------------------
# Curriculum JSON export
# ---------------------------------------------------------------------------

_DEFAULT_REWARD_CONFIG = {
    "tile:F": 0.0,
    "tile:G": 1.0,
    "tile:H": 0.0,
    "tile:S": 0.0,
}


def _empty_map_like(map_desc: List[str]) -> List[str]:
    """Return a no-hole map with the same size and S/G positions as map_desc."""
    n = len(map_desc)
    rows = [["F" for _ in range(n)] for _ in range(n)]
    start = (0, 0)
    goal = (n - 1, n - 1)
    for row_index, row in enumerate(map_desc):
        for col_index, tile in enumerate(row):
            if tile == "S":
                start = (row_index, col_index)
            elif tile == "G":
                goal = (row_index, col_index)
    rows[start[0]][start[1]] = "S"
    rows[goal[0]][goal[1]] = "G"
    return ["".join(row) for row in rows]


def _hole_positions(map_desc: List[str]) -> List[tuple[int, int]]:
    """Return target hole positions in deterministic row-major order."""
    positions: List[tuple[int, int]] = []
    for row_index, row in enumerate(map_desc):
        for col_index, tile in enumerate(row):
            if tile == "H":
                positions.append((row_index, col_index))
    return positions


def _map_with_first_holes(
    target_map: List[str],
    holes: List[tuple[int, int]],
    hole_count: int,
) -> List[str]:
    rows = [list(row) for row in _empty_map_like(target_map)]
    for row_index, col_index in holes[:hole_count]:
        rows[row_index][col_index] = "H"
    return ["".join(row) for row in rows]


def progressive_hole_curriculum_maps(
    target_map: List[str],
    steps: int = RQ1_CURRICULUM_STEPS,
) -> List[List[str]]:
    """
    Build a sequence of maps from easy to target.

    The first map is empty. The last map is exactly `target_map`. Intermediate
    maps add row-major subsets of the target holes, spread approximately evenly.
    """
    if steps < 2:
        raise ValueError("RQ1 progressive curricula require at least 2 steps.")

    holes = _hole_positions(target_map)
    total_holes = len(holes)
    maps: List[List[str]] = []
    for step_index in range(steps):
        if step_index == steps - 1:
            hole_count = total_holes
        else:
            hole_count = round((step_index / (steps - 1)) * total_holes)
        maps.append(_map_with_first_holes(target_map, holes, hole_count))
    return maps


def _make_environment_dict(
    map_desc: List[str],
    task_name: str,
    *,
    task_id: int | str = 0,
    metadata: Optional[Dict] = None,
) -> dict:
    n = len(map_desc)
    return {
        "environment_id": "frozen_lake",
        "task_id": task_id,
        "task_name": task_name,
        "metadata": metadata or {},
        "task_config": {
            "hole_probability": round(hole_density(map_desc), 6),
            "is_slippery": False,
            "map_desc": list(map_desc),
            "size": n,
            "success_rate": 0.333333,
        },
        "reward_config": _DEFAULT_REWARD_CONFIG,
        "termination_config": {},
    }


def _make_curriculum_dict(
    map_desc: List[str],
    task_name: str,
) -> dict:
    """
    Build a single-task curriculum dict matching the training system import format:

      {
        "curriculum": { "steps": [{"env_id": "0", "algorithm": "q_learning"}] },
        "environments": [{ environment definition }]
      }
    """
    return {
        "curriculum": {
            "steps": [
                {
                    "env_id": "0",
                    "algorithm": "q_learning",
                }
            ],
        },
        "environments": [
            _make_environment_dict(map_desc, task_name, task_id=0)
        ],
    }


def make_progressive_rq1_curriculum_dict(
    target_map: List[str],
    task_name: str,
    *,
    curriculum_steps: int = RQ1_CURRICULUM_STEPS,
    episodes_per_step: int = RQ1_EPISODES_PER_STEP,
    max_steps_per_episode: int = RQ1_MAX_STEPS_PER_EPISODE,
    evaluation_episodes: int = RQ1_EVAL_EPISODES,
    evaluation_max_steps_per_episode: int = RQ1_EVAL_MAX_STEPS_PER_EPISODE,
) -> dict:
    """
    Build a real RQ1 curriculum for one target map.

    The curriculum contains `curriculum_steps` environments. Step 1 is an empty
    map of the same size, and the final step is the original RQ1 target map.
    Every step uses default Q-learning parameters, except `max_episodes`, which
    is set from `episodes_per_step` to make episode-budget sweeps easy.
    """
    if episodes_per_step <= 0:
        raise ValueError("episodes_per_step must be positive.")
    if max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive.")
    if evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive.")
    if evaluation_max_steps_per_episode <= 0:
        raise ValueError("evaluation_max_steps_per_episode must be positive.")

    maps = progressive_hole_curriculum_maps(target_map, steps=curriculum_steps)
    target_hole_count = len(_hole_positions(target_map))
    environments = []
    curriculum_steps_payload = []
    for step_index, map_desc in enumerate(maps):
        step_number = step_index + 1
        hole_count = len(_hole_positions(map_desc))
        step_name = f"{task_name} - Curriculum Step {step_number:02d}/{len(maps):02d}"
        environments.append(
            _make_environment_dict(
                map_desc,
                step_name,
                task_id=step_index,
                metadata={
                    "curriculum_role": "rq1_progressive_holes",
                    "curriculum_step": step_number,
                    "curriculum_steps": len(maps),
                    "hole_count": hole_count,
                    "target_hole_count": target_hole_count,
                    "target_task_name": task_name,
                },
            )
        )
        curriculum_steps_payload.append(
            {
                "env_id": step_index,
                "algorithm": "q_learning",
                "max_episodes": episodes_per_step,
                "max_episode_length": max_steps_per_episode,
            }
        )

    return {
        "curriculum": {
            "size": len(curriculum_steps_payload),
            "steps": curriculum_steps_payload,
        },
        "environments": environments,
        "evaluation": {
            "evaluation_env": len(environments) - 1,
            "eval_episodes": evaluation_episodes,
            "max_episode_length": evaluation_max_steps_per_episode,
        },
    }


def export_rq1_curricula(
    tasks: List[List[str]],
    output_dir: str = "eval/curricula/rq1",
    curriculum_steps: int = RQ1_CURRICULUM_STEPS,
    episodes_per_step: int = RQ1_EPISODES_PER_STEP,
    max_steps_per_episode: int = RQ1_MAX_STEPS_PER_EPISODE,
    evaluation_episodes: int = RQ1_EVAL_EPISODES,
    evaluation_max_steps_per_episode: int = RQ1_EVAL_MAX_STEPS_PER_EPISODE,
) -> None:
    """
    Save each RQ1 target task both alone and as a progressive curriculum JSON.

    Output:
      eval/curricula/rq1/task_001.json … task_030.json
      eval/curricula/rq1/curriculum_001.json … curriculum_030.json
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, task in enumerate(tasks, start=1):
        name     = f"Frozen Lake {len(task)}x{len(task)} - RQ1 Task {i:03d}"
        task_payload = _make_curriculum_dict(task, name)
        task_out_path = os.path.join(output_dir, f"task_{i:03d}.json")
        with open(task_out_path, "w") as f:
            json.dump(task_payload, f, indent=2)

        curriculum_payload = make_progressive_rq1_curriculum_dict(
            task,
            name,
            curriculum_steps=curriculum_steps,
            episodes_per_step=episodes_per_step,
            max_steps_per_episode=max_steps_per_episode,
            evaluation_episodes=evaluation_episodes,
            evaluation_max_steps_per_episode=evaluation_max_steps_per_episode,
        )
        curriculum_out_path = os.path.join(output_dir, f"curriculum_{i:03d}.json")
        with open(curriculum_out_path, "w") as f:
            json.dump(curriculum_payload, f, indent=2)
    print(
        f"  Saved {len(tasks)} RQ1 tasks and progressive curricula "
        f"({curriculum_steps} steps, {episodes_per_step} episodes/step, "
        f"{max_steps_per_episode} max steps/episode, "
        f"{evaluation_episodes} eval episodes) → {output_dir}/"
    )


def export_rq2_curricula(
    target: List[str],
    test_envs: Dict[float, List[List[str]]],
    output_dir: str = "eval/curricula/rq2",
) -> None:
    """
    Save the RQ2 target task and all test environments as curriculum JSONs.

    Output:
      eval/curricula/rq2/target.json
      eval/curricula/rq2/test_10pct_env1.json … test_30pct_env5.json
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(target)

    # target task
    payload = _make_curriculum_dict(
        target,
        task_name=f"Frozen Lake {n}x{n} - RQ2 Target",
    )
    with open(os.path.join(output_dir, "target.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # test environments
    count = 0
    for density in sorted(test_envs.keys()):
        pct = int(round(density * 100))
        for j, m in enumerate(test_envs[density], start=1):
            name     = f"Frozen Lake {n}x{n} - RQ2 Test Density {density:.1f} Env {j}"
            payload  = _make_curriculum_dict(m, name)
            filename = f"test_density{density:.1f}_env{j}.json"
            with open(os.path.join(output_dir, filename), "w") as f:
                json.dump(payload, f, indent=2)
            count += 1

    print(f"  Saved RQ2 target + {count} test env curricula → {output_dir}/")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _render_map_rgb(map_desc: List[str]) -> np.ndarray:
    """
    Render a map to an RGB array using FrozenLakeEnv's pygame renderer.
    Returns a (H, W, 3) uint8 array.
    """
    env = FrozenLakeEnv(render_mode="rgb_array", desc=map_desc, is_slippery=False)
    env.reset()
    rgb = env.render()
    env.close()
    return rgb


def _draw_map(ax, map_desc: List[str], title: str = "") -> None:
    """Draw a single map on the given Axes using the game's own tile images."""
    ax.imshow(_render_map_rgb(map_desc))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def visualize_rq1_tasks(
    tasks: List[List[str]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot all N RQ1 tasks in a 6-column grid.
    Each map title shows its index and hole density.
    """
    n     = len(tasks)
    ncols = 6
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2 + 0.5))
    axes_flat = axes.flat

    for idx, (ax, task) in enumerate(zip(axes_flat, tasks)):
        _draw_map(ax, task, title=f"Task {idx + 1}  (d={hole_density(task):.2f})")
    for ax in list(axes_flat)[n:]:
        ax.axis("off")

    fig.suptitle(f"RQ1 — {n} Target Tasks  (12×12, hole density < 30%)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def visualize_rq2_envs(
    target: List[str],
    test_envs: Dict[float, List[List[str]]],
    report: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the RQ2 target task and all test environments.

    Layout:
      Row 0 : target map (centred, rest blank)
      Row 1+ : test maps per density level, titled with hole density and
               dissimilarity score vs target (if report is provided).
    """
    levels      = sorted(test_envs.keys())
    n_per_level = len(test_envs[levels[0]])
    nrows       = 1 + len(levels)
    ncols       = n_per_level

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2, nrows * 2 + 0.5))

    # Row 0 — target centred
    mid = ncols // 2
    for j in range(ncols):
        if j == mid:
            _draw_map(axes[0][j], target,
                      title=f"Target  (d={hole_density(target):.2f})")
        else:
            axes[0][j].axis("off")

    # Rows 1+ — test maps
    for row, density in enumerate(levels, start=1):
        maps = test_envs[density]
        axes[row][0].set_ylabel(f"{density*100:.0f}% holes", fontsize=8)
        for col, m in enumerate(maps):
            d_str = ""
            if report and density in report:
                d_val = report[density]["vs_target"]["scores"][col]
                d_str = f"\ndis={d_val:.2f}"
            _draw_map(axes[row][col],
                      m, title=f"d={hole_density(m):.2f}{d_str}")

    fig.suptitle("RQ2 — Target Task & Test Environments  (12×12)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RQ1/RQ2 FrozenLake evaluation curricula.")
    parser.add_argument(
        "--rq1-curriculum-steps",
        type=int,
        default=RQ1_CURRICULUM_STEPS,
        help=(
            "Number of maps in each RQ1 progressive curriculum, including "
            "the empty map and final target map."
        ),
    )
    parser.add_argument(
        "--rq1-episodes-per-step",
        type=int,
        default=RQ1_EPISODES_PER_STEP,
        help="Training episode budget for each RQ1 curriculum step.",
    )
    parser.add_argument(
        "--rq1-max-steps-per-episode",
        type=int,
        default=RQ1_MAX_STEPS_PER_EPISODE,
        help="Maximum steps per training episode for each generated RQ1 curriculum step.",
    )
    parser.add_argument(
        "--rq1-eval-episodes",
        type=int,
        default=RQ1_EVAL_EPISODES,
        help="Checkpoint evaluation episodes for each generated RQ1 curriculum.",
    )
    parser.add_argument(
        "--rq1-eval-max-steps-per-episode",
        type=int,
        default=RQ1_EVAL_MAX_STEPS_PER_EPISODE,
        help="Maximum steps per checkpoint evaluation episode for generated RQ1 curricula.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # --- RQ1 ---
    print("Generating RQ1 tasks …")
    rq1_tasks = generate_rq1_tasks()
    densities  = [hole_density(m) for m in rq1_tasks]
    print(f"  {len(rq1_tasks)} unique tasks generated.")
    print(f"  Hole density — mean: {np.mean(densities):.3f}  "
          f"std: {np.std(densities):.3f}  "
          f"range: [{min(densities):.3f}, {max(densities):.3f}]")
    visualize_rq1_tasks(rq1_tasks)

    # --- RQ2 ---
    print("\nGenerating RQ2 target task …")
    target = generate_rq2_target_task()
    print(f"  Target hole density: {hole_density(target):.3f}")

    print("\nGenerating RQ2 test environments …")
    test_envs = generate_rq2_test_envs()
    for lvl, maps in test_envs.items():
        actual = [hole_density(m) for m in maps]
        print(f"  Level {lvl*100:.0f}% — "
              f"actual density mean: {np.mean(actual):.3f}  "
              f"std: {np.std(actual):.3f}  "
              f"unique: {len(set(tuple(m) for m in maps))}/{len(maps)}")

    report = dissimilarity_report(target, test_envs)
    print_dissimilarity_report(report)
    visualize_rq2_envs(target, test_envs, report=report)

    # --- Export ---
    print("Exporting curriculum JSON files …")
    export_rq1_curricula(
        rq1_tasks,
        curriculum_steps=args.rq1_curriculum_steps,
        episodes_per_step=args.rq1_episodes_per_step,
        max_steps_per_episode=args.rq1_max_steps_per_episode,
        evaluation_episodes=args.rq1_eval_episodes,
        evaluation_max_steps_per_episode=args.rq1_eval_max_steps_per_episode,
    )
    export_rq2_curricula(target, test_envs)
