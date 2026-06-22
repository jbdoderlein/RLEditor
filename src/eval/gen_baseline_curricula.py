#!/usr/bin/env python3
"""
Generate baseline curriculum JSON files for each RQ1 task.

Two baselines per task:
  random         — intermediate maps in sampled (random) order, then target
  random_ordered — same intermediate maps sorted by ascending hole density, then target

Both baselines share identical intermediate maps; only the ordering differs.
Intermediate maps have hole densities sampled uniformly from [0, target_density].

N_CURRICULA variants are generated per baseline type, numbered v01..vNN.
Variant v01 reproduces the original seed (BASELINE_SEED), so it is identical
to the legacy unsuffixed files.  Existing files are never overwritten.

Prerequisites:
    Run gen_rq1_tasks.py first.

Usage (from project root):
    python src/eval/gen_baseline_curricula.py [--n-curricula N]

Output:
    src/eval/curricula/baselines/rq1_seed<N>/task_001_random_v01.json
    src/eval/curricula/baselines/rq1_seed<N>/task_001_random_v02.json
    ...
    src/eval/curricula/baselines/rq1_seed<N>/task_001_random_ordered_v01.json
    ...
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

from eval_config import (
    BASELINE_SEED,
    DEFAULT_REWARD_CONFIG,
    GRID_SIZE,
    IS_SLIPPERY,
    MAX_EPISODE_LENGTH,
    MAX_EPISODES,
    N_CURRICULA,
    N_INTERMEDIATE,
    RQ1_HOLE_DENSITIES,
    RQ1_SEED,
    SUCCESS_RATE,
)

_RQ1_DIR: Path = Path(__file__).parent / "curricula" / f"rq1_seed{RQ1_SEED}"
_OUTPUT_DIR: Path = (
    Path(__file__).parent / "curricula" / "baselines" / f"rq1_seed{RQ1_SEED}"
)


# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

def generate_random_map(size: int, p: float, seed: int | None) -> list[str]:
    """Return a solvable map with exactly round(p * size²) holes."""
    rng = random.Random(seed)
    mutable = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r, c) not in ((0, 0), (size - 1, size - 1))
    ]
    n_holes = min(round(p * size * size), len(mutable))
    while True:
        hole_set = set(rng.sample(mutable, n_holes))
        grid = [
            [
                "S" if (r, c) == (0, 0)
                else "G" if (r, c) == (size - 1, size - 1)
                else ("H" if (r, c) in hole_set else "F")
                for c in range(size)
            ]
            for r in range(size)
        ]
        map_desc = ["".join(row) for row in grid]
        if _has_path(map_desc):
            return map_desc


def _has_path(map_desc: list[str]) -> bool:
    size = len(map_desc)
    visited: set[tuple[int, int]] = {(0, 0)}
    queue: deque[tuple[int, int]] = deque([(0, 0)])
    goal = (size - 1, size - 1)
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < size
                and 0 <= nc < size
                and (nr, nc) not in visited
                and map_desc[nr][nc] != "H"
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def _hole_density(map_desc: list[str]) -> float:
    size = len(map_desc)
    return sum(row.count("H") for row in map_desc) / (size * size)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _variant_seeds(variant: int) -> tuple[int, int]:
    """Return (density_rng_seed, map_base_seed) for the given 1-indexed variant.

    variant=1 reproduces the original BASELINE_SEED behaviour exactly:
      density_rng_seed = BASELINE_SEED
      map_seed(j)      = BASELINE_SEED + j + 1
    """
    k = variant - 1
    return BASELINE_SEED + k, BASELINE_SEED + k * 1000


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _build_payload(
    task_idx: int,
    target_density: float,
    curriculum_maps: list[list[str]],
    baseline_type: str,
    max_eps: int,
    variant: int,
) -> dict[str, Any]:
    n_steps = len(curriculum_maps)
    label = baseline_type.replace("_", " ").title()

    steps: list[dict[str, Any]] = []
    environments: list[dict[str, Any]] = []

    for env_id, map_desc in enumerate(curriculum_maps):
        is_target = env_id == n_steps - 1
        step_num = env_id + 1
        role = "rq1_target" if is_target else f"rq1_{baseline_type}_intermediate"

        n_holes = sum(row.count("H") for row in map_desc)
        density = n_holes / (GRID_SIZE * GRID_SIZE)

        suffix = " (Target)" if is_target else ""
        task_name = (
            f"Frozen Lake {GRID_SIZE}x{GRID_SIZE} - RQ1 Task {task_idx:03d}"
            f" - {label} Baseline v{variant:02d}"
            f" - Step {step_num:02d}/{n_steps:02d}{suffix}"
        )

        steps.append({
            "env_id": env_id,
            "algorithm": "q_learning",
            "max_episodes": max_eps,
            "max_episode_length": MAX_EPISODE_LENGTH,
        })

        environments.append({
            "environment_id": "frozen_lake",
            "task_id": env_id,
            "task_name": task_name,
            "metadata": {
                "curriculum_role": role,
                "curriculum_step": step_num,
                "curriculum_steps": n_steps,
                "hole_count": n_holes,
                "hole_density": round(density, 6),
                "target_hole_density": target_density,
                "baseline_type": baseline_type,
                "baseline_seed": BASELINE_SEED,
                "variant": variant,
            },
            "task_config": {
                "hole_probability": round(density, 6),
                "is_slippery": IS_SLIPPERY,
                "map_desc": map_desc,
                "size": GRID_SIZE,
                "success_rate": round(SUCCESS_RATE, 6),
            },
            "reward_config": dict(DEFAULT_REWARD_CONFIG),
            "termination_config": {},
        })

    return {
        "curriculum": {
            "size": n_steps,
            "steps": steps,
        },
        "environments": environments,
        "evaluation": {
            "evaluation_env": n_steps - 1,
            "eval_episodes": 100,
            "max_episode_length": MAX_EPISODE_LENGTH,
        },
    }


# ---------------------------------------------------------------------------
# RQ1 loader
# ---------------------------------------------------------------------------

def _load_rq1_tasks() -> list[tuple[int, float, list[str]]]:
    """Return (task_idx, nominal_density, map_desc) for each RQ1 task."""
    tasks = []
    for i, density in enumerate(RQ1_HOLE_DENSITIES, start=1):
        path = _RQ1_DIR / f"task_{i:03d}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"RQ1 file not found: {path}\nRun gen_rq1_tasks.py first."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        map_desc: list[str] = payload["environments"][0]["task_config"]["map_desc"]
        tasks.append((i, density, map_desc))
    return tasks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="gen_baseline_curricula")
    parser.add_argument(
        "--n-curricula",
        type=int,
        default=N_CURRICULA,
        help=f"Number of variants to generate per baseline type (default: {N_CURRICULA}).",
    )
    args = parser.parse_args()
    n_curricula: int = args.n_curricula

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rq1_tasks = _load_rq1_tasks()

    for task_idx, target_density, target_map in rq1_tasks:
        n_inter = N_INTERMEDIATE[target_density]
        max_eps = MAX_EPISODES[target_density]

        for variant in range(1, n_curricula + 1):
            density_seed, map_base_seed = _variant_seeds(variant)

            # Sample intermediate densities
            rng = random.Random(density_seed)
            inter_densities = [rng.uniform(0.0, target_density) for _ in range(n_inter)]

            # Generate intermediate maps (shared by both baselines for this variant)
            inter_maps: list[list[str]] = [
                generate_random_map(GRID_SIZE, d, map_base_seed + j + 1)
                for j, d in enumerate(inter_densities)
            ]

            inter_densities_pct = [f"{d:.1%}" for d in inter_densities]

            for baseline_type, ordered_maps in (
                ("random", inter_maps + [target_map]),
                ("random_ordered", sorted(inter_maps, key=_hole_density) + [target_map]),
            ):
                out = _OUTPUT_DIR / f"task_{task_idx:03d}_{baseline_type}_v{variant:02d}.json"
                if out.exists():
                    print(f"  [SKIP] {out.name} already exists")
                    continue

                payload = _build_payload(
                    task_idx, target_density, ordered_maps, baseline_type, max_eps, variant
                )
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(
                    f"[BASELINE] task_{task_idx:03d}  target={target_density:.0%}"
                    f"  type={baseline_type}  v{variant:02d}"
                    f"  steps={n_inter + 1}  max_episodes={max_eps}"
                    f"  inter_densities={inter_densities_pct}"
                    f"  density_seed={density_seed}"
                )

    print("Done.")


if __name__ == "__main__":
    main()
