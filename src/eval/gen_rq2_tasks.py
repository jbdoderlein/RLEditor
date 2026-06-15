#!/usr/bin/env python3
"""
Generate RQ2 evaluation task JSON files.

Four 12×12 FrozenLake maps at four hole density levels: 10 / 20 / 30 / 40 %.
Each map is guaranteed to have a valid path from S to G.

Usage (from project root):
    python src/eval/gen_rq2_tasks.py

Output:
    src/eval/curricula/rq2/task_001_mutated_001.json  …  task_004_mutated_004.json
"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_SIZE: int = 12
IS_SLIPPERY: bool = False
SUCCESS_RATE: float = 1.0 / 3.0
DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "tile:F": 0.0,
    "tile:G": 1.0,
    "tile:H": 0.0,
    "tile:S": 0.0,
}

RQ1_HOLE_DENSITIES: list[float] = [0.10, 0.20, 0.30, 0.40]

RQ1_BASE_SEED: int = 121

RQ2_DISSIMILARITIES: list[float] = [0.2, 0.4, 0.6, 0.8]

_OUTPUT_DIR: Path = Path(__file__).parent / "curricula" / f"rq2_seed{RQ1_BASE_SEED}"


# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

def generate_random_map(
    size: int = 12,
    p: float = 0.20,
    seed: int | None = None,
) -> list[str]:
    """Return a solvable FrozenLake map with exactly round(p * size²) holes.

    Places exactly the target number of holes by sampling mutable positions
    without replacement, so actual density matches the target regardless of
    seed.  Retries until a map with at least one path from S to G is found.

    Args:
        size: Side length of the square grid.
        p:    Hole density — fraction of ALL cells (including S/G) that
              become holes.
        seed: RNG seed for reproducibility.

    Returns:
        A list of *size* strings, each of length *size*, using S / F / H / G.
    """
    rng = random.Random(seed)
    mutable = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r, c) not in ((0, 0), (size - 1, size - 1))
    ]
    n_holes = round(p * size * size)
    n_holes = min(n_holes, len(mutable))

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
    """BFS reachability check: True if G is reachable from S (holes are walls)."""
    size = len(map_desc)
    start = goal = None
    for r, row in enumerate(map_desc):
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
            elif cell == "G":
                goal = (r, c)
    if start is None or goal is None:
        return False

    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
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


def _mutable_positions(map_desc: list[str]) -> list[tuple[int, int]]:
    size = len(map_desc)
    return [
        (r, c)
        for r in range(size)
        for c in range(size)
        if map_desc[r][c] not in {"S", "G"}
    ]


def _fh_positions(
    map_desc: list[str],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    mutable_positions = _mutable_positions(map_desc)
    f_positions = [(r, c) for r, c in mutable_positions if map_desc[r][c] == "F"]
    h_positions = [(r, c) for r, c in mutable_positions if map_desc[r][c] == "H"]
    return f_positions, h_positions


def _max_permutations(map_desc: list[str]) -> int:
    f_positions, h_positions = _fh_positions(map_desc)
    return min(len(f_positions), len(h_positions))


def distance_map(map_desc_a: list[str], map_desc_b: list[str]) -> float:
    """Normalized H/F swap distance between two same-density maps.

    A single permutation swaps one F with one H, creating two changed cells.
    The returned distance is the number of required swaps divided by the
    maximum possible number of density-preserving swaps for the source map.
    """
    if len(map_desc_a) != len(map_desc_b):
        raise ValueError("Maps must have the same size.")

    mutable_positions = _mutable_positions(map_desc_a)
    max_permutations = _max_permutations(map_desc_a)
    if max_permutations == 0:
        return 0.0

    changed_cells = sum(
        map_desc_a[r][c] != map_desc_b[r][c]
        for r, c in mutable_positions
    )
    return (changed_cells / 2) / max_permutations


def mutate_map(map_desc: list[str], rng: random.Random) -> list[str]:
    """Swap H and F in a random mutable position, keeping S and G fixed.
    We mutate like this to maintain the same hole density."""
    f_positions, h_positions = _fh_positions(map_desc)
    if not f_positions or not h_positions:
        raise ValueError("No mutable positions to swap.")
    f_pos = rng.choice(f_positions)
    h_pos = rng.choice(h_positions)
    new_map = [list(row) for row in map_desc]
    new_map[f_pos[0]][f_pos[1]], new_map[h_pos[0]][h_pos[1]] = (
        new_map[h_pos[0]][h_pos[1]],
        new_map[f_pos[0]][f_pos[1]],
    )
    final_map = ["".join(row) for row in new_map]
    assert _actual_hole_density(final_map) == _actual_hole_density(map_desc), "Hole density changed after mutation."
    return final_map


def mutate_map_distance(map_desc: list[str], distance: float, seed: int | None = None) -> list[str]:
    """Mutate the map directly to the requested normalized swap distance."""
    if not 0 <= distance <= 1:
        raise ValueError("Distance must be between 0 and 1.")

    rng = random.Random(seed)
    f_positions, h_positions = _fh_positions(map_desc)
    max_permutations = min(len(f_positions), len(h_positions))
    if max_permutations == 0:
        raise ValueError("No mutable positions to swap.")

    n_permutations = round(distance * max_permutations)
    f_to_h = rng.sample(f_positions, n_permutations)
    h_to_f = rng.sample(h_positions, n_permutations)

    new_map = [list(row) for row in map_desc]
    for r, c in f_to_h:
        new_map[r][c] = "H"
    for r, c in h_to_f:
        new_map[r][c] = "F"

    final_map = ["".join(row) for row in new_map]
    assert _actual_hole_density(final_map) == _actual_hole_density(map_desc), "Hole density changed after mutation."
    return final_map


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _actual_hole_density(map_desc: list[str]) -> float:
    """Fraction of ALL cells (including S and G) that are holes."""
    size = len(map_desc)
    return sum(row.count("H") for row in map_desc) / (size * size)

def _build_task_payload(
    task_name: str,
    map_desc: list[str],
    hole_probability: float,
) -> dict[str, Any]:
    return {
        "curriculum": {
            "steps": [{"env_id": "0", "algorithm": "q_learning"}],
        },
        "environments": [
            {
                "environment_id": "frozen_lake",
                "task_id": 0,
                "task_name": task_name,
                "metadata": {},
                "task_config": {
                    "hole_probability": round(hole_probability, 6),
                    "is_slippery": IS_SLIPPERY,
                    "map_desc": map_desc,
                    "size": len(map_desc),
                    "success_rate": round(SUCCESS_RATE, 6),
                },
                "reward_config": dict(DEFAULT_REWARD_CONFIG),
                "termination_config": {},
            },
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, density in enumerate(RQ1_HOLE_DENSITIES, start=1):
        seed = RQ1_BASE_SEED
        map_desc = generate_random_map(size=GRID_SIZE, p=density, seed=seed)
        actual = _actual_hole_density(map_desc)
        task_name = f"Frozen Lake {GRID_SIZE}x{GRID_SIZE} - RQ1 Task {i:03d}"
        for j, dissimilarity in enumerate(RQ2_DISSIMILARITIES, start=1):
            mutated_map = mutate_map_distance(map_desc, dissimilarity, seed=seed)
            mutated_actual = _actual_hole_density(mutated_map)
            actual_dissimilarity = distance_map(map_desc, mutated_map)
            mutated_task_name = f"Frozen Lake {GRID_SIZE}x{GRID_SIZE} - RQ1 Task {i:03d} - Mutated {j:03d}"
            payload = _build_task_payload(mutated_task_name, mutated_map, mutated_actual)
            out = _OUTPUT_DIR / f"task_{i:03d}_{int(dissimilarity * 100):03d}.json"
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                f"[RQ2] {out.name}  target_density={density:.0%}  actual_density={mutated_actual:.6f}"
                f"  target_dissimilarity={dissimilarity:.2f}"
                f"  actual_dissimilarity={actual_dissimilarity:.3f}  seed={seed}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
