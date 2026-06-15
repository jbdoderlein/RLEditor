#!/usr/bin/env python3
"""
Generate RQ1 evaluation task JSON files.

Four 12×12 FrozenLake maps at four hole density levels: 10 / 20 / 30 / 40 %.
Each map is guaranteed to have a valid path from S to G.

Usage (from project root):
    python src/eval/gen_rq1_tasks.py

Output:
    src/eval/curricula/rq1/task_001.json  …  task_004.json
"""

from __future__ import annotations

import json
import random
from collections import deque
from math import comb
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

RQ1_BASE_SEED: int = 57

_OUTPUT_DIR: Path = Path(__file__).parent / "curricula" / "rq1"


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


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _actual_hole_density(map_desc: list[str]) -> float:
    """Fraction of ALL cells (including S and G) that are holes."""
    size = len(map_desc)
    return sum(row.count("H") for row in map_desc) / (size * size)


def _count_monotonic_paths(map_desc: list[str]) -> tuple[int, int]:
    """Count valid monotonic paths (right/down only) from S to G.

    Returns (valid_paths, total_paths) where total_paths is the count if
    there were no holes — C(rows+cols-2, cols-1).  The ratio gives an
    intuitive "path survival rate" as hole density increases.
    """
    rows = cols = len(map_desc)
    dp = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if map_desc[r][c] == "H":
                dp[r][c] = 0
            elif r == 0 and c == 0:
                dp[r][c] = 1
            elif r == 0:
                dp[r][c] = dp[r][c - 1]
            elif c == 0:
                dp[r][c] = dp[r - 1][c]
            else:
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
    return dp[rows - 1][cols - 1], comb(rows + cols - 2, cols - 1)


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
        payload = _build_task_payload(task_name, map_desc, actual)

        valid_paths, total_paths = _count_monotonic_paths(map_desc)
        survival = valid_paths / total_paths if total_paths else 0.0

        out = _OUTPUT_DIR / f"task_{i:03d}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[RQ1] {out.name}  target={density:.0%}  actual_density={actual:.6f}"
            f"  paths={valid_paths}/{total_paths} ({survival:.2%})  seed={seed}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
