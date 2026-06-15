"""
Shared configuration for all RQ1/baseline curriculum generation scripts.

Import from here rather than duplicating constants across scripts:
    from eval_config import (GRID_SIZE, RQ1_SEED, MAX_EPISODES, ...)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Environment
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

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

RQ1_SEED: int = 121        # seed for gen_rq1_tasks.py
BASELINE_SEED: int = 42    # seed for gen_baseline_curricula.py

# ---------------------------------------------------------------------------
# Task densities
# ---------------------------------------------------------------------------

RQ1_HOLE_DENSITIES: list[float] = [0.10, 0.20, 0.30, 0.40]

# ---------------------------------------------------------------------------
# Curriculum structure
# ---------------------------------------------------------------------------

# Number of intermediate steps per target density (target step not included)
N_INTERMEDIATE: dict[float, int] = {0.10: 2, 0.20: 4, 0.30: 9, 0.40: 14}

# Training episodes per curriculum step (all steps including target)
MAX_EPISODES: dict[float, int] = {0.10: 1000, 0.20: 2000, 0.30: 3000, 0.40: 4000}

MAX_EPISODE_LENGTH: int = 100
