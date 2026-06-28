from __future__ import annotations

import random

import numpy as np


MAX_SEED = 2_147_483_647

_application_seed: int | None = None
_seed_sequence = random.Random()


def set_application_seed(seed: int | None) -> None:
    global _application_seed
    _application_seed = None if seed is None else int(seed) % (MAX_SEED + 1)
    if _application_seed is not None:
        _seed_sequence.seed(_application_seed)


def application_seed() -> int | None:
    return _application_seed


def derive_seed(offset: int = 0, *, base_seed: int | None = None) -> int | None:
    seed = application_seed() if base_seed is None else int(base_seed)
    if seed is None:
        return None
    return (seed + max(0, int(offset))) % (MAX_SEED + 1)


def next_seed() -> int | None:
    if _application_seed is None:
        return None
    return _seed_sequence.randrange(MAX_SEED + 1)


def seed_process() -> None:
    if _application_seed is None:
        return
    random.seed(_application_seed)
    np.random.seed(_application_seed)
