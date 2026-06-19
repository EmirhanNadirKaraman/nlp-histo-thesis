"""Deterministic dev/test split for silver eval source cases."""
from __future__ import annotations

import hashlib
from typing import Literal, TypeVar

SplitName = Literal["dev", "test", "all"]

_T = TypeVar("_T")


def assign_split(case_id: str, dev_fraction: float = 0.8, seed: int = 42) -> str:
    """Hash-based split — stable under re-ordering and re-runs."""
    # Each case is hashed independently, so a case's dev/test membership never
    # shifts as the sample set grows (unlike shuffle-then-slice). seed and
    # dev_fraction are frozen so the dev/test partition is fixed across runs.
    h = int(hashlib.md5(f"{seed}:{case_id}".encode()).hexdigest(), 16)
    return "dev" if (h % 100) < int(dev_fraction * 100) else "test"


def filter_by_split(
    cases: list[_T],
    split: SplitName,
    *,
    case_id_attr: str = "case_id",
    dev_fraction: float = 0.8,
    seed: int = 42,
) -> list[_T]:
    if split == "all":
        return list(cases)
    return [
        c for c in cases
        if assign_split(getattr(c, case_id_attr), dev_fraction, seed) == split
    ]
