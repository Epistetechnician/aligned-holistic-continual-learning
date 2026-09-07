"""Deterministic synthetic continual-learning stream.

The generator is deliberately small and transparent. It creates no files and
keeps family rows in memory; the runner publishes only aggregate metrics.
State slice: aligned-holistic-continual-learning-clean-room-v1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

STATE_SLICE = "aligned-holistic-continual-learning-clean-room-v1"
SPLITS = {"fit": 48, "tune": 48, "assessment": 48}
ARMS = ("untouched_base", "fixed_update", "retention_guard", "sham_update")
REPEATS = 3
DIMENSION = 8
FEATURE_DIMENSION = 3
LEARNING_RATE = 0.18
GUARD_THRESHOLD = 0.0018
AXES = (
    "adaptation_gain",
    "retention_harm",
    "calibration_harm",
    "policy_violation_harm",
)


@dataclass(frozen=True)
class Family:
    split: str
    index: int
    protected: tuple[float, ...]
    incoming: tuple[float, ...]


@dataclass(frozen=True)
class Observation:
    split: str
    family: int
    arm: str
    features: tuple[float, ...]
    outcomes: tuple[float, ...]


def _unit(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


def _signed(*parts: object) -> float:
    return 2.0 * _unit(*parts) - 1.0


def _vector(split: str, index: int, label: str) -> tuple[float, ...]:
    return tuple(0.45 * _signed(STATE_SLICE, split, index, label, dimension) for dimension in range(DIMENSION))


def families() -> Iterable[Family]:
    """Yield the fixed 144-family fit/tune/assessment roster."""

    for split, count in SPLITS.items():
        for index in range(count):
            yield Family(split, index, _vector(split, index, "protected"), _vector(split, index, "incoming"))


def _raw_delta(family: Family, repeat: int) -> tuple[float, ...]:
    return tuple(
        LEARNING_RATE * (family.incoming[i] - family.protected[i])
        + 0.002 * _signed(STATE_SLICE, family.split, family.index, repeat, "update-noise", i)
        for i in range(DIMENSION)
    )


def update_delta(family: Family, arm: str, repeat: int) -> tuple[float, ...]:
    """Apply one predeclared update intervention."""

    if arm == "untouched_base":
        return (0.0,) * DIMENSION
    raw = _raw_delta(family, repeat)
    if arm == "fixed_update":
        return raw
    if arm == "retention_guard":
        protected_cost = sum(value * value for value in raw[:FEATURE_DIMENSION])
        return (0.0,) * DIMENSION if protected_cost > GUARD_THRESHOLD else raw
    if arm == "sham_update":
        return tuple(-value if i < FEATURE_DIMENSION else value for i, value in enumerate(raw))
    raise ValueError(f"unknown arm: {arm}")


_COEFFICIENTS = {
    "adaptation_gain": (0.72, -0.34, 0.55),
    "retention_harm": (0.41, 0.66, -0.52),
    "calibration_harm": (-0.48, 0.37, 0.61),
    "policy_violation_harm": (0.58, -0.29, 0.63),
}


def observation(family: Family, arm: str, repeat: int) -> Observation:
    delta = update_delta(family, arm, repeat)
    features = delta[:FEATURE_DIMENSION]
    outcomes = []
    for axis_index, axis in enumerate(AXES):
        coefficient = _COEFFICIENTS[axis]
        signal = sum(coefficient[i] * features[i] for i in range(FEATURE_DIMENSION))
        noise = 0.0015 * _signed(STATE_SLICE, family.split, family.index, repeat, axis, "outcome-noise")
        context = 0.0008 * _signed(STATE_SLICE, family.split, family.index, axis, "context")
        outcomes.append(signal + noise + context)
    return Observation(family.split, family.index, arm, features, tuple(outcomes))


def observations(split: str, arm: str) -> list[Observation]:
    """Return repeat-collapsed rows for one split and arm."""

    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    rows: list[Observation] = []
    for family in families():
        if family.split != split:
            continue
        repeated = [observation(family, arm, repeat) for repeat in range(REPEATS)]
        mean_features = tuple(sum(row.features[i] for row in repeated) / REPEATS for i in range(FEATURE_DIMENSION))
        mean_outcomes = tuple(sum(row.outcomes[i] for row in repeated) / REPEATS for i in range(len(AXES)))
        rows.append(Observation(split, family.index, arm, mean_features, mean_outcomes))
    return rows
