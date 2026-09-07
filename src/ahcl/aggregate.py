"""Independent aggregate calculations for validated row-level packets.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
This module intentionally does not import the production monitor. A validator
can use it to recalculate metrics from a short-lived row packet before raw
rows are deleted from external custody.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


class AggregateError(ValueError):
    """Raised when an aggregate packet is incomplete or non-finite."""


@dataclass(frozen=True)
class AggregateRow:
    family_id: str
    split: str
    repeat: int
    features: Any
    outcomes: tuple[float, ...]


def _r2(observed: Sequence[float], predicted: Sequence[float]) -> float:
    if not observed or len(observed) != len(predicted):
        raise AggregateError("observed and predicted rows must have equal nonzero length")
    mean = sum(observed) / len(observed)
    total = sum((value - mean) ** 2 for value in observed)
    residual = sum((value - estimate) ** 2 for value, estimate in zip(observed, predicted))
    return 0.0 if total == 0.0 else 1.0 - residual / total


def _sign(value: float) -> int:
    return 1 if value >= 0.0 else -1


def reproduce_metrics(
    rows: Iterable[AggregateRow],
    predict: Callable[[Any], Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Recalculate R², constant-baseline R², and sign agreement."""

    materialized = tuple(rows)
    if not materialized:
        raise AggregateError("cannot aggregate an empty packet")
    width = len(materialized[0].outcomes)
    if width == 0 or any(len(row.outcomes) != width for row in materialized):
        raise AggregateError("outcome axes must have one fixed nonzero width")
    predictions = tuple(tuple(float(value) for value in predict(row.features)) for row in materialized)
    if any(len(values) != width for values in predictions):
        raise AggregateError("predictor output width does not match outcomes")
    if any(not math.isfinite(value) for row in materialized for value in row.outcomes):
        raise AggregateError("outcomes must be finite")
    if any(not math.isfinite(value) for row in predictions for value in row):
        raise AggregateError("predictions must be finite")
    result: dict[str, dict[str, float]] = {}
    for axis in range(width):
        observed = [row.outcomes[axis] for row in materialized]
        predicted = [row[axis] for row in predictions]
        baseline = [sum(observed) / len(observed)] * len(observed)
        result[str(axis)] = {
            "r2": _r2(observed, predicted),
            "baseline_r2": _r2(observed, baseline),
            "sign_agreement": sum(_sign(a) == _sign(p) for a, p in zip(observed, predicted)) / len(observed),
        }
    return result


def fixed_permutation(size: int) -> tuple[int, ...]:
    """Return the locked deterministic permutation used by the control."""

    if size < 0:
        raise AggregateError("size must be non-negative")
    permutation = list(range(size))
    state = 0xA1C0A1C
    for index in range(size - 1, 0, -1):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        swap = state % (index + 1)
        permutation[index], permutation[swap] = permutation[swap], permutation[index]
    return tuple(permutation)


def reproduce_shuffled_r2(
    rows: Iterable[AggregateRow],
    predict: Callable[[Any], Sequence[float]],
) -> dict[str, float]:
    """Recalculate the fixed shuffled-feature R² control."""

    materialized = tuple(rows)
    if not materialized:
        raise AggregateError("cannot aggregate an empty packet")
    permutation = fixed_permutation(len(materialized))
    predictions = tuple(tuple(float(value) for value in predict(materialized[index].features)) for index in permutation)
    width = len(materialized[0].outcomes)
    if width == 0 or any(len(row.outcomes) != width for row in materialized):
        raise AggregateError("outcome axes must have one fixed nonzero width")
    if any(len(values) != width for values in predictions):
        raise AggregateError("predictor output width does not match outcomes")
    if any(not math.isfinite(value) for row in materialized for value in row.outcomes):
        raise AggregateError("outcomes must be finite")
    if any(not math.isfinite(value) for row in predictions for value in row):
        raise AggregateError("predictions must be finite")
    return {
        str(axis): _r2(
            [row.outcomes[axis] for row in materialized],
            [values[axis] for values in predictions],
        )
        for axis in range(width)
    }
