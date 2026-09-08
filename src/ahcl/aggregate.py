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


def _validate_roster(rows: Sequence[AggregateRow]) -> None:
    """Validate the family/repeat identity carried by an aggregate packet."""

    if not rows:
        raise AggregateError("cannot aggregate an empty packet")
    if any(not isinstance(row, AggregateRow) for row in rows):
        raise AggregateError("aggregate packet contains an invalid row")
    if any(
        not isinstance(row.family_id, str)
        or not row.family_id.strip()
        or not isinstance(row.split, str)
        or not row.split.strip()
        or type(row.repeat) is not int
        or row.repeat < 0
        for row in rows
    ):
        raise AggregateError("aggregate row identity is malformed")
    splits = {row.split for row in rows}
    if len(splits) != 1:
        raise AggregateError("aggregate packet mixes study splits")
    keys = [(row.family_id, row.repeat) for row in rows]
    if len(set(keys)) != len(keys):
        raise AggregateError("aggregate packet contains duplicate family-repeat rows")
    repeat_sets: dict[str, set[int]] = {}
    for row in rows:
        repeat_sets.setdefault(row.family_id, set()).add(row.repeat)
    observed = set.union(*repeat_sets.values())
    if observed != set(range(max(observed) + 1)):
        raise AggregateError("aggregate repeats must be contiguous from zero")
    if any(repeats != observed for repeats in repeat_sets.values()):
        raise AggregateError("aggregate families have incomplete repeat sets")


def _family_sort_key(family_id: str) -> tuple[int, int | str]:
    """Sort numeric family IDs numerically and opaque IDs lexically."""

    return (0, int(family_id)) if family_id.isdigit() else (1, family_id)


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
    *,
    baseline_means: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Recalculate metrics using constants locked from the fit cohort."""

    materialized = tuple(rows)
    _validate_roster(materialized)
    predictions = tuple(tuple(float(value) for value in predict(row.features)) for row in materialized)
    return reproduce_metrics_from_predictions(materialized, predictions, baseline_means=baseline_means)


def reproduce_metrics_from_predictions(
    rows: Iterable[AggregateRow],
    predictions: Iterable[Sequence[float]],
    *,
    baseline_means: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Recalculate aggregates with a fit-only baseline parameter vector."""

    materialized = tuple(rows)
    prediction_rows = tuple(tuple(float(value) for value in row) for row in predictions)
    _validate_roster(materialized)
    if len(materialized) != len(prediction_rows):
        raise AggregateError("row and prediction counts differ")
    width = len(materialized[0].outcomes)
    if width == 0 or any(len(row.outcomes) != width for row in materialized):
        raise AggregateError("outcome axes must have one fixed nonzero width")
    baseline = tuple(float(value) for value in baseline_means)
    if len(baseline) != width or any(not math.isfinite(value) for value in baseline):
        raise AggregateError("fit-only baseline means must be finite and match outcome width")
    if any(len(values) != width for values in prediction_rows):
        raise AggregateError("predictor output width does not match outcomes")
    if any(not math.isfinite(value) for row in materialized for value in row.outcomes):
        raise AggregateError("outcomes must be finite")
    if any(not math.isfinite(value) for row in prediction_rows for value in row):
        raise AggregateError("predictions must be finite")
    result: dict[str, dict[str, float]] = {}
    for axis in range(width):
        observed = [row.outcomes[axis] for row in materialized]
        predicted = [row[axis] for row in prediction_rows]
        constant = [baseline[axis]] * len(observed)
        result[str(axis)] = {
            "r2": _r2(observed, predicted),
            "baseline_r2": _r2(observed, constant),
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


def fixed_family_permutation(rows: Sequence[AggregateRow]) -> tuple[int, ...]:
    """Return a locked donor index for each row while preserving repeat blocks."""

    materialized = tuple(rows)
    _validate_roster(materialized)
    families = tuple(sorted({row.family_id for row in materialized}, key=_family_sort_key))
    indices = {
        (row.family_id, row.repeat): index
        for index, row in enumerate(materialized)
    }
    family_permutation = fixed_permutation(len(families))
    donor_family = {
        family: families[family_permutation[position]]
        for position, family in enumerate(families)
    }
    return tuple(indices[(donor_family[row.family_id], row.repeat)] for row in materialized)


def reproduce_shuffled_r2(
    rows: Iterable[AggregateRow],
    predict: Callable[[Any], Sequence[float]],
) -> dict[str, float]:
    """Recalculate the fixed shuffled-feature R² control."""

    materialized = tuple(rows)
    _validate_roster(materialized)
    permutation = fixed_family_permutation(materialized)
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
