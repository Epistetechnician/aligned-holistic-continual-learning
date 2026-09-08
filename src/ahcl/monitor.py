"""Small fixed ridge monitor with no third-party dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .synthetic import AXES, Observation


def design_row(features: Sequence[float]) -> tuple[float, ...]:
    """Use a fixed seven-term singleton plus pairwise interaction map."""

    if len(features) != 3:
        raise ValueError("the locked feature map has exactly three coordinates")
    x, y, z = features
    return (1.0, x, y, z, x * y, x * z, y * z)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a positive-definite linear system by deterministic elimination."""

    size = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(size)]
    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[pivot_row][pivot]) < 1e-12:
            raise ValueError("singular ridge system")
        augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot]
        scale = augmented[pivot][pivot]
        augmented[pivot] = [value / scale for value in augmented[pivot]]
        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            if factor == 0.0:
                continue
            augmented[row] = [
                augmented[row][column] - factor * augmented[pivot][column]
                for column in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


@dataclass(frozen=True)
class RidgeMonitor:
    coefficients: tuple[tuple[float, ...], ...]
    ridge: float

    @classmethod
    def fit(cls, rows: Iterable[Observation], ridge: float = 1.0) -> "RidgeMonitor":
        rows = list(rows)
        if not rows:
            raise ValueError("cannot fit on empty rows")
        width = len(design_row(rows[0].features))
        gram = [[0.0] * width for _ in range(width)]
        targets = [[0.0] * width for _ in AXES]
        for row in rows:
            design = design_row(row.features)
            for i in range(width):
                for j in range(width):
                    gram[i][j] += design[i] * design[j]
                for axis_index, target in enumerate(row.outcomes):
                    targets[axis_index][i] += design[i] * target
        for i in range(width):
            gram[i][i] += ridge
        coefficients = tuple(tuple(_solve([line[:] for line in gram], target[:])) for target in targets)
        return cls(coefficients, ridge)

    def predict(self, features: Sequence[float]) -> tuple[float, ...]:
        design = design_row(features)
        return tuple(sum(weight * value for weight, value in zip(axis, design)) for axis in self.coefficients)


def _r2(observed: Sequence[float], predicted: Sequence[float]) -> float:
    mean = sum(observed) / len(observed)
    total = sum((value - mean) ** 2 for value in observed)
    residual = sum((value - estimate) ** 2 for value, estimate in zip(observed, predicted))
    return 0.0 if total == 0.0 else 1.0 - residual / total


def _sign(value: float) -> int:
    return 1 if value >= 0.0 else -1


def metrics(rows: Sequence[Observation], monitor: RidgeMonitor) -> dict[str, dict[str, float]]:
    """Compute aggregate prediction and sign metrics for every behavior axis."""

    result: dict[str, dict[str, float]] = {}
    for axis_index, axis in enumerate(AXES):
        observed = [row.outcomes[axis_index] for row in rows]
        predicted = [monitor.predict(row.features)[axis_index] for row in rows]
        mean_prediction = sum(observed) / len(observed)
        baseline = [mean_prediction] * len(observed)
        result[axis] = {
            "r2": _r2(observed, predicted),
            "baseline_r2": _r2(observed, baseline),
            "sign_agreement": sum(_sign(actual) == _sign(estimate) for actual, estimate in zip(observed, predicted)) / len(observed),
        }
    return result


def shuffled_metrics(rows: Sequence[Observation], monitor: RidgeMonitor) -> dict[str, float]:
    """Use a fixed family-level permutation as the causal feature control."""

    family_order = tuple(sorted(range(len(rows)), key=lambda index: rows[index].family))
    permutation = list(family_order)
    state = 0xA1C0A1C
    for index in range(len(permutation) - 1, 0, -1):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        swap = state % (index + 1)
        permutation[index], permutation[swap] = permutation[swap], permutation[index]
    values: dict[str, float] = {}
    for axis_index, axis in enumerate(AXES):
        observed = [row.outcomes[axis_index] for row in rows]
        predicted = [monitor.predict(rows[permutation[i]].features)[axis_index] for i in range(len(rows))]
        values[axis] = _r2(observed, predicted)
    return values
