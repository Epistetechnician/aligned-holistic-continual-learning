"""Frozen statistical contracts for fit/tune/assessment.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class StatisticalPlan:
    fit_families: int = 48
    tune_families: int = 48
    assessment_families: int = 48
    repeats: int = 3
    alpha: float = 0.05
    power: float = 0.80
    icc: float = 0.50
    multiplicity: str = "Holm"


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Return Holm step-down adjusted p-values in original order."""

    if any(value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[index] = running
    return tuple(adjusted)


def power_check(plan: StatisticalPlan) -> None:
    if not 0.0 < plan.power <= 1.0 or not 0.0 <= plan.icc < 1.0:
        raise ValueError("invalid power or ICC")
    if min(plan.fit_families, plan.tune_families, plan.assessment_families, plan.repeats) < 1:
        raise ValueError("all cohort counts must be positive")
