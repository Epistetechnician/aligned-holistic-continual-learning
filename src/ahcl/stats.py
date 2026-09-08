"""Frozen statistical contracts for fit/tune/assessment.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence


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
    minimum_effect_size: float = 0.50
    power_method: str = "two_sided_normal_icc_design_effect"

    def as_config(self) -> dict[str, Any]:
        return {
            "family_counts": {
                "fit": self.fit_families,
                "tune": self.tune_families,
                "assessment": self.assessment_families,
            },
            "repeats": self.repeats,
            "alpha": self.alpha,
            "power_target": self.power,
            "icc_assumption": self.icc,
            "multiplicity": self.multiplicity,
            "minimum_standardized_effect": self.minimum_effect_size,
            "power_method": self.power_method,
        }

    def validate(self) -> None:
        power_check(self)


FROZEN_CONTROLS = (
    "activation_only",
    "text_only",
    "exact_copy_noop",
    "shuffled_donor",
    "constant_donor",
    "matched_donor",
)

FROZEN_ARMS = (
    "sgd",
    "replay",
    "ewc",
    "behavioral_selection",
    "causal_monitor",
)


def validate_locked_study(config: dict[str, Any]) -> StatisticalPlan:
    """Validate the machine-readable study lock and return its plan."""

    study = config.get("study")
    if not isinstance(study, dict):
        raise ValueError("study lock is missing")
    plan = StatisticalPlan(
        fit_families=study.get("family_counts", {}).get("fit", 0),
        tune_families=study.get("family_counts", {}).get("tune", 0),
        assessment_families=study.get("family_counts", {}).get("assessment", 0),
        repeats=study.get("repeats", 0),
        alpha=study.get("alpha", -1.0),
        power=study.get("power_target", -1.0),
        icc=study.get("icc_assumption", -1.0),
        multiplicity=study.get("multiplicity", ""),
    )
    plan.validate()
    if plan.as_config() != {key: study[key] for key in plan.as_config()}:
        raise ValueError("study lock has unexpected values")
    if tuple(config.get("controls", ())) != FROZEN_CONTROLS:
        raise ValueError("control roster is not the frozen ordered roster")
    if tuple(config.get("arms", ())) != FROZEN_ARMS:
        raise ValueError("comparison arm roster is not the frozen ordered roster")
    if study.get("missingness") != "fail_closed" or study.get("attrition") != "zero":
        raise ValueError("missingness and attrition policies must fail closed")
    if study.get("prediction_lock") != "fit_predictor_and_tune_thresholds_before_assessment":
        raise ValueError("prediction-lock policy is not frozen")
    return plan


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Return Holm step-down adjusted p-values in original order."""

    if not p_values or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("p-values must be nonempty, finite, and lie in [0, 1]")
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[index] = running
    return tuple(adjusted)


def achieved_power(plan: StatisticalPlan) -> float:
    """Compute the locked two-sided normal-approximation power target."""

    design_effect = 1.0 + (plan.repeats - 1) * plan.icc
    effective_per_arm = min(plan.fit_families, plan.tune_families, plan.assessment_families)
    effective_per_arm = effective_per_arm * plan.repeats / design_effect
    noncentrality = plan.minimum_effect_size * (effective_per_arm / 2.0) ** 0.5
    critical = NormalDist().inv_cdf(1.0 - plan.alpha / 2.0)
    cdf = lambda value: 0.5 * (1.0 + math.erf(value / (2.0**0.5)))
    return (1.0 - cdf(critical - noncentrality)) + cdf(-critical - noncentrality)


def power_check(plan: StatisticalPlan) -> None:
    if not 0.0 < plan.alpha < 1.0 or not 0.0 < plan.power <= 1.0 or not 0.0 <= plan.icc < 1.0:
        raise ValueError("invalid alpha, power, or ICC")
    if plan.multiplicity != "Holm":
        raise ValueError("only the frozen Holm correction is permitted")
    if (
        not math.isfinite(plan.minimum_effect_size)
        or not 0.0 < plan.minimum_effect_size
        or plan.power_method != "two_sided_normal_icc_design_effect"
    ):
        raise ValueError("invalid frozen power method or effect size")
    if min(plan.fit_families, plan.tune_families, plan.assessment_families, plan.repeats) < 1:
        raise ValueError("all cohort counts must be positive")
    if achieved_power(plan) < plan.power:
        raise ValueError("locked cohort does not meet the declared power target")
