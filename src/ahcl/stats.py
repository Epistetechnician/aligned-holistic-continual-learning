"""Frozen statistical contracts for fit/tune/assessment.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence

from .power_design import PowerDesign


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
    power_scope: str = "single_comparison_approximation"
    power_validation: str = "familywise_holm_power_unvalidated"
    independent_unit: str = "document"
    families_per_document: int = 8
    power_endpoints: int = 4
    joint_power_target: float = 0.80
    power_design_status: str = "unvalidated"

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
            "power_scope": self.power_scope,
            "power_validation": self.power_validation,
            "independent_unit": self.independent_unit,
            "families_per_document": self.families_per_document,
            "power_endpoints": self.power_endpoints,
            "joint_power_target": self.joint_power_target,
            "power_design_status": self.power_design_status,
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
        minimum_effect_size=study.get("minimum_standardized_effect", -1.0),
        power_method=study.get("power_method", ""),
        power_scope=study.get("power_scope", ""),
        power_validation=study.get("power_validation", ""),
        independent_unit=study.get("independent_unit", ""),
        families_per_document=study.get("families_per_document", 0),
        power_endpoints=study.get("power_endpoints", 0),
        joint_power_target=study.get("joint_power_target", -1.0),
        power_design_status=study.get("power_design_status", ""),
    )
    plan.validate()
    expected = plan.as_config()
    if any(study.get(key) != value for key, value in expected.items()):
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
    if (
        not isinstance(plan.alpha, (int, float))
        or isinstance(plan.alpha, bool)
        or not math.isfinite(plan.alpha)
        or not isinstance(plan.power, (int, float))
        or isinstance(plan.power, bool)
        or not math.isfinite(plan.power)
        or not isinstance(plan.icc, (int, float))
        or isinstance(plan.icc, bool)
        or not math.isfinite(plan.icc)
        or not 0.0 < plan.alpha < 1.0
        or not 0.0 < plan.power <= 1.0
        or not 0.0 <= plan.icc < 1.0
    ):
        raise ValueError("invalid alpha, power, or ICC")
    if plan.multiplicity != "Holm":
        raise ValueError("only the frozen Holm correction is permitted")
    if (
        not isinstance(plan.minimum_effect_size, (int, float))
        or isinstance(plan.minimum_effect_size, bool)
        or not math.isfinite(plan.minimum_effect_size)
        or not 0.0 < plan.minimum_effect_size
        or plan.power_method != "two_sided_normal_icc_design_effect"
        or plan.power_scope != "single_comparison_approximation"
        or plan.power_validation != "familywise_holm_power_unvalidated"
        or not isinstance(plan.independent_unit, str)
        or plan.independent_unit != "document"
        or isinstance(plan.families_per_document, bool)
        or not isinstance(plan.families_per_document, int)
        or plan.families_per_document < 1
        or isinstance(plan.power_endpoints, bool)
        or not isinstance(plan.power_endpoints, int)
        or plan.power_endpoints < 1
        or not isinstance(plan.joint_power_target, (int, float))
        or isinstance(plan.joint_power_target, bool)
        or not math.isfinite(plan.joint_power_target)
        or not 0.0 < plan.joint_power_target <= 1.0
        or not isinstance(plan.power_design_status, str)
        or plan.power_design_status not in {"unvalidated", "validated"}
    ):
        raise ValueError("invalid frozen power method or effect size")
    counts = (plan.fit_families, plan.tune_families, plan.assessment_families, plan.repeats)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts):
        raise ValueError("all cohort counts must be positive")
    if achieved_power(plan) < plan.power:
        raise ValueError("locked cohort does not meet the declared power target")
    if plan.power_design_status == "validated" and document_power_lower_bound(plan) < plan.joint_power_target:
        raise ValueError("validated family-wise power claim exceeds the document-level design")


def document_power_lower_bound(plan: StatisticalPlan) -> float:
    """Return the conservative document-level Holm-family power lower bound."""

    counts = (plan.fit_families, plan.tune_families, plan.assessment_families)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts):
        raise ValueError("all cohort counts must be positive")
    if isinstance(plan.families_per_document, bool) or not isinstance(plan.families_per_document, int) or plan.families_per_document < 1:
        raise ValueError("families_per_document must be a positive integer")
    documents = min(plan.fit_families, plan.tune_families, plan.assessment_families) // plan.families_per_document
    if documents < 1:
        raise ValueError("cohorts contain no complete independent document")
    design = PowerDesign(
        effect_size=plan.minimum_effect_size,
        alpha=plan.alpha,
        target_power=plan.power,
        endpoints=plan.power_endpoints,
        repeats=plan.repeats,
        joint_target=plan.joint_power_target,
    )
    return design.joint_lower_bound(documents)
