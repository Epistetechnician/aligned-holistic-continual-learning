"""End-to-end synthetic qualification and aggregate result construction."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .canonical import digest
from .monitor import RidgeMonitor, metrics, shuffled_metrics
from .synthetic import (
    ARMS,
    AXES,
    FEATURE_DIMENSION,
    GUARD_THRESHOLD,
    LEARNING_RATE,
    REPEATS,
    SPLITS,
    STATE_SLICE,
    families,
    observations,
    update_delta,
)

CONFIG = {
    "state_slice": STATE_SLICE,
    "family_counts": SPLITS,
    "arms": list(ARMS),
    "repeats": REPEATS,
    "feature_dimension": FEATURE_DIMENSION,
    "learning_rate": LEARNING_RATE,
    "guard_threshold": GUARD_THRESHOLD,
    "feature_map": ["bias", "f0", "f1", "f2", "f0*f1", "f0*f2", "f1*f2"],
    "ridge": 0.1,
    "primary_arm": "fixed_update",
    "primary_axes": list(AXES),
    "gates": {
        "minimum_sign_agreement": 0.80,
        "minimum_r2": 0.25,
        "minimum_r2_over_baseline": 0.20,
        "shuffled_control_max_r2": 0.20,
    },
}


def _round(value: float) -> float:
    return round(value, 12)


def run_experiment() -> dict[str, Any]:
    """Fit, lock, assess, and return only aggregate JSON-compatible data."""

    fit_rows = observations("fit", "fixed_update")
    tune_rows = observations("tune", "fixed_update")
    assessment_rows = observations("assessment", "fixed_update")
    monitor = RidgeMonitor.fit(fit_rows, ridge=CONFIG["ridge"])

    tune_metrics = metrics(tune_rows, monitor)
    prediction_lock = {
        "config_digest": digest(CONFIG),
        "coefficients": [[_round(value) for value in axis] for axis in monitor.coefficients],
        "ridge": monitor.ridge,
        "tune_metrics": tune_metrics,
    }
    prediction_lock_digest = digest(prediction_lock)
    assessment_metrics = metrics(assessment_rows, monitor)
    shuffled = shuffled_metrics(assessment_rows, monitor)

    arm_metrics = {}
    for arm in ARMS:
        rows = observations("assessment", arm)
        arm_metrics[arm] = metrics(rows, monitor)

    gates = CONFIG["gates"]
    primary_pass = all(
        assessment_metrics[axis]["sign_agreement"] >= gates["minimum_sign_agreement"]
        and assessment_metrics[axis]["r2"] >= gates["minimum_r2"]
        and assessment_metrics[axis]["r2"] - assessment_metrics[axis]["baseline_r2"] >= gates["minimum_r2_over_baseline"]
        for axis in AXES
    )
    controls_pass = all(shuffled[axis] <= gates["shuffled_control_max_r2"] for axis in AXES)
    status = "SyntheticCandidate" if primary_pass and controls_pass else "SyntheticNoCandidate"
    roster = {
        "families": sum(1 for _ in families()),
        "families_by_split": SPLITS,
        "repeats": REPEATS,
        "observations_per_arm": sum(SPLITS.values()) * REPEATS,
    }
    payload: dict[str, Any] = {
        "schema": "ahcl.synthetic.result.v1",
        "state_slice": STATE_SLICE,
        "config": CONFIG,
        "config_digest": digest(CONFIG),
        "roster": roster,
        "prediction_lock_digest": prediction_lock_digest,
        "tune_metrics": tune_metrics,
        "assessment_metrics": assessment_metrics,
        "shuffled_control_r2": shuffled,
        "arm_metrics": arm_metrics,
        "gates": gates,
        "gates_pass": {"primary": primary_pass, "controls": controls_pass},
        "status": status,
        "nonclaims": [
            "synthetic-only",
            "no model-bearing execution",
            "no mechanistic or alignment claim",
            "no real-stream or production claim",
        ],
    }
    payload["experiment_digest"] = digest(payload)
    return payload
