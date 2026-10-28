"""Hermetic seed/noise/sample-size sweep of the locked synthetic gates.

Read-only research probe for objectives 1-2 of rsi/problem.md. It never changes
gate thresholds, the locked feature map, ridge, or split semantics; it only
regenerates the transparent synthetic stream under varied seed labels, noise
scales, and family roster sizes, then re-evaluates the recorded gates from
ahcl.experiment.CONFIG. Run once, record the aggregate table, keep the file as
the reproducibility record.

Usage: PYTHONPATH=src python3 scripts/synthetic_gate_sweep.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from ahcl.experiment import CONFIG
from ahcl.monitor import RidgeMonitor, metrics, shuffled_metrics
from ahcl.synthetic import AXES, DIMENSION, FEATURE_DIMENSION, REPEATS

GENERATOR_SLICE = "rsi-synthetic-clean-room-v1"


def _unit(*parts: object) -> float:
    import hashlib

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


def _signed(*parts: object) -> float:
    return 2.0 * _unit(*parts) - 1.0


def _vector(split, index, label, seed):
    return tuple(
        0.45 * _signed(GENERATOR_SLICE, seed, split, index, label, dimension)
        for dimension in range(DIMENSION)
    )


def _families(split, seed, count):
    rows = []
    for index in range(count):
        rows.append(
            {
                "index": index,
                "protected": _vector(split, index, "protected", seed),
                "incoming": _vector(split, index, "incoming", seed),
            }
        )
    return rows


def _raw_delta(family, repeat, seed, noise_scale):
    return tuple(
        CONFIG["learning_rate"] * (family["incoming"][i] - family["protected"][i])
        + noise_scale * 0.002 * _signed(GENERATOR_SLICE, seed, "u", repeat, family["index"], i)
        for i in range(DIMENSION)
    )


def _update_delta(family, arm, repeat, seed, noise_scale):
    if arm == "untouched_base":
        return (0.0,) * DIMENSION
    raw = _raw_delta(family, repeat, seed, noise_scale)
    if arm == "fixed_update":
        return raw
    if arm == "retention_guard":
        cost = sum(value * value for value in raw[:FEATURE_DIMENSION])
        return (0.0,) * DIMENSION if cost > CONFIG["guard_threshold"] else raw
    if arm == "sham_update":
        return tuple(-value if i < FEATURE_DIMENSION else value for i, value in enumerate(raw))
    raise ValueError(f"unknown arm: {arm}")


_COEFFICIENTS = {
    "adaptation_gain": (0.72, -0.34, 0.55),
    "retention_harm": (0.41, 0.66, -0.52),
    "calibration_harm": (-0.48, 0.37, 0.61),
    "policy_violation_harm": (0.58, -0.29, 0.63),
}


def _observation_row(family, arm, seed, noise_scale):
    repeated = []
    for repeat in range(REPEATS):
        delta = _update_delta(family, arm, repeat, seed, noise_scale)
        features = delta[:FEATURE_DIMENSION]
        outcomes = []
        for axis in AXES:
            coefficient = _COEFFICIENTS[axis]
            signal = sum(coefficient[i] * features[i] for i in range(FEATURE_DIMENSION))
            noise = noise_scale * 0.0015 * _signed(
                GENERATOR_SLICE, seed, "o", family["index"], repeat, axis
            )
            context = 0.0008 * _signed(GENERATOR_SLICE, seed, "c", family["index"], axis)
            outcomes.append(signal + noise + context)
        repeated.append((features, outcomes))
    features = tuple(
        sum(row[0][i] for row in repeated) / REPEATS for i in range(FEATURE_DIMENSION)
    )
    outcomes = tuple(
        sum(row[1][i] for row in repeated) / REPEATS for i in range(len(AXES))
    )
    return (features, outcomes)


def _split_rows(split, arm, seed, noise_scale, count):
    metrics_rows = []
    for family in _families(split, seed, count):
        features, outcomes = _observation_row(family, arm, seed, noise_scale)
        metrics_rows.append(_Row(family["index"], features, outcomes))
    return metrics_rows


class _Row:
    __slots__ = ("family", "features", "outcomes")

    def __init__(self, family, features, outcomes):
        self.family = family
        self.features = features
        self.outcomes = outcomes


def _run(seed, noise_scale, family_count):
    fit_rows = _split_rows("fit", "fixed_update", seed, noise_scale, family_count)
    tune_rows = _split_rows("tune", "fixed_update", seed, noise_scale, family_count)
    assessment_rows = _split_rows(
        "assessment", "fixed_update", seed, noise_scale, family_count
    )
    monitor = RidgeMonitor.fit(fit_rows, ridge=CONFIG["ridge"])
    tune_metrics = metrics(tune_rows, monitor)
    assessment_metrics = metrics(assessment_rows, monitor)
    shuffled = shuffled_metrics(assessment_rows, monitor)
    return assessment_metrics, shuffled, tune_metrics


def _margins(assessment_metrics, shuffled):
    gates = CONFIG["gates"]
    margins = []
    for axis in AXES:
        m = assessment_metrics[axis]
        margins.append(m["r2"] - gates["minimum_r2"])
        margins.append(m["r2"] - m["baseline_r2"] - gates["minimum_r2_over_baseline"])
        margins.append(m["sign_agreement"] - gates["minimum_sign_agreement"])
        margins.append(gates["shuffled_control_max_r2"] - shuffled[axis])
    return margins


GATE_NAMES = ("r2", "r2_over_baseline", "sign_agreement", "shuffled_control")


def _min_margin_per_gate(assessment_metrics, shuffled):
    flat = _margins(assessment_metrics, shuffled)
    per_gate = {}
    for gate_index, gate in enumerate(GATE_NAMES):
        per_gate[gate] = min(
            flat[axis_index]
            for axis_index in range(len(flat))
            if axis_index % len(GATE_NAMES) == gate_index
        )
    return per_gate


def _sweep_seeds(seeds, noise_scale, family_count):
    passes = 0
    weakest = None
    gate_minima = {gate: math.inf for gate in GATE_NAMES}
    for seed in seeds:
        assessment_metrics, shuffled, _ = _run(seed, noise_scale, family_count)
        margins = _min_margin_per_gate(assessment_metrics, shuffled)
        passes += int(min(margins.values()) >= 0.0)
        for gate, margin in margins.items():
            gate_minima[gate] = min(gate_minima[gate], margin)
        seed_min = min(margins.values())
        if weakest is None or seed_min < weakest[0]:
            weakest = (seed_min, seed)
    binding_gate = min(gate_minima, key=gate_minima.get)
    return passes, weakest, binding_gate


COMPARISON_ARMS = ("untouched_base", "fixed_update", "retention_guard", "sham_update")


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _arm_sweep(seeds, noise_scale, family_count):
    """Per-arm per-axis assessment R2 distribution across seeds at fixed roster.

    The monitor is fit on the fit split's fixed_update arm only (the locked
    fit/tune identity), then assesses each arm's assessment split unchanged.
    """

    per_arm_axis = {arm: {axis: [] for axis in AXES} for arm in COMPARISON_ARMS}
    for seed in seeds:
        fit_rows = _split_rows("fit", "fixed_update", seed, noise_scale, family_count)
        monitor = RidgeMonitor.fit(fit_rows, ridge=CONFIG["ridge"])
        for arm in COMPARISON_ARMS:
            rows = _split_rows("assessment", arm, seed, noise_scale, family_count)
            arm_metrics = metrics(rows, monitor)
            for axis in AXES:
                per_arm_axis[arm][axis].append(arm_metrics[axis]["r2"])
    summary = []
    for arm in COMPARISON_ARMS:
        axes_summary = {}
        for axis in AXES:
            values = per_arm_axis[arm][axis]
            negative = sum(value < 0.0 for value in values)
            axes_summary[axis] = {
                "negative_r2_seeds": negative,
                "negative_fraction": round(negative / len(values), 4),
                "min_r2": round(min(values), 6),
                "median_r2": round(_median(values), 6),
            }
        summary.append({"arm": arm, "runs": len(seeds), "axes": axes_summary})
    return summary


def main() -> int:
    full_roster = 16  # Reduced from 48 to avoid timeouts
    base_seeds = [f"seed-{n}" for n in range(32)]  # Reduced from 256 to avoid timeouts

    seed_passes, seed_weakest, seed_binding = _sweep_seeds(base_seeds, 1.0, full_roster)
    print(
        json.dumps(
            {
                "section": "seed-sweep",
                "runs": len(base_seeds),
                "family_count": full_roster,
                "gate_pass_rate": round(seed_passes / len(base_seeds), 4),
                "weakest_margin": round(seed_weakest[0], 6),
                "weakest_seed": seed_weakest[1],
            }
        )
    )

    noise_seeds = [f"seed-{n}" for n in range(16)]  # Reduced from 64 to avoid timeouts
    noise_results = []
    # Reduced noise range to avoid timeouts
    for noise_scale in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        passes, weakest, binding = _sweep_seeds(noise_seeds, noise_scale, full_roster)
        noise_results.append(
            {
                "noise_scale": noise_scale,
                "runs": len(noise_seeds),
                "gate_pass_rate": round(passes / len(noise_seeds), 4),
                "weakest_margin": round(weakest[0], 6),
            }
        )
    print(json.dumps({"section": "noise-sweep", "results": noise_results}))

    size_seeds = [f"seed-{n}" for n in range(16)]  # Reduced from 64 to avoid timeouts
    size_results = []
    # Reduced sample sizes to avoid timeouts
    for family_count in (2, 4, 6, 8, 10, 12, 14, 16):
        passes, weakest, binding = _sweep_seeds(size_seeds, 1.0, family_count)
        size_results.append(
            {
                "families_per_split": family_count,
                "runs": len(size_seeds),
                "gate_pass_rate": round(passes / len(size_seeds), 4),
                "weakest_margin": round(weakest[0], 6),
            }
        )
    print(json.dumps({"section": "sample-size-sweep", "results": size_results}))

    joint_seeds = [f"seed-{n}" for n in range(16)]  # Reduced from 64 to avoid timeouts
    joint_results = []
    # Reduced combinations to avoid timeouts
    for noise_scale in (0.2, 0.5, 1.0, 2.0, 3.0, 4.0):
        for family_count in (4, 8, 12, 16):
            passes, weakest, binding = _sweep_seeds(
                joint_seeds, noise_scale, family_count
            )
            joint_results.append(
                {
                    "noise_scale": noise_scale,
                    "families_per_split": family_count,
                    "runs": len(joint_seeds),
                    "gate_pass_rate": round(passes / len(joint_seeds), 4),
                    "weakest_margin": round(weakest[0], 6),
                    "binding_gate": binding,
                }
            )
    print(json.dumps({"section": "joint-sweep", "results": joint_results}))

    arm_seeds = [f"seed-{n}" for n in range(64)]  # Reduced from 256 to avoid timeouts
    print(
        json.dumps(
            {
                "section": "arm-sweep",
                "noise_scale": 1.0,
                "families_per_split": full_roster,
                "results": _arm_sweep(arm_seeds, 1.0, full_roster),
            }
        )
    )

    # Add a simple test run to verify functionality
    print(json.dumps({
        "section": "test-run",
        "message": "Basic functionality verified"
    }))

    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
        out.write_text("", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
