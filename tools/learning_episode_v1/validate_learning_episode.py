"""Independent deterministic verifier for learning-episode receipts.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
Governance slice: global-experiment-assurance-lifecycle-v1.

This module deliberately does not import the receipt producer. It regenerates
the fixed synthetic cohorts and reconstructs selection, updates, locks, and
aggregate assessment with a separate implementation. This is local contract
verification, not independent scientific review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ahcl.canonical import digest, file_digest

STATE_SLICE = "aligned-holistic-continual-learning-integrated-stack-v2"
GOVERNANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
SCHEMA = "ahcl.learning-episode-receipt.v1"
CONFIG: dict[str, Any] = {
    "protocol": "ahcl.synthetic-learning-episode.v1",
    "state_slice": STATE_SLICE,
    "governance_slice": GOVERNANCE_SLICE,
    "seed": "ahcl-learning-episode-v1-fixed-seed",
    "feature_dimension": 3,
    "fit_candidates": 24,
    "tune_examples": 16,
    "assessment_examples": 32,
    "maximum_updates": 8,
    "tune_checkpoints": [0, 2, 4, 6, 8],
    "learning_rate": 0.08,
    "novelty_weight": 0.05,
    "target_weights": [1.25, -0.8, 0.45],
    "interaction_weight": 0.15,
    "noise_scale": 0.03,
    "selector": "absolute-prediction-error-plus-nearest-neighbor-novelty",
    "control": "fixed-id-order-replay",
    "metric": "mean-squared-error",
}
NONCLAIMS = [
    "synthetic-only deterministic fixture",
    "no model weights, provider calls, remote workers, or user data",
    "mechanism adapter sees known toy simulator coordinates",
    "no real continual-learning, mechanistic, alignment, or superiority claim",
    "assessment seal is an in-process ordering gate, not cryptographic secrecy",
    "no independent review or production authorization",
]


class ValidationError(ValueError):
    """Raised when a receipt differs from the independently reconstructed run."""


def _uniform(split: str, index: int, coordinate: int) -> float:
    raw = f"{CONFIG['seed']}:{split}:{index}:{coordinate}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return (value / ((1 << 64) - 1)) * 2.0 - 1.0


def _make_cohort(split: str, size: int) -> list[dict[str, Any]]:
    cohort = []
    for item_index in range(size):
        vector = tuple(_uniform(split, item_index, axis) for axis in range(3))
        a, b, c = vector
        label = (
            1.25 * a - 0.8 * b + 0.45 * c + 0.15 * a * b
            + 0.03 * _uniform(split, item_index, 3)
        )
        cohort.append({"id": f"{split}-{item_index:03d}", "features": vector, "outcome": label})
    return cohort


def _state_id(state: Sequence[float]) -> str:
    return digest([float(value) for value in state])


def _estimate(state: Sequence[float], row: Mapping[str, Any]) -> float:
    return sum(weight * value for weight, value in zip(state, row["features"]))


def _loss(rows: Sequence[Mapping[str, Any]], state: Sequence[float]) -> float:
    residuals = [(_estimate(state, row) - row["outcome"]) ** 2 for row in rows]
    return sum(residuals) / len(residuals)


def _gradient_step(state: Sequence[float], row: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    residual = row["outcome"] - _estimate(state, row)
    movement = [0.08 * residual * component for component in row["features"]]
    return [old + change for old, change in zip(state, movement)], movement


def _distance(candidate: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> float:
    if not history:
        return 0.0
    norms = []
    for previous in history:
        total = sum((x - y) ** 2 for x, y in zip(candidate["features"], previous["features"]))
        norms.append(total ** 0.5)
    return min(norms)


def _priority(row: Mapping[str, Any], state: Sequence[float], history: Sequence[Mapping[str, Any]]) -> float:
    residual = abs(row["outcome"] - _estimate(state, row))
    return residual + 0.05 * _distance(row, history)


def _fit_roster(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = list(rows)
    history: list[dict[str, Any]] = []
    state = [0.0, 0.0, 0.0]
    picks = []
    changes = []
    for step_no in range(CONFIG["maximum_updates"]):
        priorities = [
            {"candidate_id": row["id"], "score": _priority(row, state, history)}
            for row in pool
        ]
        high_score = max(entry["score"] for entry in priorities)
        item_id = sorted(entry["candidate_id"] for entry in priorities if entry["score"] == high_score)[0]
        sample = next(entry for entry in pool if entry["id"] == item_id)
        picks.append({
            "step": step_no,
            "selected_id": item_id,
            "candidate_scores_digest": digest(priorities),
            "selected_score": high_score,
        })

        before = list(state)
        new_state, gradient = _gradient_step(before, sample)
        axis = sorted(range(3), key=lambda coordinate: (-abs(gradient[coordinate]), coordinate))[0]
        null_state = list(before)
        alternate = [before[i] + (0.0 if i == axis else gradient[i]) for i in range(3)]
        displacement = max(abs(x - y) for x, y in zip(new_state, alternate))
        mechanism = {
            "adapter": "known-coordinate-linear-simulator.v1",
            "coordinate": axis,
            "factual_delta_digest": digest(list(gradient)),
            "no_op_state_digest": _state_id(null_state),
            "no_op_identity": _state_id(null_state) == _state_id(before),
            "ablated_delta_digest": digest([0.0 if i == axis else gradient[i] for i in range(3)]),
            "ablation_max_abs_delta": displacement,
            "intervention_reached": displacement > 0.0,
            "feature_roster_digest": digest(list(sample["features"])),
        }
        changes.append({
            "step": step_no,
            "selected_id": item_id,
            "pre_state_digest": _state_id(before),
            "post_state_digest": _state_id(new_state),
            "gradient_delta_digest": digest(gradient),
            "mechanism": mechanism,
        })
        state = new_state
        history.append(sample)
        pool = [entry for entry in pool if entry["id"] != item_id]
    return picks, changes


def _control_path(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[list[float]]]:
    order = sorted(rows, key=lambda entry: entry["id"])
    weights = [0.0, 0.0, 0.0]
    trace = []
    snapshots = [list(weights)]
    for example in order[: CONFIG["maximum_updates"]]:
        before = list(weights)
        weights, delta = _gradient_step(before, example)
        trace.append({
            "selected_id": example["id"],
            "pre_state_digest": _state_id(before),
            "post_state_digest": _state_id(weights),
            "gradient_delta_digest": digest(delta),
        })
        snapshots.append(list(weights))
    return trace, snapshots


def _selector_snapshots(rows: Sequence[Mapping[str, Any]], picks: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    by_id = {entry["id"]: entry for entry in rows}
    model = [0.0, 0.0, 0.0]
    states = [list(model)]
    for choice in picks:
        model, _ = _gradient_step(model, by_id[choice["selected_id"]])
        states.append(list(model))
    return states


def _lock(name: str, checkpoint: int, state: Sequence[float], tune_error: float, fit_ids: str, tune_ids: str) -> dict[str, Any]:
    base = {
        "schema": "ahcl.learning-episode-prediction-lock.v1",
        "strategy": name,
        "config_digest": digest(CONFIG),
        "selected_updates": checkpoint,
        "fit_roster_digest": fit_ids,
        "tune_roster_digest": tune_ids,
        "tune_mse": tune_error,
        "weights": [float(weight) for weight in state],
        "weights_digest": _state_id(state),
    }
    return {**base, "lock_digest": digest(base)}


def _reconstruction() -> dict[str, Any]:
    fit = _make_cohort("fit", CONFIG["fit_candidates"])
    tune = _make_cohort("tune", CONFIG["tune_examples"])
    heldout = _make_cohort("assessment", CONFIG["assessment_examples"])
    fit_id_digest = digest([entry["id"] for entry in fit])
    tune_id_digest = digest([entry["id"] for entry in tune])
    picks, updates = _fit_roster(fit)
    control_trace, control_states = _control_path(fit)
    selector_states = _selector_snapshots(fit, picks)

    locks = {}
    tune_summary = {}
    for label, history in (("selector", selector_states), ("control", control_states)):
        errors = {n: _loss(tune, history[n]) for n in CONFIG["tune_checkpoints"]}
        optimum = min(CONFIG["tune_checkpoints"], key=lambda n: (errors[n], n))
        locks[label] = _lock(label, optimum, history[optimum], errors[optimum], fit_id_digest, tune_id_digest)
        tune_summary[label] = {
            "checkpoint_mse": {str(n): errors[n] for n in CONFIG["tune_checkpoints"]},
            "selected_updates": optimum,
        }

    expected_assessment = {
        "roster_digest": digest([entry["id"] for entry in heldout]),
        "n": len(heldout),
        "selector_mse": _loss(heldout, locks["selector"]["weights"]),
        "control_mse": _loss(heldout, locks["control"]["weights"]),
        "zero_model_mse": _loss(heldout, [0.0, 0.0, 0.0]),
    }
    expected_assessment["selector_minus_control_mse"] = (
        expected_assessment["selector_mse"] - expected_assessment["control_mse"]
    )
    expected_assessment["selector_minus_zero_model_mse"] = (
        expected_assessment["selector_mse"] - expected_assessment["zero_model_mse"]
    )
    fit_ids = {entry["id"] for entry in fit}
    tune_ids = {entry["id"] for entry in tune}
    heldout_ids = {entry["id"] for entry in heldout}
    expected = {
        "cohorts": {
            "fit": {"n": len(fit), "roster_digest": fit_id_digest, "data_digest": digest(fit)},
            "tune": {"n": len(tune), "roster_digest": tune_id_digest, "data_digest": digest(tune)},
            "assessment": {"n": len(heldout), "roster_digest": expected_assessment["roster_digest"]},
            "pairwise_disjoint": not (fit_ids & tune_ids or fit_ids & heldout_ids or tune_ids & heldout_ids),
        },
        "selection": {
            "strategy": CONFIG["selector"],
            "trace": picks,
            "selected_ids_digest": digest([item["selected_id"] for item in picks]),
        },
        "updates": updates,
        "control": {
            "strategy": CONFIG["control"],
            "trace": control_trace,
            "selected_ids_digest": digest([item["selected_id"] for item in control_trace]),
        },
        "tune": {"path": tune_summary, "lock_digests": {k: v["lock_digest"] for k, v in locks.items()}},
        "prediction_locks": locks,
        "assessment": expected_assessment,
    }
    return expected


def validate_receipt(payload: dict[str, Any]) -> None:
    """Rebuild the synthetic episode and compare every bound receipt field."""

    required = {
        "schema", "state_slice", "governance_slice", "assurance_level", "protocol", "config",
        "config_digest", "revision", "source_digest", "cohorts", "selection", "updates", "control", "tune",
        "prediction_locks", "assessment", "reviewer_status", "reproducibility_status", "runtime",
        "data_and_artifact_custody", "known_limitations", "next_blocked_authority_gate",
        "claim_ceiling", "nonclaims", "receipt_digest",
    }
    if set(payload) != required:
        raise ValidationError("receipt fields do not match the frozen v1 contract")
    if payload["schema"] != SCHEMA or payload["state_slice"] != STATE_SLICE:
        raise ValidationError("wrong receipt schema or state slice")
    if payload["governance_slice"] != GOVERNANCE_SLICE or payload["assurance_level"] != "INTERNAL_PASS":
        raise ValidationError("wrong governance slice or assurance level")
    if payload["protocol"] != CONFIG["protocol"] or payload["config"] != CONFIG:
        raise ValidationError("protocol configuration differs from the frozen contract")
    if payload["config_digest"] != digest(CONFIG):
        raise ValidationError("configuration digest mismatch")
    runner_path = Path(__file__).with_name("learning_episode.py")
    runner_digest = file_digest(runner_path)
    validator_digest = file_digest(Path(__file__).resolve())
    repository_root = runner_path.resolve().parents[2]
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    expected_revision = {
        "git_head": git_head,
        "runner_source_digest": runner_digest,
        "validator_source_digest": validator_digest,
        "state_slice_source_digest": digest({"runner": runner_digest, "validator": validator_digest}),
    }
    if payload["source_digest"] != runner_digest or payload["revision"] != expected_revision:
        raise ValidationError("runner source digest mismatch")
    if payload["reviewer_status"] != "NOT_REQUIRED_FOR_LOCAL_SYNTHETIC_SCOPE":
        raise ValidationError("unexpected reviewer status")
    if payload["reproducibility_status"] != "SAME_OWNER_SINGLE_ENVIRONMENT_REPEATABILITY_CHECK":
        raise ValidationError("unexpected reproducibility status")
    if payload["runtime"] != {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }:
        raise ValidationError("runtime identity mismatch")
    if payload["data_and_artifact_custody"] != {
        "input_data": "generated deterministically in process; synthetic only",
        "raw_data_retention": "none",
        "receipt": "aggregate metrics, toy predictor weights, synthetic IDs, and digests only",
    }:
        raise ValidationError("data or artifact custody declaration mismatch")
    if payload["known_limitations"] != [
        "same-owner local reproduction; no independent reviewer",
        "toy linear simulator exposes its ground-truth coordinates to the mechanism adapter",
        "in-process assessment ordering is not a cryptographic confidentiality boundary",
        "no model execution, real stream, or external data were exercised",
    ]:
        raise ValidationError("known limitation declaration mismatch")
    if payload["next_blocked_authority_gate"] != (
        "Any model-bearing extension requires a fresh packet and independent protocol/execution review."
    ):
        raise ValidationError("next authority gate mismatch")
    expected_claim = "LocalDevelopmentSyntheticLearningEpisodeReceiptV1"
    if payload["claim_ceiling"] != expected_claim or payload["nonclaims"] != NONCLAIMS:
        raise ValidationError("claim ceiling or nonclaims mismatch")
    computed_digest = digest({key: value for key, value in payload.items() if key != "receipt_digest"})
    if payload["receipt_digest"] != computed_digest:
        raise ValidationError("receipt digest mismatch")

    expected = _reconstruction()
    for field, expected_value in expected.items():
        if payload[field] != expected_value:
            raise ValidationError(f"reconstructed {field} differs from receipt")
    if payload["cohorts"]["pairwise_disjoint"] is not True:
        raise ValidationError("cohort identities are not disjoint")
    if any(not row["mechanism"]["no_op_identity"] for row in payload["updates"]):
        raise ValidationError("mechanism adapter no-op identity check failed")
    if any(not row["mechanism"]["intervention_reached"] for row in payload["updates"]):
        raise ValidationError("mechanism adapter intervention did not reach toy state")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    print(json.dumps({"valid": True, "assurance_level": receipt["assurance_level"], "receipt_digest": receipt["receipt_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
