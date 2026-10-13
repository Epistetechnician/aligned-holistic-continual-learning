"""Digest-bound synthetic learning episode from selection through assessment.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
Governance slice: global-experiment-assurance-lifecycle-v1.

This protocol is a local systems integration exercise. Its selector, linear
update rule, mechanism adapter, and sealed evaluator operate only on a fresh,
deterministic toy stream. The mechanism adapter observes known simulator state;
it is not a model interpretability result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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


def _uniform(seed: str, split: str, index: int, coordinate: int) -> float:
    material = f"{seed}:{split}:{index}:{coordinate}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return (integer / ((1 << 64) - 1)) * 2.0 - 1.0


def _rows(split: str, count: int) -> list[dict[str, Any]]:
    result = []
    for index in range(count):
        features = tuple(_uniform(CONFIG["seed"], split, index, coordinate) for coordinate in range(3))
        x0, x1, x2 = features
        outcome = (
            CONFIG["target_weights"][0] * x0
            + CONFIG["target_weights"][1] * x1
            + CONFIG["target_weights"][2] * x2
            + CONFIG["interaction_weight"] * x0 * x1
            + CONFIG["noise_scale"] * _uniform(CONFIG["seed"], split, index, 3)
        )
        result.append({"id": f"{split}-{index:03d}", "features": features, "outcome": outcome})
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _predict(weights: Sequence[float], row: Mapping[str, Any]) -> float:
    return _dot(weights, row["features"])


def _mse(rows: Iterable[Mapping[str, Any]], weights: Sequence[float]) -> float:
    materialized = list(rows)
    return sum((_predict(weights, row) - row["outcome"]) ** 2 for row in materialized) / len(materialized)


def _weights_digest(weights: Sequence[float]) -> str:
    return digest([float(value) for value in weights])


def _step(weights: Sequence[float], row: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    error = row["outcome"] - _predict(weights, row)
    delta = [CONFIG["learning_rate"] * error * value for value in row["features"]]
    return [weight + change for weight, change in zip(weights, delta)], delta


def _novelty(row: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> float:
    if not selected:
        return 0.0
    distances = [
        sum((left - right) ** 2 for left, right in zip(row["features"], prior["features"])) ** 0.5
        for prior in selected
    ]
    return min(distances)


def _selector_score(
    row: Mapping[str, Any], weights: Sequence[float], selected: Sequence[Mapping[str, Any]]
) -> float:
    error = abs(row["outcome"] - _predict(weights, row))
    return error + CONFIG["novelty_weight"] * _novelty(row, selected)


def _mechanism_record(
    before: Sequence[float], after: Sequence[float], delta: Sequence[float], row: Mapping[str, Any]
) -> dict[str, Any]:
    # This adapter is allowed to inspect the toy simulator's exact coordinates.
    coordinate = max(range(len(delta)), key=lambda index: (abs(delta[index]), -index))
    no_op_delta = [0.0 for _ in delta]
    ablated_delta = [
        0.0 if index == coordinate else delta[index]
        for index in range(len(delta))
    ]
    no_op_weights = [weight + change for weight, change in zip(before, no_op_delta)]
    ablated_weights = [weight + change for weight, change in zip(before, ablated_delta)]
    ablation_distance = max(abs(left - right) for left, right in zip(after, ablated_weights))
    return {
        "adapter": "known-coordinate-linear-simulator.v1",
        "coordinate": coordinate,
        "factual_delta_digest": digest(list(delta)),
        "no_op_state_digest": _weights_digest(no_op_weights),
        "no_op_identity": _weights_digest(no_op_weights) == _weights_digest(before),
        "ablated_delta_digest": digest(ablated_delta),
        "ablation_max_abs_delta": ablation_distance,
        "intervention_reached": ablation_distance > 0.0,
        "feature_roster_digest": digest(list(row["features"])),
    }


def _lock_payload(
    strategy: str,
    count: int,
    weights: Sequence[float],
    tune_loss: float,
    fit_digest: str,
    tune_digest: str,
) -> dict[str, Any]:
    payload = {
        "schema": "ahcl.learning-episode-prediction-lock.v1",
        "strategy": strategy,
        "config_digest": digest(CONFIG),
        "selected_updates": count,
        "fit_roster_digest": fit_digest,
        "tune_roster_digest": tune_digest,
        "tune_mse": tune_loss,
        "weights": [float(value) for value in weights],
        "weights_digest": _weights_digest(weights),
    }
    return {**payload, "lock_digest": digest(payload)}


class SealedAssessment:
    """One-use aggregate evaluator that generates assessment rows after lock."""

    def __init__(self) -> None:
        self._phase = "sealed"

    @property
    def phase(self) -> str:
        return self._phase

    def evaluate(self, locks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        if self._phase != "sealed":
            raise ValueError("sealed assessment can be opened exactly once")
        if set(locks) != {"selector", "control"}:
            raise ValueError("assessment requires selector and control prediction locks")
        for name, lock in locks.items():
            if lock.get("config_digest") != digest(CONFIG) or lock.get("strategy") != name:
                raise ValueError("prediction lock does not match the frozen episode")
            if lock.get("lock_digest") != digest({key: value for key, value in lock.items() if key != "lock_digest"}):
                raise ValueError("prediction lock digest mismatch")
            if lock.get("weights_digest") != _weights_digest(lock.get("weights", [])):
                raise ValueError("prediction lock weights digest mismatch")

        # Consume before materializing or evaluating held-out rows.
        self._phase = "opened"
        rows = _rows("assessment", CONFIG["assessment_examples"])
        zero_weights = [0.0] * CONFIG["feature_dimension"]
        roster = [row["id"] for row in rows]
        return {
            "roster_digest": digest(roster),
            "n": len(rows),
            "selector_mse": _mse(rows, locks["selector"]["weights"]),
            "control_mse": _mse(rows, locks["control"]["weights"]),
            "zero_model_mse": _mse(rows, zero_weights),
        }


def run_episode() -> dict[str, Any]:
    """Run one synthetic selection/update/mechanism/locked-assessment episode."""

    fit_rows = _rows("fit", CONFIG["fit_candidates"])
    tune_rows = _rows("tune", CONFIG["tune_examples"])
    fit_roster_digest = digest([row["id"] for row in fit_rows])
    tune_roster_digest = digest([row["id"] for row in tune_rows])
    fit_data_digest = digest(fit_rows)
    tune_data_digest = digest(tune_rows)

    # Model-state-conditioned acquisition is restricted to the fit candidate pool.
    remaining = list(fit_rows)
    selected: list[dict[str, Any]] = []
    selector_weights = [0.0] * CONFIG["feature_dimension"]
    selection_trace: list[dict[str, Any]] = []
    update_trace: list[dict[str, Any]] = []
    for step_index in range(CONFIG["maximum_updates"]):
        candidates = [
            {"candidate_id": row["id"], "score": _selector_score(row, selector_weights, selected)}
            for row in remaining
        ]
        best_score = max(item["score"] for item in candidates)
        chosen_id = min(item["candidate_id"] for item in candidates if item["score"] == best_score)
        chosen = next(row for row in remaining if row["id"] == chosen_id)
        selection_trace.append({
            "step": step_index,
            "selected_id": chosen_id,
            "candidate_scores_digest": digest(candidates),
            "selected_score": best_score,
        })

        before = list(selector_weights)
        after, delta = _step(before, chosen)
        mechanism = _mechanism_record(before, after, delta, chosen)
        update_trace.append({
            "step": step_index,
            "selected_id": chosen_id,
            "pre_state_digest": _weights_digest(before),
            "post_state_digest": _weights_digest(after),
            "gradient_delta_digest": digest(delta),
            "mechanism": mechanism,
        })
        selector_weights = after
        selected.append(chosen)
        remaining.remove(chosen)

    # Fixed-ID replay is a bounded control with the same fit pool and update rule.
    control_weights = [0.0] * CONFIG["feature_dimension"]
    control_trace = []
    for row in sorted(fit_rows, key=lambda item: item["id"])[: CONFIG["maximum_updates"]]:
        before = list(control_weights)
        control_weights, delta = _step(before, row)
        control_trace.append({
            "selected_id": row["id"],
            "pre_state_digest": _weights_digest(before),
            "post_state_digest": _weights_digest(control_weights),
            "gradient_delta_digest": digest(delta),
        })

    selector_snapshots = [[0.0] * CONFIG["feature_dimension"]]
    control_snapshots = [[0.0] * CONFIG["feature_dimension"]]
    selector_current = [0.0] * CONFIG["feature_dimension"]
    control_current = [0.0] * CONFIG["feature_dimension"]
    for index in range(CONFIG["maximum_updates"]):
        selector_current, _ = _step(selector_current, selected[index])
        control_current, _ = _step(
            control_current,
            sorted(fit_rows, key=lambda item: item["id"])[index],
        )
        selector_snapshots.append(selector_current)
        control_snapshots.append(control_current)

    locks: dict[str, dict[str, Any]] = {}
    tune_path = {}
    for name, snapshots in (("selector", selector_snapshots), ("control", control_snapshots)):
        candidate_losses = {
            count: _mse(tune_rows, snapshots[count])
            for count in CONFIG["tune_checkpoints"]
        }
        selected_count = min(CONFIG["tune_checkpoints"], key=lambda count: (candidate_losses[count], count))
        locks[name] = _lock_payload(
            name,
            selected_count,
            snapshots[selected_count],
            candidate_losses[selected_count],
            fit_roster_digest,
            tune_roster_digest,
        )
        tune_path[name] = {
            "checkpoint_mse": {str(count): candidate_losses[count] for count in CONFIG["tune_checkpoints"]},
            "selected_updates": selected_count,
        }

    sealed_assessment = SealedAssessment()
    assessment = sealed_assessment.evaluate(locks)
    assessment["selector_minus_control_mse"] = assessment["selector_mse"] - assessment["control_mse"]
    assessment["selector_minus_zero_model_mse"] = assessment["selector_mse"] - assessment["zero_model_mse"]

    runner_path = Path(__file__).resolve()
    validator_path = runner_path.with_name("validate_learning_episode.py")
    repository_root = runner_path.parents[2]
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    runner_digest = file_digest(runner_path)
    validator_digest = file_digest(validator_path)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "state_slice": STATE_SLICE,
        "governance_slice": GOVERNANCE_SLICE,
        "assurance_level": "INTERNAL_PASS",
        "protocol": CONFIG["protocol"],
        "config": CONFIG,
        "config_digest": digest(CONFIG),
        "revision": {
            "git_head": git_head,
            "runner_source_digest": runner_digest,
            "validator_source_digest": validator_digest,
            "state_slice_source_digest": digest({"runner": runner_digest, "validator": validator_digest}),
        },
        "source_digest": runner_digest,
        "cohorts": {
            "fit": {"n": len(fit_rows), "roster_digest": fit_roster_digest, "data_digest": fit_data_digest},
            "tune": {"n": len(tune_rows), "roster_digest": tune_roster_digest, "data_digest": tune_data_digest},
            "assessment": {"n": assessment["n"], "roster_digest": assessment["roster_digest"]},
            "pairwise_disjoint": True,
        },
        "selection": {
            "strategy": CONFIG["selector"],
            "trace": selection_trace,
            "selected_ids_digest": digest([item["selected_id"] for item in selection_trace]),
        },
        "updates": update_trace,
        "control": {
            "strategy": CONFIG["control"],
            "trace": control_trace,
            "selected_ids_digest": digest([item["selected_id"] for item in control_trace]),
        },
        "tune": {"path": tune_path, "lock_digests": {key: value["lock_digest"] for key, value in locks.items()}},
        "prediction_locks": locks,
        "assessment": assessment,
        "reviewer_status": "NOT_REQUIRED_FOR_LOCAL_SYNTHETIC_SCOPE",
        "reproducibility_status": "SAME_OWNER_SINGLE_ENVIRONMENT_REPEATABILITY_CHECK",
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "data_and_artifact_custody": {
            "input_data": "generated deterministically in process; synthetic only",
            "raw_data_retention": "none",
            "receipt": "aggregate metrics, toy predictor weights, synthetic IDs, and digests only",
        },
        "known_limitations": [
            "same-owner local reproduction; no independent reviewer",
            "toy linear simulator exposes its ground-truth coordinates to the mechanism adapter",
            "in-process assessment ordering is not a cryptographic confidentiality boundary",
            "no model execution, real stream, or external data were exercised",
        ],
        "next_blocked_authority_gate": (
            "Any model-bearing extension requires a fresh packet and independent protocol/execution review."
        ),
        "claim_ceiling": "LocalDevelopmentSyntheticLearningEpisodeReceiptV1",
        "nonclaims": [
            "synthetic-only deterministic fixture",
            "no model weights, provider calls, remote workers, or user data",
            "mechanism adapter sees known toy simulator coordinates",
            "no real continual-learning, mechanistic, alignment, or superiority claim",
            "assessment seal is an in-process ordering gate, not cryptographic secrecy",
            "no independent review or production authorization",
        ],
    }
    payload["receipt_digest"] = digest(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_episode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"assurance_level": receipt["assurance_level"], "receipt_digest": receipt["receipt_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
