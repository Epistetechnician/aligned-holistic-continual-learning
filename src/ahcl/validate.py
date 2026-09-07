"""Independent aggregate contract validator for synthetic results."""

from __future__ import annotations

import argparse
import json
import math
import string
from pathlib import Path
from typing import Any

from .canonical import digest
from .experiment import CONFIG
from .synthetic import ARMS, AXES, REPEATS, SPLITS, STATE_SLICE


class ValidationError(ValueError):
    """Raised when an aggregate result violates the locked contract."""


def _finite(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{label} must be finite")


def validate_result(payload: dict[str, Any]) -> None:
    """Check identity, roster, metric ranges, controls, and digest integrity."""

    required = {
        "schema",
        "state_slice",
        "config",
        "config_digest",
        "roster",
        "prediction_lock_digest",
        "tune_metrics",
        "assessment_metrics",
        "shuffled_control_r2",
        "arm_metrics",
        "gates",
        "gates_pass",
        "status",
        "nonclaims",
        "experiment_digest",
    }
    if set(payload) != required:
        raise ValidationError("result keys do not match the locked schema")
    if payload["schema"] != "ahcl.synthetic.result.v1" or payload["state_slice"] != STATE_SLICE:
        raise ValidationError("wrong schema or state slice")
    if payload["config"] != CONFIG or payload["config_digest"] != digest(CONFIG):
        raise ValidationError("configuration identity mismatch")
    roster = payload["roster"]
    if roster != {
        "families": 144,
        "families_by_split": SPLITS,
        "repeats": REPEATS,
        "observations_per_arm": 432,
    }:
        raise ValidationError("roster mismatch")
    if (
        not isinstance(payload["prediction_lock_digest"], str)
        or len(payload["prediction_lock_digest"]) != 64
        or any(character not in string.hexdigits for character in payload["prediction_lock_digest"])
    ):
        raise ValidationError("prediction lock digest is malformed")
    for metric_name in ("tune_metrics", "assessment_metrics"):
        metrics = payload[metric_name]
        if set(metrics) != set(AXES):
            raise ValidationError(f"{metric_name} axis mismatch")
        for axis in AXES:
            values = metrics[axis]
            if set(values) != {"r2", "baseline_r2", "sign_agreement"}:
                raise ValidationError(f"{metric_name} metric shape mismatch")
            for key, value in values.items():
                _finite(value, f"{metric_name}.{axis}.{key}")
            if not 0.0 <= values["sign_agreement"] <= 1.0:
                raise ValidationError("sign agreement outside [0, 1]")
    if set(payload["shuffled_control_r2"]) != set(AXES):
        raise ValidationError("shuffled control axis mismatch")
    for axis, value in payload["shuffled_control_r2"].items():
        _finite(value, f"shuffled_control_r2.{axis}")
    if set(payload["arm_metrics"]) != set(ARMS):
        raise ValidationError("arm mismatch")
    for arm in ARMS:
        if set(payload["arm_metrics"][arm]) != set(AXES):
            raise ValidationError("arm axis mismatch")
    gates = payload["gates"]
    if gates != CONFIG["gates"]:
        raise ValidationError("gate mismatch")
    expected_digest = digest({key: value for key, value in payload.items() if key != "experiment_digest"})
    if payload["experiment_digest"] != expected_digest:
        raise ValidationError("experiment digest mismatch")
    if set(payload["gates_pass"]) != {"primary", "controls"} or not all(
        isinstance(value, bool) for value in payload["gates_pass"].values()
    ):
        raise ValidationError("gate status shape mismatch")
    primary = payload["gates_pass"]["primary"]
    controls = payload["gates_pass"]["controls"]
    if payload["status"] not in {"SyntheticCandidate", "SyntheticNoCandidate"}:
        raise ValidationError("unknown status")
    if payload["status"] == "SyntheticCandidate" and not (primary and controls):
        raise ValidationError("candidate status without passing gates")
    if payload["status"] == "SyntheticNoCandidate" and primary and controls:
        raise ValidationError("no-candidate status despite passing gates")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    validate_result(payload)
    print(json.dumps({"valid": True, "experiment_digest": payload["experiment_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
