"""Combined native/instrumented qualification for one model feature path.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
This module receives caller-owned outputs from an already configured adapter
lane and returns aggregate evidence. It does not load models, retain traces,
or decide whether scientific assessment may open.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .checks import (
    assert_intervention_reach,
    assert_noop_identity,
    assert_parity,
    assert_repeatable,
    causal_scrub_effect,
    normalized_reconstruction_mse,
)


@dataclass(frozen=True)
class QualificationThresholds:
    parity_atol: float = 1e-5
    repeatability_atol: float = 1e-6
    reach_minimum_delta: float = 1e-6
    reconstruction_nmse_max: float = 0.05
    scrub_minimum_delta: float = 1e-6

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("qualification thresholds must be finite and non-negative")


@dataclass(frozen=True)
class QualificationEvidence:
    parity_delta: float
    repeatability_delta: float
    noop_delta: float
    intervention_delta: float
    reconstruction_nmse: float
    scrub_delta: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def qualify_path(
    *,
    native_output: Any,
    instrumented_output: Any,
    repeated_output: Any,
    no_op_output: Any,
    intervened_output: Any,
    base_output: Any,
    activation: Any,
    reconstructed_activation: Any,
    clean_output: Any,
    scrubbed_output: Any,
    thresholds: QualificationThresholds | None = None,
) -> QualificationEvidence:
    """Run all locked qualification checks and return aggregate evidence."""

    thresholds = thresholds or QualificationThresholds()
    thresholds.validate()
    parity_delta = assert_parity(native_output, instrumented_output, atol=thresholds.parity_atol)
    repeatability_delta = assert_repeatable(
        instrumented_output,
        repeated_output,
        atol=thresholds.repeatability_atol,
    )
    noop_delta = assert_noop_identity(base_output, no_op_output, atol=0.0)
    intervention_delta = assert_intervention_reach(
        base_output,
        intervened_output,
        minimum_delta=thresholds.reach_minimum_delta,
    )
    reconstruction_nmse = normalized_reconstruction_mse(activation, reconstructed_activation)
    if reconstruction_nmse > thresholds.reconstruction_nmse_max:
        raise ValueError(
            f"normalized reconstruction MSE exceeded threshold: "
            f"{reconstruction_nmse} > {thresholds.reconstruction_nmse_max}"
        )
    scrub = causal_scrub_effect(clean_output, scrubbed_output, minimum_delta=thresholds.scrub_minimum_delta)
    if not scrub["effect_reached"]:
        raise ValueError("causal scrub did not reach the measured output")
    return QualificationEvidence(
        parity_delta=parity_delta,
        repeatability_delta=repeatability_delta,
        noop_delta=noop_delta,
        intervention_delta=intervention_delta,
        reconstruction_nmse=reconstruction_nmse,
        scrub_delta=float(scrub["max_abs_delta"]),
    )
