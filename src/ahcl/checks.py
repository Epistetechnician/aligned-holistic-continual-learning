"""Numerical qualification checks shared by native and instrumented runs.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


class CheckError(ValueError):
    """Raised when a predeclared intervention or parity check fails."""


def _flatten(value: Any) -> Iterable[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        value = value.reshape(-1).tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, list, tuple)):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
        return
    yield float(value)


def max_abs_delta(left: Any, right: Any) -> float:
    left_values = list(_flatten(left))
    right_values = list(_flatten(right))
    if len(left_values) != len(right_values):
        raise CheckError("comparison values have different sizes")
    return max((abs(a - b) for a, b in zip(left_values, right_values)), default=0.0)


def assert_parity(native: Any, instrumented: Any, *, atol: float = 1e-5) -> float:
    delta = max_abs_delta(native, instrumented)
    if not math.isfinite(delta) or delta > atol:
        raise CheckError(f"native/instrumented parity exceeded atol: {delta} > {atol}")
    return delta


def assert_repeatable(first: Any, second: Any, *, atol: float = 1e-6) -> float:
    return assert_parity(first, second, atol=atol)


def assert_noop_identity(base: Any, no_op: Any, *, atol: float = 0.0) -> float:
    return assert_parity(base, no_op, atol=atol)


def assert_intervention_reach(base: Any, intervened: Any, *, minimum_delta: float = 1e-6) -> float:
    delta = max_abs_delta(base, intervened)
    if not math.isfinite(delta) or delta < minimum_delta:
        raise CheckError(f"intervention did not reach the measured output: {delta}")
    return delta


def normalized_reconstruction_mse(original: Any, reconstructed: Any) -> float:
    original_values = list(_flatten(original))
    reconstructed_values = list(_flatten(reconstructed))
    if len(original_values) != len(reconstructed_values):
        raise CheckError("reconstruction values have different sizes")
    denominator = sum(value * value for value in original_values)
    numerator = sum((a - b) ** 2 for a, b in zip(original_values, reconstructed_values))
    if not all(math.isfinite(value) for value in original_values + reconstructed_values):
        raise CheckError("reconstruction values must be finite")
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def causal_scrub_effect(clean: Any, scrubbed: Any, *, minimum_delta: float = 1e-6) -> dict[str, float | bool]:
    delta = max_abs_delta(clean, scrubbed)
    return {"max_abs_delta": delta, "effect_reached": bool(delta >= minimum_delta)}
