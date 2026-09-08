"""Numerical qualification checks shared by native and instrumented runs.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


class CheckError(ValueError):
    """Raised when a predeclared intervention or parity check fails."""


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        shapes = [_shape(item) for item in value]
        if shapes and any(shape != shapes[0] for shape in shapes):
            raise CheckError("ragged values are not supported")
        return (len(value),) + (shapes[0] if shapes else ())
    return ()


def _paired(left: Any, right: Any) -> tuple[list[float], list[float]]:
    if _shape(left) != _shape(right):
        raise CheckError("comparison values have different shapes")
    a, b = list(_flatten(left)), list(_flatten(right))
    if not a or not all(math.isfinite(v) for v in a + b):
        raise CheckError("comparison values must be nonempty and finite")
    return a, b


def _threshold(value: float, *, positive: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise CheckError("threshold must be finite and nonnegative (positive for reach)")


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
    left_values, right_values = _paired(left, right)
    delta = max(abs(a - b) for a, b in zip(left_values, right_values))
    if not math.isfinite(delta):
        raise CheckError("comparison overflow")
    return delta


def assert_parity(native: Any, instrumented: Any, *, atol: float = 1e-5) -> float:
    _threshold(atol)
    delta = max_abs_delta(native, instrumented)
    if not math.isfinite(delta) or delta > atol:
        raise CheckError(f"native/instrumented parity exceeded atol: {delta} > {atol}")
    return delta


def assert_repeatable(first: Any, second: Any, *, atol: float = 1e-6) -> float:
    return assert_parity(first, second, atol=atol)


def assert_noop_identity(base: Any, no_op: Any, *, atol: float = 0.0) -> float:
    return assert_parity(base, no_op, atol=atol)


def assert_intervention_reach(base: Any, intervened: Any, *, minimum_delta: float = 1e-6) -> float:
    _threshold(minimum_delta, positive=True)
    delta = max_abs_delta(base, intervened)
    if not math.isfinite(delta) or delta < minimum_delta:
        raise CheckError(f"intervention did not reach the measured output: {delta}")
    return delta


def normalized_reconstruction_mse(original: Any, reconstructed: Any) -> float:
    original_values, reconstructed_values = _paired(original, reconstructed)
    try:
        denominator = sum(value * value for value in original_values)
        numerator = sum((a - b) ** 2 for a, b in zip(original_values, reconstructed_values))
    except OverflowError as exc:
        raise CheckError("reconstruction arithmetic overflow") from exc
    if not math.isfinite(denominator) or not math.isfinite(numerator):
        raise CheckError("reconstruction arithmetic overflow")
    if not all(math.isfinite(value) for value in original_values + reconstructed_values):
        raise CheckError("reconstruction values must be finite")
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def causal_scrub_effect(clean: Any, scrubbed: Any, *, minimum_delta: float = 1e-6) -> dict[str, float | bool]:
    _threshold(minimum_delta, positive=True)
    delta = max_abs_delta(clean, scrubbed)
    return {"max_abs_delta": delta, "effect_reached": bool(delta >= minimum_delta)}
