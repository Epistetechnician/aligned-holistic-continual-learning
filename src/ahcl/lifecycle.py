"""Locked fit, tune, prediction-lock, and assessment lifecycle.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
The lifecycle never stores raw rows after a phase callback returns. Model
execution is opened only through :func:`open_model_trial`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .canonical import digest
from .execution import require_execution_open
from .review import ReviewReceipt
from .stats import StatisticalPlan, validate_locked_study


class LifecycleError(ValueError):
    """Raised when a study phase is called out of order or is malformed."""


@dataclass(frozen=True)
class TrialRow:
    """One family-repeat row passed to a phase callback."""

    family_id: str
    split: str
    repeat: int
    features: Any
    outcome: Any


@dataclass(frozen=True)
class PredictionLock:
    """Immutable tune lock consumed by held-out assessment."""

    config_digest: str
    predictor_digest: str
    fit_family_ids: tuple[str, ...]
    tune_family_ids: tuple[str, ...]
    tune_summary: Mapping[str, float]
    lock_digest: str


def _phase_rows(rows: Iterable[TrialRow], split: str, plan: StatisticalPlan) -> tuple[TrialRow, ...]:
    if split not in {"fit", "tune", "assessment"}:
        raise LifecycleError(f"unknown study split: {split}")
    materialized = tuple(rows)
    expected_families = {
        "fit": plan.fit_families,
        "tune": plan.tune_families,
        "assessment": plan.assessment_families,
    }[split]
    expected_rows = expected_families * plan.repeats
    if len(materialized) != expected_rows:
        raise LifecycleError(f"{split} requires {expected_rows} rows")
    keys = {(row.family_id, row.repeat) for row in materialized}
    if len(keys) != len(materialized):
        raise LifecycleError(f"{split} contains duplicate family-repeat rows")
    family_ids = {row.family_id for row in materialized}
    if len(family_ids) != expected_families:
        raise LifecycleError(f"{split} requires {expected_families} family identities")
    for family_id in family_ids:
        repeats = {row.repeat for row in materialized if row.family_id == family_id}
        if repeats != set(range(plan.repeats)):
            raise LifecycleError(f"{split} family {family_id} has incomplete repeats")
    if any(row.split != split for row in materialized):
        raise LifecycleError(f"rows are bound to a different split than {split}")
    return materialized


def _summary(values: Mapping[str, float]) -> dict[str, float]:
    result = dict(values)
    for key, value in result.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise LifecycleError(f"summary value {key} must be finite")
        result[key] = float(value)
    return result


class LockedTrial:
    """Enforce disjoint cohorts and prediction locking without raw retention."""

    def __init__(self, config: Mapping[str, Any], plan: StatisticalPlan | None = None):
        self.config = dict(config)
        self.plan = plan or validate_locked_study(self.config)
        self.plan.validate()
        self._phase = "ready"
        self._predictor: Any = None
        self._predictor_payload: Mapping[str, Any] | None = None
        self._fit_family_ids: tuple[str, ...] = ()
        self._tune_family_ids: tuple[str, ...] = ()
        self._lock: PredictionLock | None = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def prediction_lock(self) -> PredictionLock:
        if self._lock is None:
            raise LifecycleError("prediction lock is not available before tune")
        return self._lock

    def fit(
        self,
        rows: Iterable[TrialRow],
        fit_predictor: Callable[[Iterable[TrialRow]], tuple[Any, Mapping[str, Any]]],
    ) -> None:
        if self._phase != "ready":
            raise LifecycleError("fit may run exactly once before tune")
        materialized = _phase_rows(rows, "fit", self.plan)
        predictor, predictor_payload = fit_predictor(materialized)
        if not isinstance(predictor_payload, Mapping):
            raise LifecycleError("fit callback must return a mapping predictor payload")
        # Canonical serialization rejects non-finite values and non-JSON data.
        digest(dict(predictor_payload))
        self._predictor = predictor
        self._predictor_payload = dict(predictor_payload)
        self._fit_family_ids = tuple(sorted({row.family_id for row in materialized}))
        self._phase = "fit"

    def tune(
        self,
        rows: Iterable[TrialRow],
        evaluate: Callable[[Any, Iterable[TrialRow]], Mapping[str, float]],
    ) -> PredictionLock:
        if self._phase != "fit":
            raise LifecycleError("tune requires a completed fit")
        materialized = _phase_rows(rows, "tune", self.plan)
        self._tune_family_ids = tuple(sorted({row.family_id for row in materialized}))
        if set(self._fit_family_ids) & set(self._tune_family_ids):
            raise LifecycleError("fit and tune families must be disjoint")
        if self._predictor_payload is None:
            raise LifecycleError("fit predictor payload is missing")
        summary = _summary(evaluate(self._predictor, materialized))
        payload = {
            "schema": "ahcl.prediction-lock.v1",
            "config_digest": digest(self.config),
            "predictor_digest": digest(dict(self._predictor_payload)),
            "fit_family_ids": self._fit_family_ids,
            "tune_family_ids": self._tune_family_ids,
            "tune_summary": summary,
        }
        lock = PredictionLock(
            config_digest=payload["config_digest"],
            predictor_digest=payload["predictor_digest"],
            fit_family_ids=self._fit_family_ids,
            tune_family_ids=self._tune_family_ids,
            tune_summary=summary,
            lock_digest=digest(payload),
        )
        self._lock = lock
        self._phase = "locked"
        return lock

    def assess(
        self,
        rows: Iterable[TrialRow],
        evaluate: Callable[[Any, Iterable[TrialRow]], Mapping[str, float]],
    ) -> dict[str, float]:
        if self._phase != "locked":
            raise LifecycleError("assessment requires a completed tune lock")
        materialized = _phase_rows(rows, "assessment", self.plan)
        assessment_ids = {row.family_id for row in materialized}
        if assessment_ids & (set(self._fit_family_ids) | set(self._tune_family_ids)):
            raise LifecycleError("assessment families must be disjoint from fit and tune")
        return _summary(evaluate(self._predictor, materialized))


def open_model_trial(
    config: Mapping[str, Any],
    receipt: ReviewReceipt | None,
    *,
    packet_digest: str,
    operator_id: str,
) -> LockedTrial:
    """Open a model-bearing lifecycle only after independent review."""

    config_dict = dict(config)
    require_execution_open(config_dict, receipt, packet_digest=packet_digest, operator_id=operator_id)
    return LockedTrial(config_dict)
