"""Update-control policies driven by a locked consequence predictor.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
The controller is deliberately separate from model and optimizer code. It can
therefore be compared with ordinary SGD, EWC, and a behavioral guard while
recording the decision and prediction used for each update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .continual import StreamBatch, UpdatePolicy


class ControlError(ValueError):
    """Raised when a locked controller receives an invalid prediction."""


@dataclass(frozen=True)
class ControlDecision:
    action: str
    prediction: tuple[float, ...]
    reason: str


class UpdateController(Protocol):
    def decide(self, prediction: Sequence[float]) -> ControlDecision:
        """Return the locked action for one candidate update."""


class FixedUpdateController:
    """Always apply the candidate update; the ordinary SGD comparator."""

    def decide(self, prediction: Sequence[float]) -> ControlDecision:
        values = _finite_prediction(prediction)
        return ControlDecision("apply", values, "fixed_update")


class RetentionGuardController:
    """Reject updates whose predicted retention harm exceeds a fixed ceiling."""

    def __init__(
        self,
        *,
        retention_harm_ceiling: float,
        adaptation_gain_floor: float = -math.inf,
        retention_index: int = 1,
        adaptation_index: int = 0,
    ):
        if not math.isfinite(retention_harm_ceiling):
            raise ValueError("retention_harm_ceiling must be finite")
        if adaptation_gain_floor != -math.inf and not math.isfinite(adaptation_gain_floor):
            raise ValueError("adaptation_gain_floor must be finite or -inf")
        if min(retention_index, adaptation_index) < 0:
            raise ValueError("prediction indices must be non-negative")
        self.retention_harm_ceiling = retention_harm_ceiling
        self.adaptation_gain_floor = adaptation_gain_floor
        self.retention_index = retention_index
        self.adaptation_index = adaptation_index

    def decide(self, prediction: Sequence[float]) -> ControlDecision:
        values = _finite_prediction(prediction)
        required = max(self.retention_index, self.adaptation_index)
        if len(values) <= required:
            raise ControlError("prediction does not contain the locked behavior axes")
        retention_harm = values[self.retention_index]
        adaptation_gain = values[self.adaptation_index]
        adaptation_gate = self.adaptation_gain_floor == -math.inf or adaptation_gain < self.adaptation_gain_floor
        if retention_harm > self.retention_harm_ceiling and adaptation_gate:
            return ControlDecision("skip", values, "predicted_retention_harm")
        return ControlDecision("apply", values, "within_locked_guard")


class BehavioralGuardController:
    """Behavior-only comparator using a pre-update metric supplied by the caller."""

    def __init__(self, *, retention_harm_ceiling: float, metric_key: str = "retention_harm"):
        if not math.isfinite(retention_harm_ceiling):
            raise ValueError("retention_harm_ceiling must be finite")
        if not metric_key:
            raise ValueError("metric_key must be non-empty")
        self.retention_harm_ceiling = retention_harm_ceiling
        self.metric_key = metric_key

    def decide_from_metrics(self, metrics: dict[str, float]) -> ControlDecision:
        value = metrics.get(self.metric_key)
        if value is None or not math.isfinite(float(value)):
            raise ControlError(f"behavior metric {self.metric_key!r} is missing or non-finite")
        action = "skip" if float(value) > self.retention_harm_ceiling else "apply"
        reason = "behavioral_retention_harm" if action == "skip" else "behavior_within_ceiling"
        return ControlDecision(action, (float(value),), reason)


def _finite_prediction(prediction: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in prediction)
    if not values or any(not math.isfinite(value) for value in values):
        raise ControlError("controller predictions must be finite and non-empty")
    return values


class MonitorControlledUpdatePolicy:
    """Gate an update policy with a fit-and-tune-locked consequence predictor."""

    def __init__(
        self,
        base_policy: UpdatePolicy,
        feature_extractor: Callable[[StreamBatch], Sequence[float]],
        predictor: Callable[[Sequence[float]], Sequence[float]],
        controller: UpdateController,
    ):
        self.base_policy = base_policy
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.controller = controller
        self.last_decision: ControlDecision | None = None
        self.applied_updates = 0
        self.skipped_updates = 0

    def update(self, batch: StreamBatch) -> float:
        features = tuple(float(value) for value in self.feature_extractor(batch))
        prediction = self.predictor(features)
        decision = self.controller.decide(prediction)
        self.last_decision = decision
        if decision.action == "skip":
            self.skipped_updates += 1
            return 0.0
        self.applied_updates += 1
        return self.base_policy.update(batch)

    def control_summary(self) -> dict[str, float | int]:
        """Return aggregate control counts and reject an all-skip trajectory."""

        total = self.applied_updates + self.skipped_updates
        if total == 0:
            raise ControlError("controller received no updates")
        if self.applied_updates == 0:
            raise ControlError("rejecting every update is an invalid experiment")
        return {
            "applied_updates": self.applied_updates,
            "skipped_updates": self.skipped_updates,
            "skip_rate": self.skipped_updates / total,
        }
