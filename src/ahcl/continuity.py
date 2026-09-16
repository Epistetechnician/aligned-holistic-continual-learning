"""Aggregate-only mechanistic continuity measurements across checkpoints.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.

This module consumes caller-owned summaries, not models or raw traces. Circuit
lineage is explicit input: the calculator never rematches features after the
fact. A report exposes representation, circuit, intervention, and behavioral
components separately, so a combined score cannot conceal a failed invariant.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .canonical import digest


STATE_SLICE = "aligned-holistic-continual-learning-integrated-stack-v2"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ContinuityError(ValueError):
    """Raised when a checkpoint or lineage packet is malformed."""


def _label(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise ContinuityError(f"{field} must be a nonempty printable string")
    return value


def _state_digest(value: Any) -> str:
    value = _label(value, "state_digest")
    if not _DIGEST.fullmatch(value):
        raise ContinuityError("state_digest must be 64 lowercase hexadecimal characters")
    return value


def _vector(values: Any, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ContinuityError(f"{field} must be a nonempty numeric vector")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ContinuityError(f"{field} must be a nonempty numeric vector") from error
    if not result or any(not math.isfinite(value) for value in result):
        raise ContinuityError(f"{field} must be nonempty and finite")
    return result


def _effects(values: Any, field: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ContinuityError(f"{field} must be a nonempty mapping")
    result: dict[str, float] = {}
    for key, value in values.items():
        name = _label(key, f"{field} id")
        if name in result:
            raise ContinuityError(f"{field} contains duplicate ids")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ContinuityError(f"{field} values must be numeric") from error
        if not math.isfinite(numeric):
            raise ContinuityError(f"{field} values must be finite")
        result[name] = numeric
    return result


@dataclass(frozen=True)
class CheckpointMeasurement:
    """One aggregate measurement at a locked checkpoint."""

    checkpoint_id: str
    step: int
    state_digest: str
    representation: Sequence[float]
    circuit_effects: Mapping[str, float]
    intervention_effects: Mapping[str, float]
    behavior: Sequence[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_id", _label(self.checkpoint_id, "checkpoint_id"))
        if type(self.step) is not int or self.step < 0:
            raise ContinuityError("checkpoint step must be a nonnegative integer")
        object.__setattr__(self, "state_digest", _state_digest(self.state_digest))
        object.__setattr__(self, "representation", _vector(self.representation, "representation"))
        object.__setattr__(self, "circuit_effects", _effects(self.circuit_effects, "circuit_effects"))
        object.__setattr__(self, "intervention_effects", _effects(self.intervention_effects, "intervention_effects"))
        object.__setattr__(self, "behavior", _vector(self.behavior, "behavior"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "state_digest": self.state_digest,
            "representation": self.representation,
            "circuit_effects": dict(sorted(self.circuit_effects.items())),
            "intervention_effects": dict(sorted(self.intervention_effects.items())),
            "behavior": self.behavior,
        }


@dataclass(frozen=True)
class LineageEdge:
    """Caller-supplied source-to-target circuit mass for one transition."""

    source_id: str
    target_id: str
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _label(self.source_id, "lineage source_id"))
        object.__setattr__(self, "target_id", _label(self.target_id, "lineage target_id"))
        try:
            weight = float(self.weight)
        except (TypeError, ValueError) as error:
            raise ContinuityError("lineage weight must be numeric") from error
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            raise ContinuityError("lineage weight must be finite and in (0, 1]")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class LineageTransition:
    """Explicit lineage edges between two adjacent checkpoint identities."""

    before_checkpoint: str
    after_checkpoint: str
    edges: Sequence[LineageEdge]

    def __post_init__(self) -> None:
        object.__setattr__(self, "before_checkpoint", _label(self.before_checkpoint, "before_checkpoint"))
        object.__setattr__(self, "after_checkpoint", _label(self.after_checkpoint, "after_checkpoint"))
        materialized = tuple(self.edges)
        if any(not isinstance(edge, LineageEdge) for edge in materialized):
            raise ContinuityError("lineage edges must be LineageEdge values")
        identities = [(edge.source_id, edge.target_id) for edge in materialized]
        if len(identities) != len(set(identities)):
            raise ContinuityError("lineage contains duplicate source-target edges")
        source_mass: dict[str, float] = {}
        target_mass: dict[str, float] = {}
        for edge in materialized:
            source_mass[edge.source_id] = source_mass.get(edge.source_id, 0.0) + edge.weight
            target_mass[edge.target_id] = target_mass.get(edge.target_id, 0.0) + edge.weight
        if any(value > 1.0 + 1e-12 for value in (*source_mass.values(), *target_mass.values())):
            raise ContinuityError("lineage source and target mass must not exceed one")
        object.__setattr__(self, "edges", materialized)


@dataclass(frozen=True)
class ContinuityPolicy:
    """Locked per-component gates; every component must pass."""

    representation_min: float = 0.8
    circuit_min: float = 0.8
    intervention_min: float = 0.8
    behavior_min: float = 0.8
    drift_behavior_min: float = 0.95
    drift_circuit_max: float = 0.5

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
            raise ContinuityError("continuity thresholds must be finite values in [0, 1]")


def _cosine(left: Sequence[float], right: Sequence[float], field: str) -> float:
    if len(left) != len(right):
        raise ContinuityError(f"{field} dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ContinuityError(f"{field} cosine is undefined for a zero vector")
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)))


def _behavioral_continuity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ContinuityError("behavior dimensions differ")
    ratios = []
    for before, after in zip(left, right):
        denominator = abs(before) + abs(after)
        ratios.append(0.0 if denominator == 0.0 else abs(before - after) / denominator)
    return 1.0 - sum(ratios) / len(ratios)


def _lineage_scores(
    before: CheckpointMeasurement,
    after: CheckpointMeasurement,
    transition: LineageTransition,
) -> tuple[float, float, float]:
    sources = set(before.circuit_effects)
    targets = set(after.circuit_effects)
    source_mass = {source: 0.0 for source in sources}
    target_mass = {target: 0.0 for target in targets}
    for edge in transition.edges:
        if edge.source_id not in sources or edge.target_id not in targets:
            raise ContinuityError("lineage edge refers to a circuit absent from its checkpoint")
        source_mass[edge.source_id] += edge.weight
        target_mass[edge.target_id] += edge.weight
    source_preservation = sum(source_mass.values()) / len(sources)
    target_coverage = sum(target_mass.values()) / len(targets)
    return source_preservation, target_coverage, (source_preservation + target_coverage) / 2.0


def build_continuity_report(
    checkpoints: Sequence[CheckpointMeasurement],
    transitions: Sequence[LineageTransition],
    *,
    policy: ContinuityPolicy | None = None,
) -> dict[str, Any]:
    """Validate a trajectory and return a digest-bound aggregate report."""

    policy = policy or ContinuityPolicy()
    policy.validate()
    points = tuple(checkpoints)
    if len(points) < 2:
        raise ContinuityError("at least two checkpoints are required")
    if any(not isinstance(point, CheckpointMeasurement) for point in points):
        raise ContinuityError("checkpoints must be CheckpointMeasurement values")
    if [point.step for point in points] != list(range(len(points))):
        raise ContinuityError("checkpoint steps must be complete and ordered from zero")
    checkpoint_ids = [point.checkpoint_id for point in points]
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ContinuityError("checkpoint ids must be unique")
    representation_width = len(points[0].representation)
    behavior_width = len(points[0].behavior)
    intervention_ids = set(points[0].intervention_effects)
    if any(len(point.representation) != representation_width or len(point.behavior) != behavior_width for point in points):
        raise ContinuityError("representation and behavior dimensions must remain fixed")
    if any(set(point.intervention_effects) != intervention_ids for point in points):
        raise ContinuityError("intervention probe ids must remain fixed across checkpoints")

    supplied = tuple(transitions)
    if len(supplied) != len(points) - 1 or any(not isinstance(item, LineageTransition) for item in supplied):
        raise ContinuityError("one lineage transition is required for every adjacent checkpoint pair")
    expected_pairs = tuple(zip(checkpoint_ids, checkpoint_ids[1:]))
    supplied_pairs = tuple((item.before_checkpoint, item.after_checkpoint) for item in supplied)
    if supplied_pairs != expected_pairs:
        raise ContinuityError("lineage transitions must follow checkpoint order exactly")

    rows: list[dict[str, Any]] = []
    for before, after, transition in zip(points, points[1:], supplied):
        source_preservation, target_coverage, circuit_continuity = _lineage_scores(before, after, transition)
        representation = _cosine(before.representation, after.representation, "representation")
        intervention = _cosine(
            tuple(before.intervention_effects[key] for key in sorted(intervention_ids)),
            tuple(after.intervention_effects[key] for key in sorted(intervention_ids)),
            "intervention effects",
        )
        behavior = _behavioral_continuity(before.behavior, after.behavior)
        components = {
            "representation_continuity": representation,
            "circuit_continuity": circuit_continuity,
            "intervention_continuity": intervention,
            "behavioral_continuity": behavior,
        }
        passes = {
            "representation": representation >= policy.representation_min,
            "circuit": circuit_continuity >= policy.circuit_min,
            "intervention": intervention >= policy.intervention_min,
            "behavior": behavior >= policy.behavior_min,
        }
        rows.append({
            "from": before.checkpoint_id,
            "to": after.checkpoint_id,
            "source_preservation": source_preservation,
            "target_coverage": target_coverage,
            **components,
            "continuity_mean": sum(components.values()) / len(components),
            "invariants_pass": all(passes.values()),
            "failed_invariants": tuple(name for name, passed in passes.items() if not passed),
            "behaviorally_equivalent_mechanistic_drift": (
                behavior >= policy.drift_behavior_min and circuit_continuity < policy.drift_circuit_max
            ),
        })

    component_names = (
        "representation_continuity",
        "circuit_continuity",
        "intervention_continuity",
        "behavioral_continuity",
    )
    aggregate = {
        name: {
            "mean": sum(row[name] for row in rows) / len(rows),
            "minimum": min(row[name] for row in rows),
        }
        for name in component_names
    }
    payload: dict[str, Any] = {
        "schema": "ahcl.mechanistic-continuity.report.v1",
        "state_slice": STATE_SLICE,
        "checkpoint_ids": tuple(checkpoint_ids),
        "checkpoint_identities": tuple(
            {
                "checkpoint_id": point.checkpoint_id,
                "step": point.step,
                "state_digest": point.state_digest,
            }
            for point in points
        ),
        "checkpoint_measurement_digests": tuple(digest(point.as_dict()) for point in points),
        "lineage_digests": tuple(
            digest({
                "before_checkpoint": item.before_checkpoint,
                "after_checkpoint": item.after_checkpoint,
                "edges": tuple({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "weight": edge.weight,
                } for edge in item.edges),
            })
            for item in supplied
        ),
        "transition_count": len(rows),
        "policy": asdict(policy),
        "transitions": tuple(rows),
        "aggregate": aggregate,
        "all_invariants_pass": all(row["invariants_pass"] for row in rows),
        "mechanistic_drift_detected": any(row["behaviorally_equivalent_mechanistic_drift"] for row in rows),
        "nonclaims": (
            "aggregate caller-owned measurements only",
            "no mechanistic understanding claim",
            "no alignment or continual-learning superiority claim",
            "no model, provider, or production authorization",
        ),
    }
    payload["report_digest"] = digest(payload)
    return payload
