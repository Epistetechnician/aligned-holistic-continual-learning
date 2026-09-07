"""Vendor-neutral ports for future model-backed interpretability adapters.

State slice: aligned-holistic-continual-learning-clean-room-v1.
The ports deliberately carry typed records rather than owning storage or
provider credentials. All registrations are disabled in this slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ActivationRecord:
    """An in-memory activation batch owned by the caller."""

    run_id: str
    module: str
    values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class FeatureRecord:
    """A feature-basis representation returned by a dictionary adapter."""

    layer: int
    feature_ids: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]


class ActivationBackend(Protocol):
    def capture(self, input_ids: Sequence[Sequence[int]], modules: Sequence[str]) -> tuple[ActivationRecord, ...]:
        """Capture named module activations for one caller-owned run."""


class FeatureBackend(Protocol):
    def encode(self, activation: ActivationRecord) -> FeatureRecord:
        """Encode activations into a declared feature basis."""

    def decode(self, features: FeatureRecord) -> ActivationRecord:
        """Decode features for a reconstruction check."""


class InterventionBackend(Protocol):
    def replace(
        self,
        activation: ActivationRecord,
        donor: ActivationRecord,
        position: int,
    ) -> ActivationRecord:
        """Return a new activation record after a declared replacement."""


class BehaviorBackend(Protocol):
    def score(self, input_ids: Sequence[Sequence[int]]) -> Mapping[str, float]:
        """Return named behavior metrics without mutating the model."""


@dataclass(frozen=True)
class BackendRegistration:
    name: str
    source_url: str
    purpose: str
    enabled: bool = False


PUBLIC_BACKENDS = (
    BackendRegistration(
        "transformerlens",
        "https://github.com/TransformerLensOrg/TransformerLens",
        "activation cache and edit reference",
    ),
    BackendRegistration(
        "saelens",
        "https://github.com/decoderesearch/SAELens",
        "sparse feature dictionary reference",
    ),
    BackendRegistration(
        "circuit-tracer",
        "https://github.com/safety-research/circuit-tracer",
        "attribution graph and intervention reference",
    ),
    BackendRegistration(
        "nnsight",
        "https://github.com/ndif-team/nnsight",
        "PyTorch tracing reference",
    ),
    BackendRegistration(
        "pyvene",
        "https://github.com/stanfordnlp/pyvene",
        "intervention composition reference",
    ),
)


def registry_is_quarantined() -> bool:
    """Return whether every public adapter remains disabled in this slice."""

    return all(not registration.enabled for registration in PUBLIC_BACKENDS)
