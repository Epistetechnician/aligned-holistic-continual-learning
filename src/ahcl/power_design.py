"""Conservative document-level power planning.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.

The independent unit is a document. Repeats are validated metadata and never
count as additional independent observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable


def _finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normal_cdf(value: float) -> float:
    return NormalDist().cdf(value)


def two_sided_power(*, clusters: int, effect_size: float, alpha: float, repeats: int = 1) -> float:
    """Normal-approximation power for a paired document contrast.

    ``effect_size`` is defined for the document-level paired contrast, so
    repeats do not alter variance or inflate sample size.
    """

    _positive_int(clusters, "clusters")
    _positive_int(repeats, "repeats")
    effect_size = _finite(effect_size, "effect_size")
    alpha = _finite(alpha, "alpha")
    if effect_size <= 0 or not 0 < alpha < 1:
        raise ValueError("effect_size must be positive and alpha in (0, 1)")
    noncentrality = effect_size * math.sqrt(clusters)
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return (1.0 - _normal_cdf(critical - noncentrality)) + _normal_cdf(-critical - noncentrality)


@dataclass(frozen=True)
class PowerDesign:
    """Frozen inputs and derived document counts for a multi-endpoint contrast."""

    effect_size: float
    alpha: float = 0.05
    target_power: float = 0.80
    endpoints: int = 1
    repeats: int = 1
    joint_target: float | None = None
    max_clusters: int = 1_000_000

    def __post_init__(self) -> None:
        _finite(self.effect_size, "effect_size")
        _finite(self.alpha, "alpha")
        _finite(self.target_power, "target_power")
        if self.joint_target is not None:
            _finite(self.joint_target, "joint_target")
        _positive_int(self.endpoints, "endpoints")
        _positive_int(self.repeats, "repeats")
        _positive_int(self.max_clusters, "max_clusters")
        if self.effect_size <= 0 or not 0 < self.alpha < 1 or not 0 < self.target_power <= 1:
            raise ValueError("effect_size must be positive, alpha in (0, 1), and target_power in (0, 1]")
        if self.joint_target is not None and not 0 < self.joint_target <= 1:
            raise ValueError("joint_target must be in (0, 1]")

    @property
    def per_contrast_alpha(self) -> float:
        """Bonferroni alpha, conservatively usable before Holm adjustment."""

        return self.alpha / self.endpoints

    @property
    def marginal_required_clusters(self) -> int:
        """Smallest bounded document count meeting target power per endpoint."""

        return self._search(lambda n: self.marginal_power(n) >= self.target_power)

    @property
    def joint_required_clusters(self) -> int | None:
        """Smallest bounded count whose union-bound joint lower bound meets target."""

        if self.joint_target is None:
            return None
        return self._search(lambda n: self.joint_lower_bound(n) >= self.joint_target)

    def required_families_per_cohort(self, *, families_per_document: int) -> int:
        """Return the family count needed in each cohort for the joint target.

        The conversion is document based: one independent document contributes
        ``families_per_document`` repeated endpoint families. Repeats remain
        metadata and never enter this conversion.
        """

        if self.joint_target is None:
            raise ValueError("joint_target is required for a family-wise cohort size")
        _positive_int(families_per_document, "families_per_document")
        return self.joint_required_clusters * families_per_document

    def marginal_power(self, clusters: int) -> float:
        return two_sided_power(
            clusters=clusters, effect_size=self.effect_size, alpha=self.per_contrast_alpha,
            repeats=self.repeats
        )

    def joint_lower_bound(self, clusters: int) -> float:
        """Bonferroni/union-bound lower bound, with no independence assumption."""

        marginal = self.marginal_power(clusters)
        return max(0.0, 1.0 - self.endpoints * (1.0 - marginal))

    def _search(self, predicate: Callable[[int], bool]) -> int:
        if not predicate(self.max_clusters):
            raise ValueError("required independent document count exceeds max_clusters")
        low, high = 1, self.max_clusters
        while low < high:
            middle = (low + high) // 2
            if predicate(middle):
                high = middle
            else:
                low = middle + 1
        return low
