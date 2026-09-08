"""Model-free comparison and resource contracts; not an execution benchmark.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from dataclasses import dataclass
import math
from typing import Mapping


BASELINES = {
    "sgd": "One gradient update per fresh batch; no replay or selection.",
    "replay": "SGD plus a fit-locked bounded reservoir and replay schedule.",
    "ewc": "SGD plus a fit-locked diagonal Fisher penalty and refresh schedule.",
    "behavioral_selection": "Candidate updates selected using permitted probe behavior only.",
    "causal_monitor": "Same candidates and probes plus locked causal feature measurements.",
}
COST_CATEGORIES = frozenset({
    "update", "replay", "fisher", "evaluation", "monitor", "candidate", "rollback", "setup"
})


def _integer(value: int, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("resource quantities must be nonnegative integers (caps positive)")


@dataclass(frozen=True)
class ResourceCaps:
    compute_ops: int
    peak_memory_bytes: int
    elapsed_ns: int

    def __post_init__(self):
        for value in (self.compute_ops, self.peak_memory_bytes, self.elapsed_ns):
            _integer(value, positive=True)


@dataclass(frozen=True)
class Cost:
    category: str
    compute_ops: int
    live_memory_bytes: int
    elapsed_ns: int

    def __post_init__(self):
        if self.category not in COST_CATEGORIES:
            raise ValueError("unrecognized cost category")
        for value in (self.compute_ops, self.live_memory_bytes, self.elapsed_ns):
            _integer(value)


class BudgetExceeded(RuntimeError):
    pass


class ResourceLedger:
    """Account all work, including rejected candidates; exceeding caps is terminal.

    Memory is total simultaneously live memory, including model/optimizer,
    feature dictionary, buffers, Fisher state and candidate copies, not a delta.
    The caller must reserve conservative bounds before execution using preflight;
    these model-free checks cannot interrupt a hardware operation themselves.
    """

    def __init__(self, caps: ResourceCaps):
        self.caps = caps
        self.compute_ops = self.elapsed_ns = self.peak_memory_bytes = 0
        self.stopped = False
        self.by_category: dict[str, int] = {}

    def preflight(self, bound: Cost) -> None:
        if self.stopped:
            raise BudgetExceeded("ledger is terminal")
        if (self.compute_ops + bound.compute_ops > self.caps.compute_ops
                or self.elapsed_ns + bound.elapsed_ns > self.caps.elapsed_ns
                or max(self.peak_memory_bytes, bound.live_memory_bytes) > self.caps.peak_memory_bytes):
            self.stopped = True
            raise BudgetExceeded("resource cap would be exceeded")

    def charge(self, actual: Cost) -> None:
        self.preflight(actual)
        self.compute_ops += actual.compute_ops
        self.elapsed_ns += actual.elapsed_ns
        self.peak_memory_bytes = max(self.peak_memory_bytes, actual.live_memory_bytes)
        self.by_category[actual.category] = self.by_category.get(actual.category, 0) + actual.compute_ops


@dataclass(frozen=True)
class Cohort:
    families: frozenset[str]
    documents: frozenset[str]
    authors: frozenset[str]


def validate_splits(splits: Mapping[str, Cohort]) -> None:
    if set(splits) != {"fit", "tune", "assessment"}:
        raise ValueError("exactly fit, tune and assessment required")
    for field in ("families", "documents", "authors"):
        seen: set[str] = set()
        for cohort in splits.values():
            values = getattr(cohort, field)
            if not values or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError("empty or invalid cohort identity")
            if seen.intersection(values):
                raise ValueError(f"split leakage in {field}")
            seen.update(values)


def validate_comparison(caps_by_arm: Mapping[str, ResourceCaps]) -> None:
    if set(caps_by_arm) != set(BASELINES):
        raise ValueError("complete baseline roster required")
    if len(set(caps_by_arm.values())) != 1:
        raise ValueError("all arms require identical resource caps")


def learning_floor_passes(*, gain_lower_bound: float, minimum_gain: float,
                          accepted_updates: int, ledger: ResourceLedger,
                          completed_stream: bool) -> bool:
    """Positive preregistered progress is mandatory, independently of retention.

    gain_lower_bound comes from a separately reviewed interval estimator on
    new-task progress against the frozen initial checkpoint. No estimator or
    scientific threshold is invented by this contract.
    """
    if not math.isfinite(minimum_gain) or minimum_gain <= 0:
        raise ValueError("a positive finite learning floor must be preregistered")
    _integer(accepted_updates)
    return (math.isfinite(gain_lower_bound) and gain_lower_bound >= minimum_gain
            and accepted_updates > 0 and not ledger.stopped and completed_stream)
