"""ACDC circuit-discovery adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Callable

from ..interfaces import CircuitDiscoveryBackend
from .common import optional_import, require_enabled


class ACDCDiscoveryAdapter(CircuitDiscoveryBackend):
    """Run the pinned ACDC experiment with caller-owned model and tensors."""

    def __init__(self, experiment: Any):
        self.experiment = experiment

    @classmethod
    def prepare(
        cls,
        model: Any,
        dataset: Any,
        reference_dataset: Any,
        *,
        threshold: float,
        metric: Callable[[Any], Any],
        zero_ablation: bool = False,
        enabled: bool = False,
    ) -> "ACDCDiscoveryAdapter":
        require_enabled(enabled, "ACDC model discovery")
        module = optional_import("acdc.TLACDCExperiment", "ACDC")
        experiment = module.TLACDCExperiment(
            model,
            dataset,
            reference_dataset,
            threshold,
            metric,
            zero_ablation=zero_ablation,
            using_wandb=False,
            online_cache_cpu=True,
            corrupted_cache_cpu=True,
        )
        return cls(experiment)

    def discover(self, *, max_steps: int = 100_000) -> Any:
        for _ in range(max_steps):
            if self.experiment.current_node is None:
                return self.experiment.save_subgraph(return_it=True)
            self.experiment.step()
        raise TimeoutError("ACDC discovery exceeded the locked step budget")
