"""Circuit-tracer ReplacementModel adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..interfaces import FeatureInterventionBackend
from .common import optional_import, require_enabled


class CircuitTracerAdapter(FeatureInterventionBackend):
    """Expose circuit-tracer feature interventions behind one typed boundary."""

    def __init__(self, replacement_model: Any, model_name: str, transcoder_set: str, backend: str):
        self.model = replacement_model
        self.model_name = model_name
        self.transcoder_set = transcoder_set
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "openai-community/gpt2",
        transcoder_set: str = "",
        *,
        backend: str = "transformerlens",
        device: str = "cpu",
        enabled: bool = False,
    ) -> "CircuitTracerAdapter":
        require_enabled(enabled, "circuit-tracer model execution")
        if not transcoder_set:
            raise ValueError("transcoder_set must be an exact local path or public hub identifier")
        module = optional_import("circuit_tracer.replacement_model", "circuit-tracer")
        torch = optional_import("torch", "PyTorch")
        model = module.ReplacementModel.from_pretrained(
            model_name,
            transcoder_set=transcoder_set,
            backend=backend,
            device=torch.device(device),
        )
        return cls(model, model_name, transcoder_set, backend)

    def feature_intervention(
        self,
        input_ids: Any,
        interventions: Sequence[tuple[int, int, int, float]],
        **kwargs: Any,
    ) -> Any:
        if not interventions:
            raise ValueError("at least one intervention is required for a causal trial")
        return self.model.feature_intervention(input_ids, list(interventions), **kwargs)
