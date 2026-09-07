"""TransformerLens 3 bridge adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..interfaces import ActivationBackend, ActivationRecord
from .common import OptionalDependencyError, optional_import, require_enabled, run_id, tensor_to_tree


class TransformerLensActivationAdapter(ActivationBackend):
    """Capture named activations through the pinned TransformerBridge API."""

    def __init__(self, bridge: Any, model_name: str, revision: str):
        self.bridge = bridge
        self.model_name = model_name
        self.revision = revision

    @classmethod
    def boot(
        cls,
        model_name: str = "openai-community/gpt2",
        *,
        revision: str = "main",
        device: str = "cpu",
        enabled: bool = False,
    ) -> "TransformerLensActivationAdapter":
        require_enabled(enabled, "TransformerLens model execution")
        try:
            module = optional_import("transformer_lens.model_bridge", "TransformerLens")
            torch = optional_import("torch", "PyTorch")
            bridge = module.TransformerBridge.boot_transformers(
                model_name,
                revision=revision,
                device=device,
                dtype=torch.float32,
            )
        except OptionalDependencyError:
            raise
        return cls(bridge, model_name, revision)

    def capture(self, input_ids: Sequence[Sequence[int]], modules: Sequence[str]) -> tuple[ActivationRecord, ...]:
        if not modules:
            raise ValueError("at least one module name is required")
        torch = optional_import("torch", "PyTorch")
        tokens = input_ids if hasattr(input_ids, "shape") else torch.tensor(input_ids, dtype=torch.long)
        _, cache = self.bridge.run_with_cache(tokens, return_cache_object=False, names_filter=list(modules))
        identity = run_id(self.model_name, self.revision, input_ids, tuple(modules))
        return tuple(
            ActivationRecord(identity, module, tensor_to_tree(cache[module])) for module in modules
        )

    def logits(self, input_ids: Any) -> Any:
        """Run the native bridge forward pass without retaining activations."""

        return self.bridge(input_ids)
