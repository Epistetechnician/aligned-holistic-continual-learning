"""SAELens pretrained SAE adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any

from ..interfaces import FeatureBackend, FeatureRecord
from .common import optional_import, require_enabled, tensor_to_tree, tree_to_tensor


class SAELensFeatureAdapter(FeatureBackend):
    """Encode and decode a caller-owned activation record with SAELens."""

    def __init__(self, sae: Any, layer: int = 0):
        self.sae = sae
        self.layer = layer

    @classmethod
    def from_pretrained(
        cls,
        release: str = "gpt2-small-mlp-out-v5-32k",
        sae_id: str = "layer_6",
        *,
        device: str = "cpu",
        layer: int = 6,
        enabled: bool = False,
    ) -> "SAELensFeatureAdapter":
        require_enabled(enabled, "SAELens asset loading")
        module = optional_import("sae_lens", "SAELens")
        sae = module.SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        return cls(sae, layer)

    def encode(self, activation: Any) -> FeatureRecord:
        torch = optional_import("torch", "PyTorch")
        values = tree_to_tensor(activation.values, dtype=torch.float32, device=getattr(self.sae, "device", None))
        features = self.sae.encode(values)
        feature_count = int(features.shape[-1])
        return FeatureRecord(self.layer, tuple(range(feature_count)), tensor_to_tree(features))

    def decode(self, features: FeatureRecord) -> Any:
        values = tree_to_tensor(features.values, device=getattr(self.sae, "device", None))
        from ..interfaces import ActivationRecord

        decoded = self.sae.decode(values)
        return ActivationRecord(f"sae:{self.layer}", f"layer:{self.layer}", tensor_to_tree(decoded))
