"""SAELens pretrained SAE adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..interfaces import FeatureBackend, FeatureRecord
from .common import (
    optional_import,
    require_enabled,
    tensor_to_tree,
    tree_to_tensor,
    validate_imported_module,
    validate_local_custody_file,
)


class SAELensFeatureAdapter(FeatureBackend):
    """Encode and decode a caller-owned activation record with SAELens."""

    def __init__(self, sae: Any, layer: int = 0):
        self.sae = sae
        self.layer = layer

    @classmethod
    def from_pretrained(
        cls,
        release: str = "gpt2-small-mlp-out-v5-32k",
        sae_id: str = "v5_32k_layer_6",
        *,
        asset_root: str | Path | None = None,
        weights_sha256: str | None = None,
        weights_name: str = "sae_weights.safetensors",
        source_root: str | Path | None = None,
        device: str = "cpu",
        layer: int = 6,
        enabled: bool = False,
    ) -> "SAELensFeatureAdapter":
        require_enabled(enabled, "SAELens asset loading")
        if asset_root is None or not weights_sha256 or source_root is None:
            raise RuntimeError("exact local SAE custody root, weight digest, and source root are required")
        module = optional_import("sae_lens", "SAELens")
        validate_imported_module(module, source_root, "SAELens")
        root = Path(asset_root).expanduser().resolve()
        validate_local_custody_file(root / weights_name, weights_sha256, "SAE weights")
        if not hasattr(module.SAE, "load_from_disk"):
            raise RuntimeError("the pinned SAELens environment lacks local SAE loading")
        sae = module.SAE.load_from_disk(root, device=device)
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
