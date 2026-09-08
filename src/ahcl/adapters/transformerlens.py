"""TransformerLens 3 bridge adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..interfaces import ActivationBackend, ActivationRecord
from .common import (
    OptionalDependencyError,
    optional_import,
    require_enabled,
    run_id,
    tensor_to_tree,
    validate_imported_module,
    validate_local_custody_file,
)


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
        revision: str | None = None,
        model_root: str | Path | None = None,
        model_weights_sha256: str | None = None,
        model_weights_name: str = "model.safetensors",
        source_root: str | Path | None = None,
        device: str = "cpu",
        enabled: bool = False,
    ) -> "TransformerLensActivationAdapter":
        require_enabled(enabled, "TransformerLens model execution")
        if not revision or model_root is None or not model_weights_sha256 or source_root is None:
            raise RuntimeError("exact model revision, local custody root, weight digest, and source root are required")
        try:
            module = optional_import("transformer_lens.model_bridge", "TransformerLens")
            validate_imported_module(module, source_root, "TransformerLens")
            torch = optional_import("torch", "PyTorch")
            transformers = optional_import("transformers", "Transformers")
            root = Path(model_root).expanduser().resolve()
            validate_local_custody_file(root / model_weights_name, model_weights_sha256, "model weights")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(root), revision=revision, local_files_only=True
            )
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                str(root), revision=revision, local_files_only=True, torch_dtype=torch.float32
            )
            bridge = module.TransformerBridge.boot_transformers(
                model_name,
                revision=revision,
                device=device,
                dtype=torch.float32,
                hf_model=hf_model,
                tokenizer=tokenizer,
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
