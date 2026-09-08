"""NNsight local tracing adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..interfaces import ActivationBackend, ActivationRecord
from .common import (
    optional_import,
    require_enabled,
    run_id,
    tensor_to_tree,
    validate_imported_module,
    validate_local_custody_file,
)


def _resolve_path(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


class NNsightActivationAdapter(ActivationBackend):
    """Capture saved module outputs using NNsight's trace context."""

    def __init__(self, model: Any, model_name: str):
        self.model = model
        self.model_name = model_name

    @classmethod
    def load(
        cls,
        model_name: str = "openai-community/gpt2",
        *,
        device_map: str = "auto",
        dispatch: bool = True,
        revision: str | None = None,
        model_root: str | Path | None = None,
        model_weights_sha256: str | None = None,
        source_root: str | Path | None = None,
        source_files: Mapping[str, str] | None = None,
        enabled: bool = False,
    ) -> "NNsightActivationAdapter":
        require_enabled(enabled, "NNsight model execution")
        if (
            not revision
            or model_root is None
            or not model_weights_sha256
            or source_root is None
            or source_files is None
        ):
            raise RuntimeError(
                "exact model revision, local custody root, weight digest, source root, and source manifest are required"
            )
        module = optional_import("nnsight", "NNsight")
        validate_imported_module(module, source_root, "NNsight", expected_files=source_files)
        transformers = optional_import("transformers", "Transformers")
        torch = optional_import("torch", "PyTorch")
        root = Path(model_root).expanduser().resolve()
        validate_local_custody_file(root / "model.safetensors", model_weights_sha256, "model weights")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(root), revision=revision, local_files_only=True
        )
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            str(root), revision=revision, local_files_only=True,
            device_map=device_map, torch_dtype=torch.float32,
        )
        return cls(module.LanguageModel(hf_model, tokenizer=tokenizer), model_name)

    def capture(self, input_ids: Sequence[Sequence[int]], modules: Sequence[str]) -> tuple[ActivationRecord, ...]:
        if not modules:
            raise ValueError("at least one module path is required")
        saved: list[tuple[str, Any]] = []
        with self.model.trace(input_ids):
            for module_path in modules:
                saved.append((module_path, _resolve_path(self.model, module_path).output.save()))
        identity = run_id(self.model_name, input_ids, tuple(modules))
        return tuple(ActivationRecord(identity, path, tensor_to_tree(value)) for path, value in saved)
