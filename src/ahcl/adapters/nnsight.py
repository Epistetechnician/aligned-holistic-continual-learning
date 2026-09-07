"""NNsight local tracing adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..interfaces import ActivationBackend, ActivationRecord
from .common import optional_import, require_enabled, run_id, tensor_to_tree


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
        enabled: bool = False,
    ) -> "NNsightActivationAdapter":
        require_enabled(enabled, "NNsight model execution")
        module = optional_import("nnsight", "NNsight")
        return cls(module.LanguageModel(model_name, device_map=device_map, dispatch=dispatch), model_name)

    def capture(self, input_ids: Sequence[Sequence[int]], modules: Sequence[str]) -> tuple[ActivationRecord, ...]:
        if not modules:
            raise ValueError("at least one module path is required")
        saved: list[tuple[str, Any]] = []
        with self.model.trace(input_ids):
            for module_path in modules:
                saved.append((module_path, _resolve_path(self.model, module_path).output.save()))
        identity = run_id(self.model_name, input_ids, tuple(modules))
        return tuple(ActivationRecord(identity, path, tensor_to_tree(value)) for path, value in saved)
