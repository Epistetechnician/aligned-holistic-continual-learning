"""Circuit-tracer ReplacementModel adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..interfaces import FeatureInterventionBackend
from .common import (
    optional_import,
    require_enabled,
    validate_imported_module,
    validate_local_custody_file,
    validate_local_custody_tree,
)


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
        revision: str | None = None,
        model_root: str | Path | None = None,
        model_weights_sha256: str | None = None,
        source_root: str | Path | None = None,
        transcoder_root: str | Path | None = None,
        transcoder_files: Mapping[str, str] | None = None,
        enabled: bool = False,
    ) -> "CircuitTracerAdapter":
        require_enabled(enabled, "circuit-tracer model execution")
        if (
            not transcoder_set or not revision or model_root is None or not model_weights_sha256
            or source_root is None or transcoder_root is None or transcoder_files is None
        ):
            raise RuntimeError("exact local model and transcoder custody identities are required")
        if Path(transcoder_set).expanduser().resolve() != Path(transcoder_root).expanduser().resolve():
            raise RuntimeError("transcoder_set must name the locked local transcoder root")
        module = optional_import("circuit_tracer.replacement_model", "circuit-tracer")
        validate_imported_module(module, source_root, "circuit-tracer")
        torch = optional_import("torch", "PyTorch")
        transformers = optional_import("transformers", "Transformers")
        model_path = Path(model_root).expanduser().resolve()
        validate_local_custody_file(model_path / "model.safetensors", model_weights_sha256, "model weights")
        validate_local_custody_tree(transcoder_root, transcoder_files, "circuit-tracer transcoder")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_path), revision=revision, local_files_only=True
        )
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path), revision=revision, local_files_only=True, torch_dtype=torch.float32
        )
        yaml = optional_import("yaml", "PyYAML")
        config_path = Path(transcoder_root).expanduser().resolve() / "config.yaml"
        if "config.yaml" not in transcoder_files:
            raise RuntimeError("the locked transcoder roster must include config.yaml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("subfolder"):
            raise RuntimeError("local circuit-tracer config must be a mapping without subfolder resolution")
        model_kind = config.get("model_kind")
        if model_kind not in {"transcoder_set", "cross_layer_transcoder"}:
            raise RuntimeError("local circuit-tracer config has an unsupported model_kind")
        local_root = Path(transcoder_root).expanduser().resolve()
        paths = config.get("transcoders")
        if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
            raise RuntimeError("local circuit-tracer config must list transcoder files")
        resolved_paths: list[str] = []
        for relative in paths:
            if relative.startswith("hf://") or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RuntimeError("local circuit-tracer config cannot reference remote or escaping files")
            candidate = (local_root / relative).resolve()
            if local_root not in candidate.parents or not candidate.is_file():
                raise RuntimeError("local circuit-tracer config references a missing file")
            if str(candidate.relative_to(local_root)) not in transcoder_files:
                raise RuntimeError("every configured transcoder file must be in the locked roster")
            resolved_paths.append(str(candidate))
        torch_device = torch.device(device)
        scan_name = f"local:{local_root.name}@{revision}"
        if model_kind == "transcoder_set":
            transcoder_module = optional_import(
                "circuit_tracer.transcoder.single_layer_transcoder", "circuit-tracer"
            )
            special_load_fn = config.get("special_load_fn")
            if special_load_fn not in {None, "gemma-scope", "gemma-scope-2"}:
                raise RuntimeError("local transcoder special_load_fn is unsupported")
            transcoder_config = transcoder_module.load_transcoder_set(
                {index: path for index, path in enumerate(resolved_paths)},
                scan_name=scan_name,
                feature_input_hook=config.get("feature_input_hook", "hook_resid_mid"),
                feature_output_hook=config.get("feature_output_hook", "hook_mlp_out"),
                special_load_fn=special_load_fn,
                activation=config.get("activation"),
                k=config.get("k"),
                device=torch_device,
                dtype=torch.float32,
                lazy_encoder=False,
                lazy_decoder=False,
            )
        else:
            transcoder_module = optional_import(
                "circuit_tracer.transcoder.cross_layer_transcoder", "circuit-tracer"
            )
            if config.get("format") == "gemma-scope-2":
                transcoder_config = transcoder_module.load_gemma_scope_2_clt(
                    {index: path for index, path in enumerate(resolved_paths)},
                    scan_name=scan_name,
                    feature_input_hook=config.get("feature_input_hook", "hook_resid_mid"),
                    feature_output_hook=config.get("feature_output_hook", "hook_mlp_out"),
                    device=torch_device,
                    dtype=torch.float32,
                    lazy_encoder=False,
                    lazy_decoder=False,
                )
            else:
                transcoder_config = transcoder_module.load_clt(
                    str(local_root),
                    scan_name=scan_name,
                    feature_input_hook=config.get("feature_input_hook", "hook_resid_mid"),
                    feature_output_hook=config.get("feature_output_hook", "hook_mlp_out"),
                    device=torch_device,
                    dtype=torch.float32,
                    lazy_encoder=False,
                    lazy_decoder=False,
                )
        replacement_module = optional_import("circuit_tracer.replacement_model", "circuit-tracer")
        replacement_model = replacement_module.ReplacementModel.from_pretrained_and_transcoders(
            model_name=str(model_path),
            transcoders=transcoder_config,
            backend=backend,
            device=torch_device,
            dtype=torch.float32,
            hf_model=hf_model,
            tokenizer=tokenizer,
            local_files_only=True,
        )
        return cls(replacement_model, model_name, transcoder_set, backend)

    def feature_intervention(
        self,
        input_ids: Any,
        interventions: Sequence[tuple[int, int, int, float]],
        **kwargs: Any,
    ) -> Any:
        if not interventions:
            raise ValueError("at least one intervention is required for a causal trial")
        return self.model.feature_intervention(input_ids, list(interventions), **kwargs)
