"""Shared lazy-import and tensor conversion helpers for upstream adapters.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

import importlib
import hashlib
from pathlib import Path
from typing import Any


class OptionalDependencyError(RuntimeError):
    """Raised when a selected adapter's environment is not installed."""


def require_enabled(enabled: bool, capability: str) -> None:
    if not enabled:
        raise RuntimeError(f"{capability} is closed until the independent review receipt is accepted")


def optional_import(module_name: str, package_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise OptionalDependencyError(
            f"{package_name} is required for this adapter; install the pinned environment first"
        ) from error


def validate_local_custody_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    """Require one regular local file whose bytes match a locked SHA-256."""

    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise RuntimeError(f"{label} must be a regular file, not a symlink")
    candidate = raw.resolve()
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or not candidate.is_file()
    ):
        raise RuntimeError(f"{label} must be an existing locally custodied file with a canonical SHA-256")
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"{label} custody digest does not match the locked SHA-256")
    return candidate


def validate_imported_module(module: Any, source_root: str | Path, label: str) -> Path:
    """Require an imported upstream module to resolve inside its pinned checkout."""

    root = Path(source_root).expanduser().resolve()
    location = getattr(module, "__file__", None)
    if not isinstance(location, str):
        raise RuntimeError(f"{label} module location is unavailable")
    module_path = Path(location).resolve()
    if root not in module_path.parents:
        raise RuntimeError(f"{label} resolved outside its pinned checkout")
    return module_path


def tensor_to_tree(value: Any) -> Any:
    """Convert a tensor-like value to immutable nested tuples for records."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, list):
        return tuple(tensor_to_tree(item) for item in value)
    if isinstance(value, tuple):
        return tuple(tensor_to_tree(item) for item in value)
    if isinstance(value, (int, float, bool)):
        return value
    raise TypeError(f"cannot canonicalize adapter value of type {type(value).__name__}")


def tree_to_tensor(value: Any, *, dtype: Any = None, device: Any = None) -> Any:
    torch = optional_import("torch", "PyTorch")
    tensor = torch.tensor(value, dtype=dtype)
    return tensor.to(device=device) if device is not None else tensor


def run_id(*parts: object) -> str:
    import hashlib

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
