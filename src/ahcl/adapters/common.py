"""Shared lazy-import and tensor conversion helpers for upstream adapters.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

import importlib
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping


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


def validate_imported_module(
    module: Any,
    source_root: str | Path,
    label: str,
    *,
    expected_files: Mapping[str, str] | None = None,
) -> Path:
    """Require an imported upstream module to resolve inside its pinned checkout.

    When ``expected_files`` is supplied, the checkout is also required to have
    that exact regular-file roster and every listed byte must match its digest.
    """

    raw_root = Path(source_root).expanduser()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise RuntimeError(f"{label} source root must be a real directory")
    root = raw_root.resolve()
    location = getattr(module, "__file__", None)
    if not isinstance(location, (str, Path)):
        raise RuntimeError(f"{label} module location is unavailable")
    raw_module_path = Path(location).expanduser()
    if raw_module_path.is_symlink():
        raise RuntimeError(f"{label} module location is a symlink")
    module_path = raw_module_path.resolve()
    if root != module_path and root not in module_path.parents:
        raise RuntimeError(f"{label} resolved outside its pinned checkout")
    if not module_path.is_file():
        raise RuntimeError(f"{label} module location is not a regular file")
    if expected_files is not None:
        checked = validate_local_custody_tree(
            root, expected_files, label, ignore_python_bytecode=True
        )
        if module_path not in checked:
            raise RuntimeError(f"{label} module bytes are not present in the locked source manifest")
    return module_path


def validate_local_custody_tree(
    root: str | Path,
    expected_files: Mapping[str, str],
    label: str,
    *,
    ignore_python_bytecode: bool = False,
) -> tuple[Path, ...]:
    """Require an exact regular-file roster and digest map below one custody directory."""

    raw_root = Path(root).expanduser()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise RuntimeError(f"{label} custody root must be a real directory")
    custody_root = raw_root.resolve()
    if not isinstance(expected_files, Mapping) or not expected_files:
        raise RuntimeError(f"{label} custody roster must be non-empty")
    normalized: dict[str, str] = {}
    for relative, checksum in expected_files.items():
        if not isinstance(relative, str) or not isinstance(checksum, str):
            raise RuntimeError(f"{label} custody roster contains an invalid path or digest")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
            or relative_path.as_posix() != relative
            or "\x00" in relative
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise RuntimeError(f"{label} custody roster contains an invalid path or digest")
        if ignore_python_bytecode and (
            relative_path.suffix in {".pyc", ".pyo"} or "__pycache__" in relative_path.parts
        ):
            raise RuntimeError(f"{label} source roster cannot contain Python bytecode")
        normalized[relative] = checksum
    actual: dict[str, Path] = {}
    for current, directories, files in os.walk(custody_root, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            candidate = current_path / name
            if candidate.is_symlink():
                raise RuntimeError(f"{label} custody tree contains a symlink")
        if ignore_python_bytecode:
            directories[:] = [name for name in directories if name != "__pycache__"]
            files = [name for name in files if Path(name).suffix not in {".pyc", ".pyo"}]
        for name in files:
            candidate = current_path / name
            if not candidate.is_file():
                raise RuntimeError(f"{label} custody tree contains a non-regular file")
            relative = candidate.relative_to(custody_root).as_posix()
            actual[relative] = candidate
    if set(actual) != set(normalized):
        raise RuntimeError(f"{label} custody file roster does not match the locked roster")
    checked: list[Path] = []
    for relative, checksum in sorted(normalized.items()):
        candidate = actual[relative]
        actual_checksum = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual_checksum != checksum:
            raise RuntimeError(f"{label} custody digest does not match for {relative}")
        checked.append(candidate.resolve())
    return tuple(checked)


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
