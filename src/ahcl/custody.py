"""External custody manifest creation and validation.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .canonical import digest


class CustodyError(ValueError):
    """Raised when a custody root or manifest is invalid."""


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_manifest(root: Path, files: list[Path], *, state_slice: str) -> dict[str, Any]:
    root = root.resolve()
    if (root.stat().st_mode & 0o777) != 0o700:
        raise CustodyError("custody root must have owner-only 0700 permissions")
    entries = []
    seen_paths: set[str] = set()
    for path in sorted(files):
        if path.is_symlink():
            raise CustodyError("manifest files must be regular non-symlink files")
        resolved = path.resolve()
        if root not in resolved.parents:
            raise CustodyError("manifest file is outside custody root")
        if not resolved.is_file():
            raise CustodyError("manifest files must be regular non-symlink files")
        relative = str(resolved.relative_to(root))
        if relative in seen_paths:
            raise CustodyError("manifest files must have unique relative paths")
        seen_paths.add(relative)
        entries.append({"path": relative, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)})
    payload = {
        "schema": "ahcl.custody.manifest.v1",
        "state_slice": state_slice,
        "root": str(root),
        "files": entries,
    }
    payload["manifest_digest"] = digest(payload)
    return payload


def validate_manifest(manifest: dict[str, Any], *, expected_root: Path | None = None) -> None:
    required = {"schema", "state_slice", "root", "files", "manifest_digest"}
    if not required.issubset(manifest) or manifest["schema"] != "ahcl.custody.manifest.v1":
        raise CustodyError("custody manifest schema mismatch")
    root = Path(manifest["root"]).resolve()
    if expected_root is not None and root != expected_root.resolve():
        raise CustodyError("custody root mismatch")
    if not root.is_dir() or (root.stat().st_mode & 0o777) != 0o700:
        raise CustodyError("custody root is missing or not owner-only")
    if digest({key: value for key, value in manifest.items() if key != "manifest_digest"}) != manifest["manifest_digest"]:
        raise CustodyError("custody manifest digest mismatch")
    files = manifest["files"]
    if not isinstance(files, list):
        raise CustodyError("manifest files must be a list")
    seen_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise CustodyError("custody file entry schema mismatch")
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen_paths
        ):
            raise CustodyError("custody file paths must be unique and relative")
        if type(entry["bytes"]) is not int or entry["bytes"] < 0 or not _canonical_sha256(entry["sha256"]):
            raise CustodyError("custody file size or hash is malformed")
        seen_paths.add(relative)
        candidate = root / relative
        if candidate.is_symlink():
            raise CustodyError("custody entry is a symlink")
        path = candidate.resolve()
        if root not in path.parents or not path.is_file():
            raise CustodyError("custody entry escapes root or is not a regular file")
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise CustodyError(f"custody hash mismatch for {relative}")


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
