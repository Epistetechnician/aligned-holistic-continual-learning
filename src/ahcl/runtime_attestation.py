"""Launch-time attestation for isolated Python dependency lanes.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
This module records exact interpreter bytes, distribution versions, installed
file bytes, and imported module locations and bytes. It never downloads
packages or models.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_bytes


class RuntimeAttestationError(ValueError):
    """Raised when a runtime cannot satisfy its locked attestation contract."""


_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_VERSION = re.compile(r"\A\d+\.\d+\.\d+\Z")


@dataclass(frozen=True)
class RuntimeRequirement:
    """One distribution, import, version, and pinned source checkout."""

    distribution: str
    import_name: str
    version: str
    source_root: str | Path


@dataclass(frozen=True)
class RuntimeAttestation:
    """Canonical, immutable runtime evidence produced by :func:`attest_runtime`."""

    _canonical_payload: bytes

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RuntimeAttestation":
        try:
            encoded = canonical_bytes(dict(payload))
        except (TypeError, ValueError) as exc:
            raise RuntimeAttestationError("runtime attestation payload is not canonical JSON") from exc
        return cls(encoded)

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._canonical_payload.decode("utf-8"))

    @property
    def attestation_digest(self) -> str:
        return hashlib.sha256(self._canonical_payload).hexdigest()

    def document(self) -> dict[str, Any]:
        payload = self.payload
        payload["attestation_digest"] = self.attestation_digest
        return payload


def _regular_file_digest(path: Path, *, root: Path, label: str) -> tuple[str, str]:
    if path.is_symlink():
        raise RuntimeAttestationError(f"{label} contains a symlink")
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise RuntimeAttestationError(f"{label} escapes its distribution root")
    if not resolved.is_file():
        raise RuntimeAttestationError(f"{label} is missing or not a regular file")
    return str(resolved.relative_to(root)), hashlib.sha256(resolved.read_bytes()).hexdigest()


def _interpreter_identity() -> dict[str, str]:
    """Return the resolved interpreter path and bytes digest for this process."""

    raw = Path(sys.executable).expanduser()
    resolved = raw.resolve()
    if not resolved.is_file():
        raise RuntimeAttestationError("Python interpreter must be an existing regular file")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _attest_requirement(requirement: RuntimeRequirement) -> dict[str, Any]:
    if not isinstance(requirement, RuntimeRequirement):
        raise RuntimeAttestationError("runtime requirement must be a RuntimeRequirement")
    if (
        not isinstance(requirement.distribution, str)
        or not isinstance(requirement.import_name, str)
        or not isinstance(requirement.version, str)
        or not isinstance(requirement.source_root, (str, Path))
        or not requirement.distribution.strip()
        or not requirement.import_name.strip()
        or not _VERSION.fullmatch(requirement.version)
    ):
        raise RuntimeAttestationError("runtime requirement identity is malformed")
    source_root = requirement.source_root.expanduser()
    if source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeAttestationError("runtime source root must be a real directory")
    source_root = source_root.resolve()
    try:
        distribution = importlib.metadata.distribution(requirement.distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeAttestationError(f"missing runtime distribution: {requirement.distribution}") from exc
    if distribution.version != requirement.version:
        raise RuntimeAttestationError(
            f"runtime version mismatch for {requirement.distribution}: "
            f"expected {requirement.version}, found {distribution.version}"
        )
    files = distribution.files
    if not files:
        raise RuntimeAttestationError(f"runtime distribution has no file manifest: {requirement.distribution}")
    distribution_root = Path(distribution.locate_file("")).resolve()
    entries = []
    seen: set[str] = set()
    for entry in sorted(files, key=str):
        relative, checksum = _regular_file_digest(
            Path(distribution.locate_file(entry)),
            root=distribution_root,
            label=f"{requirement.distribution}:{entry}",
        )
        if relative in seen:
            raise RuntimeAttestationError(f"runtime distribution has duplicate path: {relative}")
        seen.add(relative)
        entries.append({"path": relative, "sha256": checksum})
    try:
        module = importlib.import_module(requirement.import_name)
    except ImportError as exc:
        raise RuntimeAttestationError(f"missing runtime import: {requirement.import_name}") from exc
    location = getattr(module, "__file__", None)
    if not isinstance(location, str):
        raise RuntimeAttestationError(f"runtime import has no file: {requirement.import_name}")
    module_raw = Path(location)
    if module_raw.is_symlink():
        raise RuntimeAttestationError(f"runtime import is a symlink: {requirement.import_name}")
    module_path = module_raw.resolve()
    if source_root != module_path and source_root not in module_path.parents:
        raise RuntimeAttestationError(f"runtime import escaped pinned source root: {requirement.import_name}")
    if not module_path.is_file():
        raise RuntimeAttestationError(f"runtime import is not a regular file: {requirement.import_name}")
    return {
        "distribution": requirement.distribution,
        "version": requirement.version,
        "import_name": requirement.import_name,
        "module_file": str(module_path.relative_to(source_root)),
        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "files": entries,
    }


def attest_runtime(
    *,
    state_slice: str,
    expected_python: str,
    requirements: Sequence[RuntimeRequirement],
) -> RuntimeAttestation:
    """Hash the current interpreter and every file in each required distribution."""

    if (
        not isinstance(state_slice, str)
        or not isinstance(expected_python, str)
        or not state_slice.strip()
        or not _VERSION.fullmatch(expected_python)
    ):
        raise RuntimeAttestationError("runtime attestation identity is malformed")
    if platform.python_version() != expected_python:
        raise RuntimeAttestationError(
            f"Python version mismatch: expected {expected_python}, found {platform.python_version()}"
        )
    if not requirements:
        raise RuntimeAttestationError("at least one runtime distribution is required")
    interpreter = _interpreter_identity()
    distributions = [_attest_requirement(requirement) for requirement in requirements]
    names = [entry["distribution"] for entry in distributions]
    if len(set(names)) != len(names):
        raise RuntimeAttestationError("runtime distributions must be unique")
    payload = {
        "schema": "ahcl.runtime.attestation.v1",
        "state_slice": state_slice,
        "python": expected_python,
        "interpreter": interpreter,
        "distributions": {entry["distribution"]: entry for entry in distributions},
    }
    return RuntimeAttestation.from_payload(payload)


def validate_runtime_attestation(
    attestation: RuntimeAttestation,
    *,
    state_slice: str,
    expected_python: str,
) -> None:
    """Validate an attestation's canonical schema before model execution."""

    if not isinstance(attestation, RuntimeAttestation):
        raise RuntimeAttestationError("a RuntimeAttestation produced by attest_runtime is required")
    payload = attestation.payload
    if set(payload) != {"schema", "state_slice", "python", "interpreter", "distributions"}:
        raise RuntimeAttestationError("runtime attestation schema mismatch")
    if (
        not isinstance(state_slice, str)
        or not isinstance(payload["schema"], str)
        or not isinstance(payload["state_slice"], str)
        or payload["schema"] != "ahcl.runtime.attestation.v1"
        or payload["state_slice"] != state_slice
    ):
        raise RuntimeAttestationError("runtime attestation state slice mismatch")
    if (
        not isinstance(expected_python, str)
        or not isinstance(payload["python"], str)
        or payload["python"] != expected_python
        or not _VERSION.fullmatch(expected_python)
    ):
        raise RuntimeAttestationError("runtime attestation Python identity mismatch")
    interpreter = payload["interpreter"]
    if not isinstance(interpreter, dict) or set(interpreter) != {"path", "sha256"}:
        raise RuntimeAttestationError("runtime attestation interpreter identity is malformed")
    if not isinstance(interpreter["path"], str) or not interpreter["path"]:
        raise RuntimeAttestationError("runtime attestation interpreter path is malformed")
    interpreter_path = Path(interpreter["path"])
    if not interpreter_path.is_absolute() or ".." in interpreter_path.parts:
        raise RuntimeAttestationError("runtime attestation interpreter path is malformed")
    if not isinstance(interpreter["sha256"], str) or not _SHA256.fullmatch(interpreter["sha256"]):
        raise RuntimeAttestationError("runtime attestation interpreter hash is malformed")
    distributions = payload["distributions"]
    if not isinstance(distributions, dict) or not distributions:
        raise RuntimeAttestationError("runtime attestation distribution roster is empty")
    for name, entry in distributions.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise RuntimeAttestationError("runtime attestation distribution entry is malformed")
        if set(entry) != {"distribution", "version", "import_name", "module_file", "module_sha256", "files"}:
            raise RuntimeAttestationError("runtime attestation distribution schema mismatch")
        if (
            entry["distribution"] != name
            or not isinstance(entry["version"], str)
            or not _VERSION.fullmatch(entry["version"])
        ):
            raise RuntimeAttestationError("runtime attestation distribution identity is malformed")
        if not isinstance(entry["import_name"], str) or not entry["import_name"].strip():
            raise RuntimeAttestationError("runtime attestation import identity is malformed")
        if not isinstance(entry["module_file"], str) or not entry["module_file"]:
            raise RuntimeAttestationError("runtime attestation module path is not relative")
        module_file = Path(entry["module_file"])
        if module_file.is_absolute() or ".." in module_file.parts:
            raise RuntimeAttestationError("runtime attestation module path is not relative")
        if not isinstance(entry["module_sha256"], str) or not _SHA256.fullmatch(entry["module_sha256"]):
            raise RuntimeAttestationError("runtime attestation module hash is malformed")
        files = entry["files"]
        if not isinstance(files, list) or not files:
            raise RuntimeAttestationError("runtime attestation file roster is empty")
        seen: set[str] = set()
        for file_entry in files:
            if not isinstance(file_entry, dict) or set(file_entry) != {"path", "sha256"}:
                raise RuntimeAttestationError("runtime attestation file entry schema mismatch")
            if not isinstance(file_entry["path"], str) or not file_entry["path"]:
                raise RuntimeAttestationError("runtime attestation file path is malformed")
            path = Path(file_entry["path"])
            if path.is_absolute() or not path.parts or ".." in path.parts or str(path) in seen:
                raise RuntimeAttestationError("runtime attestation file path is malformed")
            if not isinstance(file_entry["sha256"], str) or not _SHA256.fullmatch(file_entry["sha256"]):
                raise RuntimeAttestationError("runtime attestation file hash is malformed")
            seen.add(str(path))


__all__ = [
    "RuntimeAttestation",
    "RuntimeAttestationError",
    "RuntimeRequirement",
    "attest_runtime",
    "validate_runtime_attestation",
]


def _parse_requirement(raw: str) -> RuntimeRequirement:
    parts = raw.split(",", 3)
    if len(parts) != 4:
        raise RuntimeAttestationError(
            "--requirement must be DISTRIBUTION,IMPORT_NAME,VERSION,SOURCE_ROOT"
        )
    distribution, import_name, version, source_root = parts
    return RuntimeRequirement(distribution, import_name, version, Path(source_root))


def main(argv: Sequence[str] | None = None) -> int:
    """Emit one canonical runtime attestation document as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-slice", required=True)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument(
        "--requirement",
        action="append",
        required=True,
        metavar="DISTRIBUTION,IMPORT_NAME,VERSION,SOURCE_ROOT",
    )
    args = parser.parse_args(argv)
    try:
        attestation = attest_runtime(
            state_slice=args.state_slice,
            expected_python=args.expected_python,
            requirements=tuple(_parse_requirement(raw) for raw in args.requirement),
        )
    except RuntimeAttestationError as error:
        parser.error(str(error))
    print(json.dumps(attestation.document(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
