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
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_bytes


class RuntimeAttestationError(ValueError):
    """Raised when a runtime cannot satisfy its locked attestation contract."""


_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_VERSION = re.compile(r"\A\d+\.\d+\.\d+\Z")
_RUNTIME_FIELDS = {
    "modern": {"transformers": "transformers", "torch": "torch"},
    "circuit_tracer": {
        "transformers": "circuit_tracer_transformers",
        "transformer-lens": "circuit_tracer_transformer_lens",
        "torch": "circuit_tracer_torch",
    },
    "acdc": {
        "numpy": "acdc_numpy",
        "transformers": "acdc_transformers",
        "transformer-lens": "acdc_transformer_lens",
        "torch": "acdc_torch",
    },
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


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

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "RuntimeAttestation":
        """Load and verify a JSON document emitted by :meth:`document`."""

        if not isinstance(document, Mapping):
            raise RuntimeAttestationError("runtime attestation document must be an object")
        payload = dict(document)
        if set(payload) != {"schema", "state_slice", "python", "interpreter", "distributions", "attestation_digest"}:
            raise RuntimeAttestationError("runtime attestation document schema mismatch")
        expected = payload.pop("attestation_digest")
        attestation = cls.from_payload(payload)
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise RuntimeAttestationError("runtime attestation document digest is malformed")
        if attestation.attestation_digest != expected:
            raise RuntimeAttestationError("runtime attestation document digest mismatch")
        return attestation

    @classmethod
    def from_json(cls, raw: str | bytes) -> "RuntimeAttestation":
        """Load a document from JSON without allowing duplicate object keys."""

        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            document = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except (TypeError, UnicodeDecodeError, ValueError) as exc:
            raise RuntimeAttestationError("runtime attestation JSON is malformed") from exc
        return cls.from_document(document)

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


@dataclass(frozen=True)
class RuntimeLaneAttestation:
    """Canonical evidence binding one runtime attestation to a declared lane."""

    _canonical_payload: bytes

    @classmethod
    def from_attestation(
        cls,
        *,
        lane: str,
        state_slice: str,
        attestation: RuntimeAttestation,
    ) -> "RuntimeLaneAttestation":
        if not isinstance(lane, str) or not lane.strip():
            raise RuntimeAttestationError("runtime lane identity is malformed")
        if not isinstance(state_slice, str) or not state_slice.strip():
            raise RuntimeAttestationError("runtime lane state slice is malformed")
        if not isinstance(attestation, RuntimeAttestation):
            raise RuntimeAttestationError("runtime lane requires a RuntimeAttestation")
        payload = {
            "schema": "ahcl.runtime.lane-attestation.v1",
            "state_slice": state_slice,
            "lane": lane,
            "runtime": attestation.document(),
        }
        return cls(canonical_bytes(payload))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RuntimeLaneAttestation":
        try:
            encoded = canonical_bytes(dict(payload))
        except (TypeError, ValueError) as exc:
            raise RuntimeAttestationError("runtime lane attestation payload is not canonical JSON") from exc
        return cls(encoded)

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "RuntimeLaneAttestation":
        """Load and verify a JSON document emitted by :meth:`document`."""

        if not isinstance(document, Mapping):
            raise RuntimeAttestationError("runtime lane attestation document must be an object")
        payload = dict(document)
        if set(payload) != {"schema", "state_slice", "lane", "runtime", "lane_attestation_digest"}:
            raise RuntimeAttestationError("runtime lane attestation document schema mismatch")
        expected = payload.pop("lane_attestation_digest")
        attestation = cls.from_payload(payload)
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise RuntimeAttestationError("runtime lane attestation document digest is malformed")
        if attestation.lane_attestation_digest != expected:
            raise RuntimeAttestationError("runtime lane attestation document digest mismatch")
        return attestation

    @classmethod
    def from_json(cls, raw: str | bytes) -> "RuntimeLaneAttestation":
        """Load a lane document from JSON without allowing duplicate keys."""

        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            document = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except (TypeError, UnicodeDecodeError, ValueError) as exc:
            raise RuntimeAttestationError("runtime lane attestation JSON is malformed") from exc
        return cls.from_document(document)

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._canonical_payload.decode("utf-8"))

    @property
    def lane_attestation_digest(self) -> str:
        return hashlib.sha256(self._canonical_payload).hexdigest()

    @property
    def runtime_attestation(self) -> RuntimeAttestation:
        """Return the verified nested runtime evidence for a launch boundary."""

        return RuntimeAttestation.from_document(self.payload["runtime"])

    def document(self) -> dict[str, Any]:
        payload = self.payload
        payload["lane_attestation_digest"] = self.lane_attestation_digest
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
    distribution_root = source_root
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
        "files_root": str(distribution_root),
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
    verify_bytes: bool = True,
) -> None:
    """Validate an attestation's canonical schema before model execution."""

    if not isinstance(attestation, RuntimeAttestation):
        raise RuntimeAttestationError("a RuntimeAttestation produced by attest_runtime is required")
    payload = attestation.payload
    _validate_runtime_payload(
        payload,
        state_slice=state_slice,
        expected_python=expected_python,
        current_interpreter=_interpreter_identity(),
        verify_bytes=verify_bytes,
    )


def _validate_runtime_payload(
    payload: Mapping[str, Any],
    *,
    state_slice: str,
    expected_python: str,
    current_interpreter: Mapping[str, str] | None,
    verify_bytes: bool,
) -> None:
    """Validate runtime evidence, optionally binding it to this interpreter."""

    if not isinstance(verify_bytes, bool):
        raise RuntimeAttestationError("runtime byte verification mode is malformed")
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
    if current_interpreter is not None and interpreter != current_interpreter:
        raise RuntimeAttestationError("runtime attestation interpreter does not match this process")
    distributions = payload["distributions"]
    if not isinstance(distributions, dict) or not distributions:
        raise RuntimeAttestationError("runtime attestation distribution roster is empty")
    for name, entry in distributions.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise RuntimeAttestationError("runtime attestation distribution entry is malformed")
        if set(entry) != {
            "distribution", "version", "import_name", "files_root", "module_file",
            "module_sha256", "files",
        }:
            raise RuntimeAttestationError("runtime attestation distribution schema mismatch")
        if (
            entry["distribution"] != name
            or not isinstance(entry["version"], str)
            or not _VERSION.fullmatch(entry["version"])
        ):
            raise RuntimeAttestationError("runtime attestation distribution identity is malformed")
        if not isinstance(entry["import_name"], str) or not entry["import_name"].strip():
            raise RuntimeAttestationError("runtime attestation import identity is malformed")
        if not isinstance(entry["files_root"], str) or not Path(entry["files_root"]).is_absolute():
            raise RuntimeAttestationError("runtime attestation distribution root is malformed")
        files_root = Path(entry["files_root"]).expanduser()
        if files_root.is_symlink() or not files_root.is_dir():
            raise RuntimeAttestationError("runtime attestation distribution root is unavailable")
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
        if verify_bytes:
            _validate_distribution_bytes(name, entry)


def _validate_distribution_bytes(name: str, entry: Mapping[str, Any]) -> None:
    """Re-hash the current distribution and imported module against evidence."""

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeAttestationError(f"runtime distribution is not installed: {name}") from exc
    if distribution.version != entry["version"]:
        raise RuntimeAttestationError(f"runtime distribution version changed: {name}")
    files = distribution.files
    if not files:
        raise RuntimeAttestationError(f"runtime distribution has no file manifest: {name}")
    distribution_root = Path(entry["files_root"]).expanduser().resolve()
    actual_files = []
    for file_entry in sorted(files, key=str):
        relative, checksum = _regular_file_digest(
            Path(distribution.locate_file(file_entry)),
            root=distribution_root,
            label=f"{name}:{file_entry}",
        )
        actual_files.append({"path": relative, "sha256": checksum})
    expected_files = sorted(entry["files"], key=lambda item: item["path"])
    if actual_files != expected_files:
        raise RuntimeAttestationError(f"runtime distribution file bytes changed: {name}")
    try:
        module = importlib.import_module(entry["import_name"])
    except ImportError as exc:
        raise RuntimeAttestationError(f"runtime import is unavailable: {entry['import_name']}") from exc
    location = getattr(module, "__file__", None)
    if not isinstance(location, (str, Path)):
        raise RuntimeAttestationError(f"runtime import has no file: {entry['import_name']}")
    raw_module_path = Path(location).expanduser()
    if raw_module_path.is_symlink():
        raise RuntimeAttestationError(f"runtime import is a symlink: {entry['import_name']}")
    module_path = raw_module_path.resolve()
    if not module_path.is_file():
        raise RuntimeAttestationError(f"runtime import is not a regular file: {entry['import_name']}")
    if distribution_root != module_path and distribution_root not in module_path.parents:
        raise RuntimeAttestationError(f"runtime import escaped its distribution root: {entry['import_name']}")
    if str(module_path.relative_to(distribution_root)) != entry["module_file"]:
        raise RuntimeAttestationError(f"runtime import path changed: {entry['import_name']}")
    module_checksum = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if module_checksum != entry["module_sha256"]:
        raise RuntimeAttestationError(f"runtime import bytes changed: {entry['import_name']}")


def validate_runtime_lane_attestation(
    attestation: RuntimeLaneAttestation,
    *,
    state_slice: str,
    lane: str,
    expected_python: str,
) -> None:
    """Validate a lane wrapper without confusing the caller's Python for the target."""

    if not isinstance(attestation, RuntimeLaneAttestation):
        raise RuntimeAttestationError("a RuntimeLaneAttestation is required")
    payload = attestation.payload
    if set(payload) != {"schema", "state_slice", "lane", "runtime"}:
        raise RuntimeAttestationError("runtime lane attestation schema mismatch")
    if (
        payload["schema"] != "ahcl.runtime.lane-attestation.v1"
        or payload["state_slice"] != state_slice
        or payload["lane"] != lane
    ):
        raise RuntimeAttestationError("runtime lane attestation identity mismatch")
    runtime = payload["runtime"]
    if not isinstance(runtime, dict):
        raise RuntimeAttestationError("runtime lane attestation runtime document is malformed")
    try:
        nested = RuntimeAttestation.from_document(runtime)
    except RuntimeAttestationError as exc:
        raise RuntimeAttestationError("runtime lane attestation runtime document is malformed") from exc
    _validate_runtime_payload(
        nested.payload,
        state_slice=state_slice,
        expected_python=expected_python,
        current_interpreter=None,
        verify_bytes=False,
    )


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeAttestationError(f"{label} cannot be read as JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeAttestationError(f"{label} must be a JSON object")
    return value


def _lane_contract(
    *,
    lanes_path: str | Path,
    integrated_config_path: str | Path,
    lane: str,
    state_slice: str | None,
    import_names: Mapping[str, str] | None,
) -> tuple[str, str, tuple[tuple[str, str, str], ...]]:
    """Read one exact lane contract without installing or importing packages."""

    lanes = _read_json(lanes_path, label="dependency-lanes config")
    config = _read_json(integrated_config_path, label="integrated stack config")
    if lanes.get("schema") != "ahcl.dependency.lanes.v1":
        raise RuntimeAttestationError("dependency lanes schema mismatch")
    if config.get("schema") != "ahcl.integrated.stack.config.v1":
        raise RuntimeAttestationError("integrated stack config schema mismatch")
    lane_state = lanes.get("state_slice")
    config_state = config.get("state_slice")
    if not isinstance(lane_state, str) or lane_state != config_state:
        raise RuntimeAttestationError("lane and integrated config state slices differ")
    if state_slice is not None and state_slice != lane_state:
        raise RuntimeAttestationError("requested state slice does not match configs")
    if not isinstance(lane, str) or not lane.strip():
        raise RuntimeAttestationError("runtime lane identity is malformed")
    lanes_by_name = lanes.get("lanes")
    if not isinstance(lanes_by_name, dict) or lane not in lanes_by_name:
        raise RuntimeAttestationError(f"runtime lane is not declared: {lane}")
    lane_config = lanes_by_name[lane]
    constraints = lane_config.get("constraints") if isinstance(lane_config, dict) else None
    if not isinstance(constraints, dict):
        raise RuntimeAttestationError(f"runtime lane has no pinned constraints: {lane}")
    expected_python = constraints.get("python")
    runtime_config = config.get("runtime")
    configured_python = runtime_config.get("python") if isinstance(runtime_config, dict) else None
    if (
        not isinstance(expected_python, str)
        or not _VERSION.fullmatch(expected_python)
        or expected_python != configured_python
    ):
        raise RuntimeAttestationError(f"runtime lane Python pin is malformed or stale: {lane}")
    package_names = [name for name in constraints if name != "python"]
    if any(not isinstance(name, str) or not name.strip() for name in package_names):
        raise RuntimeAttestationError(f"runtime lane distribution identity is malformed: {lane}")
    packages = []
    for distribution in sorted(package_names):
        pin = constraints[distribution]
        if not isinstance(pin, str) or not pin.startswith("==") or not _VERSION.fullmatch(pin[2:]):
            raise RuntimeAttestationError(f"runtime lane package pin is malformed: {lane}:{distribution}")
        runtime_field = _RUNTIME_FIELDS.get(lane, {}).get(distribution)
        if runtime_field is not None and (
            not isinstance(runtime_config, dict) or runtime_config.get(runtime_field) != pin[2:]
        ):
            raise RuntimeAttestationError(f"runtime lane package pin is stale: {lane}:{distribution}")
        import_name = (import_names or {}).get(distribution, distribution.replace("-", "_"))
        if not isinstance(import_name, str) or not import_name.strip():
            raise RuntimeAttestationError(f"runtime lane import mapping is malformed: {lane}:{distribution}")
        packages.append((distribution, import_name, pin[2:]))
    if not packages:
        raise RuntimeAttestationError(f"runtime lane has no pinned distributions: {lane}")
    unknown_imports = set(import_names or {}) - set(package_names)
    if unknown_imports:
        raise RuntimeAttestationError(f"runtime lane import mapping is unknown: {lane}")
    return lane_state, expected_python, tuple(packages)


def _operator_interpreter(
    *,
    environment: str | Path | None,
    interpreter: str | Path | None,
) -> tuple[Path, Path | None]:
    if (environment is None) == (interpreter is None):
        raise RuntimeAttestationError("provide exactly one runtime environment or interpreter")
    if environment is not None:
        raw_environment = Path(environment).expanduser()
        if raw_environment.is_symlink() or not raw_environment.is_dir():
            raise RuntimeAttestationError("runtime environment must be a real directory")
        root = raw_environment.resolve()
        candidates = (root / "bin/python", root / "bin/python3", root / "Scripts/python.exe")
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is None:
            raise RuntimeAttestationError("runtime environment has no regular Python interpreter")
        resolved = selected.resolve()
        if not resolved.is_file():
            raise RuntimeAttestationError("runtime environment interpreter does not resolve to a file")
        return selected, root
    raw_interpreter = Path(interpreter).expanduser()
    if not raw_interpreter.is_file():
        raise RuntimeAttestationError("runtime interpreter must be a regular file")
    resolved = raw_interpreter.resolve()
    if not resolved.is_file():
        raise RuntimeAttestationError("runtime interpreter does not resolve to a file")
    return raw_interpreter, None


_CHILD_ATTESTATION = """
import json
import sys
from pathlib import Path
from ahcl.runtime_attestation import RuntimeRequirement, attest_runtime, validate_runtime_attestation

spec = json.load(sys.stdin)
source_root = Path(spec[\"source_root\"]) if spec[\"source_root\"] else Path(sys.prefix)
requirements = tuple(
    RuntimeRequirement(item[\"distribution\"], item[\"import_name\"], item[\"version\"], source_root)
    for item in spec[\"requirements\"]
)
attestation = attest_runtime(
    state_slice=spec[\"state_slice\"],
    expected_python=spec[\"expected_python\"],
    requirements=requirements,
)
validate_runtime_attestation(
    attestation,
    state_slice=spec[\"state_slice\"],
    expected_python=spec[\"expected_python\"],
)
print(json.dumps(attestation.document(), sort_keys=True, separators=(\",\", ":\")))
"""


def attest_lane(
    *,
    lane: str,
    lanes_path: str | Path,
    integrated_config_path: str | Path,
    environment: str | Path | None = None,
    interpreter: str | Path | None = None,
    state_slice: str | None = None,
    import_names: Mapping[str, str] | None = None,
) -> RuntimeLaneAttestation:
    """Attest one declared lane in an operator-selected Python runtime."""

    lane_state, expected_python, packages = _lane_contract(
        lanes_path=lanes_path,
        integrated_config_path=integrated_config_path,
        lane=lane,
        state_slice=state_slice,
        import_names=import_names,
    )
    target_interpreter, environment_root = _operator_interpreter(
        environment=environment,
        interpreter=interpreter,
    )
    spec = {
        "state_slice": lane_state,
        "expected_python": expected_python,
        "source_root": None if environment_root is None else str(environment_root),
        "requirements": [
            {"distribution": distribution, "import_name": import_name, "version": version}
            for distribution, import_name, version in packages
        ],
    }
    child_environment = dict(os.environ)
    source_path = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = source_path + ((":" + existing_pythonpath) if existing_pythonpath else "")
    child_environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [str(target_interpreter), "-c", _CHILD_ATTESTATION],
            input=canonical_bytes(spec),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_environment,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeAttestationError("operator runtime attestation process failed") from exc
    if completed.returncode != 0:
        raise RuntimeAttestationError("operator runtime failed its attestation contract")
    try:
        runtime = RuntimeAttestation.from_json(completed.stdout)
        if completed.stdout != canonical_bytes(runtime.document()) + b"\n":
            raise ValueError("runtime document is not canonical")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeAttestationError("operator runtime returned malformed attestation") from exc
    lane_attestation = RuntimeLaneAttestation.from_attestation(
        lane=lane,
        state_slice=lane_state,
        attestation=runtime,
    )
    validate_runtime_lane_attestation(
        lane_attestation,
        state_slice=lane_state,
        lane=lane,
        expected_python=expected_python,
    )
    return lane_attestation


__all__ = [
    "RuntimeAttestation",
    "RuntimeAttestationError",
    "RuntimeLaneAttestation",
    "RuntimeRequirement",
    "attest_lane",
    "attest_runtime",
    "validate_runtime_lane_attestation",
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
    """Emit one canonical runtime or runtime-lane attestation document as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-slice")
    parser.add_argument("--expected-python")
    parser.add_argument("--lanes-config", type=Path, default=Path("configs/dependency-lanes.json"))
    parser.add_argument("--integrated-config", type=Path, default=Path("configs/integrated-stack-v2.json"))
    parser.add_argument("--lane")
    runtime_selector = parser.add_mutually_exclusive_group()
    runtime_selector.add_argument("--environment", type=Path)
    runtime_selector.add_argument("--interpreter", type=Path)
    parser.add_argument("--import-name", action="append", default=[], metavar="DISTRIBUTION=IMPORT_NAME")
    parser.add_argument(
        "--requirement",
        action="append",
        metavar="DISTRIBUTION,IMPORT_NAME,VERSION,SOURCE_ROOT",
    )
    args = parser.parse_args(argv)
    try:
        if args.lane is not None:
            if args.requirement or args.state_slice is not None or args.expected_python is not None:
                parser.error("--lane cannot be combined with legacy runtime arguments")
            mappings = {}
            for raw_mapping in args.import_name:
                parts = raw_mapping.split("=", 1)
                if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                    raise RuntimeAttestationError("--import-name must be DISTRIBUTION=IMPORT_NAME")
                if parts[0] in mappings:
                    raise RuntimeAttestationError("duplicate --import-name distribution")
                mappings[parts[0]] = parts[1]
            attestation = attest_lane(
                lane=args.lane,
                lanes_path=args.lanes_config,
                integrated_config_path=args.integrated_config,
                environment=args.environment,
                interpreter=args.interpreter,
                import_names=mappings,
            )
        else:
            if (
                args.state_slice is None
                or args.expected_python is None
                or not args.requirement
                or args.environment is not None
                or args.interpreter is not None
                or args.import_name
            ):
                parser.error("legacy mode requires --state-slice, --expected-python, and --requirement")
            attestation = attest_runtime(
                state_slice=args.state_slice,
                expected_python=args.expected_python,
                requirements=tuple(_parse_requirement(raw) for raw in args.requirement),
            )
    except RuntimeAttestationError as error:
        parser.error(str(error))
    document = attestation.document()
    print(canonical_bytes(document).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
