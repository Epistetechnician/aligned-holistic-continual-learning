"""Preflight checks for exact upstream and external input identities.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .canonical import digest, file_digest


class PreflightError(ValueError):
    """Raised when a pinned input identity is not reproducible."""


def validate_upstream_lock(lock_path: Path) -> str:
    root = lock_path.resolve().parent.parent
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "ahcl.upstreams.lock.v1":
        raise PreflightError("wrong upstream lock schema")
    for item in lock["submodules"]:
        submodule = root / item["path"]
        actual = subprocess.check_output(["git", "-C", str(submodule), "rev-parse", "HEAD"], text=True).strip()
        if actual != item["commit"]:
            raise PreflightError(f"submodule drift: {item['name']}")
        for license_file in item["license_files"]:
            if not (submodule / license_file).is_file():
                raise PreflightError(f"missing license file: {item['name']}:{license_file}")
    return digest(lock)


def validate_dependency_lanes(lanes_path: Path, config: dict[str, Any]) -> str:
    """Validate that the selected runtime versions match every lane lock."""

    lanes = json.loads(lanes_path.read_text(encoding="utf-8"))
    if lanes.get("schema") != "ahcl.dependency.lanes.v1" or lanes.get("single_environment") is not False:
        raise PreflightError("dependency lanes must remain explicitly isolated")
    runtime = config.get("runtime", {})
    modern = lanes.get("lanes", {}).get("modern", {}).get("constraints", {})
    if modern != {
        "python": "3.11.11",
        "transformers": f"=={runtime.get('transformers')}",
        "torch": f"=={runtime.get('torch')}",
    }:
        raise PreflightError("modern lane does not match the configured runtime")
    circuit = lanes.get("lanes", {}).get("circuit_tracer", {}).get("constraints", {})
    if circuit != {
        "python": "3.11.11",
        "transformers": f"=={runtime.get('circuit_tracer_transformers')}",
        "nnsight": "==0.6.1",
        "transformer-lens": f"=={runtime.get('circuit_tracer_transformer_lens')}",
        "torch": f"=={runtime.get('circuit_tracer_torch')}",
    }:
        raise PreflightError("circuit-tracer lane does not match the configured runtime")
    acdc = lanes.get("lanes", {}).get("acdc", {}).get("constraints", {})
    if acdc != {
        "python": "3.11.11",
        "transformers": f"=={runtime.get('acdc_transformers')}",
        "transformer-lens": f"=={runtime.get('acdc_transformer_lens')}",
        "torch": f"=={runtime.get('acdc_torch')}",
    }:
        raise PreflightError("ACDC lane does not match the configured runtime")
    return digest(lanes)


def source_tree_digest(root: Path) -> str:
    """Hash the authored source, tests, configs, and protocol records."""

    authored = []
    for directory in ("src", "tests", "configs", "docs"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                # The packet records this digest, so including the packet would
                # create a circular identity. Review records carry the packet
                # digest too, so they are validated separately below.
                relative = path.relative_to(root)
                if (
                    path.name == "review-packet.v2.json"
                    or relative.parts[:2] == ("docs", "review")
                    or "__pycache__" in relative.parts
                    or path.suffix in {".pyc", ".pyo"}
                ):
                    continue
                authored.append({"path": str(relative), "sha256": file_digest(path)})
    return digest(authored)


def validate_input_bindings(config: dict[str, Any], custody_manifest: dict[str, Any]) -> None:
    """Check the configured model, feature, corpus, and custody identities."""

    if config.get("state_slice") != custody_manifest.get("state_slice"):
        raise PreflightError("config and custody state slices differ")
    if config.get("custody", {}).get("manifest_digest") != custody_manifest.get("manifest_digest"):
        raise PreflightError("config is bound to a different custody manifest")
    files = {entry["path"]: entry["sha256"] for entry in custody_manifest.get("files", [])}
    if files.get(config["model"]["custody_file"]) != config["model"]["sha256"]:
        raise PreflightError("model custody hash mismatch")
    if files.get(config["feature_asset"]["custody_file"]) != config["feature_asset"]["sha256"]:
        raise PreflightError("feature asset custody hash mismatch")
    identities = custody_manifest.get("input_identities", {})
    model_identity = identities.get("model", {})
    if (model_identity.get("id"), model_identity.get("revision")) != (
        config["model"]["id"],
        config["model"]["revision"],
    ):
        raise PreflightError("custody model identity mismatch")
    feature_identity = identities.get("feature_asset", {})
    if (feature_identity.get("repository"), feature_identity.get("revision"), feature_identity.get("path")) != (
        config["feature_asset"]["repository"],
        config["feature_asset"]["revision"],
        config["feature_asset"]["sae_id"],
    ):
        raise PreflightError("custody feature identity mismatch")
    corpus_identity = identities.get("corpus", {})
    if corpus_identity.get("documents") != 18 or corpus_identity.get("splits") != {
        "fit": 6,
        "tune": 6,
        "assessment": 6,
    }:
        raise PreflightError("custody corpus roster mismatch")


def validate_corpus_manifest(corpus_manifest: dict[str, Any], *, config: dict[str, Any] | None = None) -> str:
    """Validate the public corpus roster and return its canonical digest."""

    required = {"schema", "state_slice", "documents", "selection_rule", "source", "tokenizer", "manifest_digest"}
    if not required.issubset(corpus_manifest) or corpus_manifest["schema"] != "ahcl.corpus.manifest.v1":
        raise PreflightError("corpus manifest schema mismatch")
    if config is not None and corpus_manifest["state_slice"] != config.get("state_slice"):
        raise PreflightError("corpus manifest state slice mismatch")
    if digest({key: value for key, value in corpus_manifest.items() if key != "manifest_digest"}) != corpus_manifest["manifest_digest"]:
        raise PreflightError("corpus manifest digest mismatch")
    documents = corpus_manifest["documents"]
    if not isinstance(documents, list) or len(documents) != 18:
        raise PreflightError("corpus manifest must contain exactly 18 documents")
    if len({document.get("author") for document in documents}) != 18:
        raise PreflightError("corpus authors must be disjoint")
    split_counts = {split: sum(document.get("split") == split for document in documents) for split in ("fit", "tune", "assessment")}
    if split_counts != {"fit": 6, "tune": 6, "assessment": 6}:
        raise PreflightError("corpus split roster mismatch")
    if config is not None:
        tokenizer = corpus_manifest["tokenizer"]
        if (tokenizer.get("name"), tokenizer.get("revision")) != (
            config["model"]["id"],
            config["model"]["revision"],
        ):
            raise PreflightError("corpus tokenizer identity mismatch")
    return corpus_manifest["manifest_digest"]


def build_review_packet(
    *,
    config: dict[str, Any],
    upstream_lock: dict[str, Any],
    corpus_manifest: dict[str, Any],
    source_digest: str,
    dependency_lanes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind every identity needed before a reviewer may accept execution."""

    packet = {
        "schema": "ahcl.review.packet.v1",
        "state_slice": config["state_slice"],
        "config": config,
        "upstream_lock_digest": digest(upstream_lock),
        "dependency_lanes_digest": None if dependency_lanes is None else digest(dependency_lanes),
        "corpus_manifest_digest": corpus_manifest.get("manifest_digest"),
        "source_digest": source_digest,
    }
    packet["packet_digest"] = digest(packet)
    return packet


def validate_review_packet(packet: dict[str, Any]) -> None:
    required = {
        "schema",
        "state_slice",
        "config",
        "upstream_lock_digest",
        "dependency_lanes_digest",
        "corpus_manifest_digest",
        "source_digest",
        "packet_digest",
    }
    if set(packet) != required or packet["schema"] != "ahcl.review.packet.v1":
        raise PreflightError("review packet schema mismatch")
    if (
        not isinstance(packet["config"], dict)
        or not packet["state_slice"]
        or packet["config"].get("state_slice") != packet["state_slice"]
    ):
        raise PreflightError("review packet state slice mismatch")
    for field in ("upstream_lock_digest", "dependency_lanes_digest", "corpus_manifest_digest", "source_digest"):
        value = packet[field]
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise PreflightError(f"review packet {field} is not a canonical digest")
    expected = digest({key: value for key, value in packet.items() if key != "packet_digest"})
    if expected != packet["packet_digest"]:
        raise PreflightError("review packet digest mismatch")
