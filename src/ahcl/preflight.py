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
                if path.name == "review-packet.v2.json" or relative.parts[:2] == ("docs", "review"):
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


def build_review_packet(
    *,
    config: dict[str, Any],
    upstream_lock: dict[str, Any],
    corpus_manifest: dict[str, Any],
    source_digest: str,
) -> dict[str, Any]:
    """Bind every identity needed before a reviewer may accept execution."""

    packet = {
        "schema": "ahcl.review.packet.v1",
        "state_slice": config["state_slice"],
        "config": config,
        "upstream_lock_digest": digest(upstream_lock),
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
        "corpus_manifest_digest",
        "source_digest",
        "packet_digest",
    }
    if set(packet) != required or packet["schema"] != "ahcl.review.packet.v1":
        raise PreflightError("review packet schema mismatch")
    expected = digest({key: value for key, value in packet.items() if key != "packet_digest"})
    if expected != packet["packet_digest"]:
        raise PreflightError("review packet digest mismatch")
