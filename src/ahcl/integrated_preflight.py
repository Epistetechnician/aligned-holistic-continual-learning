"""Read-only preflight for the integrated model-backed study.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
This command validates identities and custody only; it never loads a model or
changes the execution gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .custody import validate_manifest
from .preflight import (
    source_tree_digest,
    validate_corpus_manifest,
    validate_dependency_lanes,
    validate_input_bindings,
    validate_lane_sources,
    validate_review_packet,
    validate_upstream_lock,
)
from .stats import validate_locked_study


def run_preflight(repo_root: Path, custody_root: Path) -> dict[str, Any]:
    """Validate the exact integrated packet and return aggregate identities."""

    repo_root = repo_root.resolve()
    custody_root = custody_root.resolve()
    config = json.loads((repo_root / "configs/integrated-stack-v2.json").read_text(encoding="utf-8"))
    corpus_manifest = json.loads((repo_root / "configs/corpus-manifest.v2.json").read_text(encoding="utf-8"))
    packet = json.loads((repo_root / "configs/review-packet.v2.json").read_text(encoding="utf-8"))
    manifest = json.loads((custody_root / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest, expected_root=custody_root)
    validate_input_bindings(config, manifest)
    corpus_digest = validate_corpus_manifest(corpus_manifest, config=config)
    upstream_digest = validate_upstream_lock(repo_root / "third_party/UPSTREAMS.lock.json")
    validate_lane_sources(repo_root / "configs/dependency-lanes.json", repo_root / "third_party/UPSTREAMS.lock.json")
    lane_digest = validate_dependency_lanes(repo_root / "configs/dependency-lanes.json", config)
    validate_locked_study(config)
    validate_review_packet(packet)
    if packet["config"] != config:
        raise ValueError("review packet configuration is stale")
    if packet["upstream_lock_digest"] != upstream_digest:
        raise ValueError("review packet upstream digest is stale")
    if packet["dependency_lanes_digest"] != lane_digest:
        raise ValueError("review packet dependency-lane digest is stale")
    if packet["corpus_manifest_digest"] != corpus_digest:
        raise ValueError("review packet corpus digest is stale")
    actual_source_digest = source_tree_digest(repo_root)
    if actual_source_digest != packet["source_digest"]:
        raise ValueError("review packet source digest is stale")
    return {
        "schema": "ahcl.integrated.preflight.v1",
        "state_slice": config["state_slice"],
        "status": "READY_FOR_INDEPENDENT_REVIEW",
        "execution_enabled": bool(config["execution"]["enabled"]),
        "custody_manifest_digest": manifest["manifest_digest"],
        "upstream_lock_digest": upstream_digest,
        "dependency_lanes_digest": lane_digest,
        "corpus_manifest_digest": corpus_digest,
        "source_digest": actual_source_digest,
        "packet_digest": packet["packet_digest"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--custody-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.repo_root, args.custody_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
