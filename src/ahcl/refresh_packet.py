"""Refresh the exact integrated review packet after authorized source edits.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
This command validates the public identities before replacing only the packet;
it never loads a model, reads model bytes, calls a provider, or opens execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .preflight import (
    build_review_packet,
    source_tree_digest,
    validate_corpus_manifest,
    validate_dependency_lanes,
    validate_lane_sources,
    validate_review_packet,
    validate_upstream_lock,
)
from .stats import validate_locked_study


def build_current_packet(repo_root: Path) -> dict[str, Any]:
    """Build and validate a packet from the current checkout's public bytes."""

    repo_root = repo_root.resolve()
    config = json.loads((repo_root / "configs/integrated-stack-v2.json").read_text(encoding="utf-8"))
    upstream_lock = json.loads((repo_root / "third_party/UPSTREAMS.lock.json").read_text(encoding="utf-8"))
    dependency_lanes = json.loads((repo_root / "configs/dependency-lanes.json").read_text(encoding="utf-8"))
    corpus_manifest = json.loads((repo_root / "configs/corpus-manifest.v2.json").read_text(encoding="utf-8"))
    validate_corpus_manifest(corpus_manifest, config=config)
    validate_upstream_lock(repo_root / "third_party/UPSTREAMS.lock.json")
    validate_lane_sources(
        repo_root / "configs/dependency-lanes.json",
        repo_root / "third_party/UPSTREAMS.lock.json",
    )
    validate_dependency_lanes(repo_root / "configs/dependency-lanes.json", config)
    validate_locked_study(config)
    packet = build_review_packet(
        config=config,
        upstream_lock=upstream_lock,
        dependency_lanes=dependency_lanes,
        corpus_manifest=corpus_manifest,
        source_digest=source_tree_digest(repo_root),
    )
    validate_review_packet(packet)
    return packet


def refresh_packet(repo_root: Path) -> dict[str, Any]:
    """Validate then replace the tracked review packet and return its identity."""

    repo_root = repo_root.resolve()
    packet = build_current_packet(repo_root)
    packet_path = repo_root / "configs/review-packet.v2.json"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packet


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    packet = refresh_packet(args.repo_root)
    print(json.dumps({"packet_digest": packet["packet_digest"], "source_digest": packet["source_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
