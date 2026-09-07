"""Canonical serialization used for reproducible experiment identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data with one deterministic byte layout."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON bytes."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file without normalizing its bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()
