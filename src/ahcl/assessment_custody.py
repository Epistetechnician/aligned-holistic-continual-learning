"""Durable local one-use guard for an assessment attempt.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.

This module records an attempt reservation only.  It does not validate a
review, authorize a provider, or establish scientific evidence.  The owner of
the custody root can delete a marker, so this guard is not adversarial custody
or external authorization.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path


STATE_SLICE = "aligned-holistic-continual-learning-integrated-stack-v2"
_IDENTITY = re.compile(r"\A[0-9a-f]{64}\Z")


class AssessmentAttemptError(ValueError):
    """Raised when an assessment attempt cannot be claimed safely."""


@dataclass(frozen=True)
class AssessmentAttemptClaim:
    """The durable marker created for one packet and predictor pair."""

    path: Path
    packet_id: str
    predictor_id: str


def _validate_identity(name: str, value: str) -> str:
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise AssessmentAttemptError(f"{name} must be a canonical lowercase SHA-256 ID")
    return value


def _validate_root(root: Path) -> Path:
    if not isinstance(root, Path):
        root = Path(root)
    return root


def claim_assessment_attempt(
    root: Path,
    *,
    packet_id: str,
    predictor_id: str,
) -> AssessmentAttemptClaim:
    """Atomically consume the assessment attempt for a packet/predictor pair.

    The marker is intentionally created before any work can begin.  A marker
    left by a failed or interrupted process therefore still prevents retry.
    """

    packet_id = _validate_identity("packet_id", packet_id)
    predictor_id = _validate_identity("predictor_id", predictor_id)
    custody_root = _validate_root(root)
    marker_name = f"assessment-attempt.{packet_id}.{predictor_id}.json"
    marker = custody_root / marker_name

    payload = {
        "schema": "ahcl.assessment-attempt-claim.v1",
        "state_slice": STATE_SLICE,
        "packet_id": packet_id,
        "predictor_id": predictor_id,
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory = os.open(custody_root, directory_flags)
    except OSError as exc:
        raise AssessmentAttemptError("assessment custody root must be a real directory") from exc
    try:
        directory_stat = os.fstat(directory)
        if directory_stat.st_uid != os.getuid():
            raise AssessmentAttemptError("assessment custody root must be owned by the current user")
        if stat.S_IMODE(directory_stat.st_mode) != 0o700:
            raise AssessmentAttemptError("assessment custody root must have owner-only 0700 permissions")
        try:
            existing = os.lstat(marker_name, dir_fd=directory)
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise AssessmentAttemptError("assessment attempt marker must not be a symlink")
        try:
            descriptor = os.open(
                marker_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
        except FileExistsError as exc:
            try:
                if stat.S_ISLNK(os.lstat(marker_name, dir_fd=directory).st_mode):
                    raise AssessmentAttemptError("assessment attempt marker must not be a symlink") from exc
            except FileNotFoundError:
                raise AssessmentAttemptError("assessment attempt could not be claimed") from exc
            raise AssessmentAttemptError("assessment attempt already consumed") from exc
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise AssessmentAttemptError("assessment attempt marker must not be a symlink") from exc
            raise AssessmentAttemptError("assessment attempt marker could not be created") from exc
        try:
            os.fsync(directory)
        except OSError as exc:
            os.close(descriptor)
            raise AssessmentAttemptError("assessment attempt marker directory sync failed; attempt is consumed") from exc
    finally:
        os.close(directory)

    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        # The exclusive marker remains by design: failed attempts cannot retry.
        raise AssessmentAttemptError("assessment attempt marker write failed; attempt is consumed") from exc
    return AssessmentAttemptClaim(marker, packet_id, predictor_id)


__all__ = ["AssessmentAttemptClaim", "AssessmentAttemptError", "claim_assessment_attempt"]
