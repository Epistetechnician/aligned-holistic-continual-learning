"""Fail-closed execution gate for model-backed work.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Mapping

from .canonical import digest
from .review import Ed25519Verifier, ReviewReceipt, require_independent_accept
from .runtime_attestation import (
    RuntimeAttestation,
    RuntimeAttestationError,
    RuntimeLaneAttestation,
    validate_runtime_attestation,
    validate_runtime_lane_attestation,
)


class ExecutionClosed(RuntimeError):
    """Raised when model-bearing execution lacks an accepted packet."""


def require_execution_open(
    config: dict[str, Any],
    receipt: ReviewReceipt | None,
    *,
    packet_digest: str,
    reviewed_packet: Mapping[str, Any] | None = None,
    operator_id: str,
    runtime_attestation: RuntimeAttestation | RuntimeLaneAttestation | None = None,
    runtime_lane: str | None = None,
    verify_signature: Ed25519Verifier | None = None,
    trusted_public_key_ids: frozenset[str] | None = None,
) -> None:
    if not config.get("execution", {}).get("enabled", False):
        raise ExecutionClosed("model-bearing execution is closed in this state slice")
    if receipt is None:
        raise ExecutionClosed("independent review receipt is required")
    if reviewed_packet is None or digest(dict(reviewed_packet)) != packet_digest:
        raise ExecutionClosed("the exact reviewed packet is required at launch")
    if reviewed_packet.get("config") != config:
        raise ExecutionClosed("launch config does not match the reviewed packet")
    try:
        if isinstance(runtime_attestation, RuntimeLaneAttestation):
            if not isinstance(runtime_lane, str) or not runtime_lane.strip():
                raise RuntimeAttestationError("the selected runtime lane is required at launch")
            validate_runtime_lane_attestation(
                runtime_attestation,
                state_slice=config.get("state_slice", ""),
                lane=runtime_lane,
                expected_python=config.get("runtime", {}).get("python", ""),
            )
            runtime_attestation = runtime_attestation.runtime_attestation
        elif runtime_lane is not None:
            raise RuntimeAttestationError("a runtime lane attestation is required for the selected lane")
        validate_runtime_attestation(
            runtime_attestation,
            state_slice=config.get("state_slice", ""),
            expected_python=config.get("runtime", {}).get("python", ""),
        )
    except RuntimeAttestationError as error:
        raise ExecutionClosed("verified runtime distribution attestation is required at launch") from error
    require_independent_accept(
        receipt,
        expected_packet_digest=packet_digest,
        operator_id=operator_id,
        verify_signature=verify_signature,
        trusted_public_key_ids=trusted_public_key_ids,
    )
