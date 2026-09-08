"""Fail-closed execution gate for model-backed work.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Mapping

from .canonical import digest
from .runtime_attestation import (
    RuntimeAttestation,
    RuntimeAttestationError,
    RuntimeLaneAttestation,
    validate_runtime_attestation,
    validate_runtime_lane_attestation,
)
from .verified_review import TrustedReviewer, verify_launch_receipt


class ExecutionClosed(RuntimeError):
    """Raised when model-bearing execution lacks an accepted packet."""


def require_execution_open(
    config: Mapping[str, Any],
    receipt: Mapping[str, Any] | None,
    *,
    packet_digest: str,
    reviewed_packet: Mapping[str, Any] | None = None,
    operator_id: str,
    runtime_attestation: RuntimeAttestation | RuntimeLaneAttestation | None = None,
    runtime_lane: str | None = None,
    registry: Mapping[str, TrustedReviewer] | None = None,
    review_purpose: str = "qualification",
) -> None:
    if not config.get("execution", {}).get("enabled", False):
        raise ExecutionClosed("model-bearing execution is closed in this state slice")
    if receipt is None:
        raise ExecutionClosed("independent review receipt is required")
    if reviewed_packet is None or digest(
        {key: value for key, value in reviewed_packet.items() if key != "packet_digest"}
    ) != packet_digest:
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
    try:
        verify_launch_receipt(
            reviewed_packet,
            config,
            receipt,
            registry,
            operator_id=operator_id,
            purpose=review_purpose,
        )
    except (ImportError, KeyError, TypeError, ValueError) as error:
        raise ExecutionClosed("independent review receipt failed canonical verification") from error
