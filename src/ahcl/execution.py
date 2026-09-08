"""Fail-closed execution gate for model-backed work.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .canonical import digest
from .review import ReviewReceipt, require_independent_accept


class ExecutionClosed(RuntimeError):
    """Raised when model-bearing execution lacks an accepted packet."""


def require_execution_open(
    config: dict[str, Any],
    receipt: ReviewReceipt | None,
    *,
    packet_digest: str,
    reviewed_packet: Mapping[str, Any] | None = None,
    operator_id: str,
    verify_signature: Callable[[ReviewReceipt, str], bool] | None = None,
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
    require_independent_accept(
        receipt,
        expected_packet_digest=packet_digest,
        operator_id=operator_id,
        verify_signature=verify_signature,
        trusted_public_key_ids=trusted_public_key_ids,
    )
