"""Fail-closed execution gate for model-backed work.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Callable

from .review import ReviewReceipt, require_independent_accept


class ExecutionClosed(RuntimeError):
    """Raised when model-bearing execution lacks an accepted packet."""


def require_execution_open(
    config: dict[str, Any],
    receipt: ReviewReceipt | None,
    *,
    packet_digest: str,
    operator_id: str,
    verify_signature: Callable[[ReviewReceipt, str], bool] | None = None,
) -> None:
    if not config.get("execution", {}).get("enabled", False):
        raise ExecutionClosed("model-bearing execution is closed in this state slice")
    if receipt is None:
        raise ExecutionClosed("independent review receipt is required")
    require_independent_accept(
        receipt,
        expected_packet_digest=packet_digest,
        operator_id=operator_id,
        verify_signature=verify_signature,
    )
