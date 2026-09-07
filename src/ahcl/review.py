"""Review-packet and receipt checks; signing remains outside the runner.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import digest


class ReviewError(ValueError):
    """Raised when a review receipt cannot authorize execution."""


@dataclass(frozen=True)
class ReviewReceipt:
    packet_digest: str
    reviewer_id: str
    decision: str
    signature: str


def packet_digest(packet: dict) -> str:
    return digest(packet)


def require_independent_accept(
    receipt: ReviewReceipt,
    *,
    expected_packet_digest: str,
    operator_id: str,
) -> None:
    if receipt.decision != "ACCEPT":
        raise ReviewError("review decision is not ACCEPT")
    if receipt.packet_digest != expected_packet_digest:
        raise ReviewError("review receipt is bound to a different packet")
    if not receipt.reviewer_id or receipt.reviewer_id == operator_id:
        raise ReviewError("reviewer must be distinct from the operator")
    if not receipt.signature:
        raise ReviewError("review receipt has no signature")
