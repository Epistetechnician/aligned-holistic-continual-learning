"""Review-packet and receipt checks; signing remains outside the runner.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .canonical import digest


class ReviewError(ValueError):
    """Raised when a review receipt cannot authorize execution."""


@dataclass(frozen=True)
class ReviewReceipt:
    packet_digest: str
    reviewer_id: str
    decision: str
    signature: str
    signature_scheme: str = "Ed25519"
    public_key_id: str = ""


def canonical_review_message(packet_digest: str) -> str:
    """Return the exact ASCII message an independent reviewer signs."""

    if (
        not isinstance(packet_digest, str)
        or len(packet_digest) != 64
        or any(character not in "0123456789abcdef" for character in packet_digest)
    ):
        raise ReviewError("packet digest is not canonical")
    return f"ahcl.review.packet.v1:{packet_digest}"


class Ed25519Verifier:
    """Verify hex-encoded Ed25519 receipts against an external public-key registry."""

    def __init__(self, public_keys: Mapping[str, bytes | str]):
        self._public_keys = dict(public_keys)

    @property
    def trusted_key_ids(self) -> frozenset[str]:
        return frozenset(self._public_keys)

    def __call__(self, receipt: ReviewReceipt, message: str) -> bool:
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError:
            return False
        try:
            key_material = self._public_keys[receipt.public_key_id]
            public_key = bytes.fromhex(key_material) if isinstance(key_material, str) else key_material
            signature = bytes.fromhex(receipt.signature)
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message.encode("ascii"))
            return True
        except (KeyError, ValueError, TypeError, InvalidSignature):
            return False


def packet_digest(packet: dict) -> str:
    return digest(packet)


def require_independent_accept(
    receipt: ReviewReceipt,
    *,
    expected_packet_digest: str,
    operator_id: str,
    verify_signature: Ed25519Verifier | None = None,
    trusted_public_key_ids: frozenset[str] | None = None,
) -> None:
    if receipt.decision != "ACCEPT":
        raise ReviewError("review decision is not ACCEPT")
    if receipt.packet_digest != expected_packet_digest:
        raise ReviewError("review receipt is bound to a different packet")
    if not receipt.reviewer_id or receipt.reviewer_id == operator_id:
        raise ReviewError("reviewer must be distinct from the operator")
    if not receipt.signature:
        raise ReviewError("review receipt has no signature")
    if receipt.signature_scheme != "Ed25519" or not receipt.public_key_id:
        raise ReviewError("review receipt must identify an Ed25519 public key")
    if not isinstance(verify_signature, Ed25519Verifier):
        raise ReviewError("the concrete Ed25519 verifier is required")
    if trusted_public_key_ids != verify_signature.trusted_key_ids:
        raise ReviewError("trusted reviewer registry does not match the verifier")
    if receipt.public_key_id not in verify_signature.trusted_key_ids:
        raise ReviewError("reviewer public key is not in the trusted registry")
    if not verify_signature(receipt, canonical_review_message(expected_packet_digest)):
        raise ReviewError("review signature verification failed")
