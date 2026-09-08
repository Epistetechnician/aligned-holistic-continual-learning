"""Concrete verification for externally issued launch receipts.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
The trusted reviewer registry must be provisioned independently of the operator.
This module does not create keys, sign receipts, or enable model execution.
"""
from dataclasses import dataclass
from .canonical import canonical_bytes, digest
from .preflight import validate_review_packet


@dataclass(frozen=True)
class TrustedReviewer:
    reviewer_id: str
    public_key: bytes


def signed_message(packet_id, reviewer_id, key_id, purpose):
    return canonical_bytes({'domain': 'ahcl.external-review.v2', 'packet_digest': packet_id,
                            'reviewer_id': reviewer_id, 'public_key_id': key_id,
                            'decision': 'ACCEPT', 'purpose': purpose})


def verify_launch_receipt(packet, config, receipt, registry, *, operator_id, purpose):
    """Check exact packet and config plus signature and reviewer-key binding.

    Returns a digest only, not an authorization object for legacy adapters.
    Registry entries are trusted inputs, not trust inferred from a receipt.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    validate_review_packet(packet)
    if packet['config'] != config:
        raise ValueError('launch config differs from reviewed config')
    if purpose not in ('qualification', 'assessment'):
        raise ValueError('invalid review purpose')
    required = {'packet_digest', 'reviewer_id', 'public_key_id', 'decision', 'purpose', 'signature'}
    if set(receipt) != required or receipt['decision'] != 'ACCEPT' or receipt['purpose'] != purpose:
        raise ValueError('invalid receipt schema or purpose')
    if receipt['packet_digest'] != packet['packet_digest']:
        raise ValueError('receipt packet mismatch')
    reviewer = registry.get(receipt['public_key_id'])
    if (type(reviewer) is not TrustedReviewer or not reviewer.reviewer_id
            or reviewer.reviewer_id != receipt['reviewer_id'] or reviewer.reviewer_id == operator_id):
        raise ValueError('reviewer key is not independently trusted')
    try:
        Ed25519PublicKey.from_public_bytes(reviewer.public_key).verify(
            bytes.fromhex(receipt['signature']),
            signed_message(packet['packet_digest'], reviewer.reviewer_id, receipt['public_key_id'], purpose))
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ValueError('invalid review signature') from exc
    return digest(receipt)
