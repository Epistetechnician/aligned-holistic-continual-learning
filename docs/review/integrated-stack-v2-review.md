# Integrated stack v2 review record

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

Status: `PENDING_INDEPENDENT_REVIEW`

Readiness packet: `configs/review-packet.v2.json`

Readiness packet digest: `3c927dc32dcd025045356e0e4b68e4beeaf6c4a572bf5e03d3458496fa2057a7`

Unsigned launch packet digest: `1b6d96428c492a345257123f1ac6b0119ea55974fe6eabc7f24fe0aa78767f2d`

The readiness packet is immutable and keeps `execution.enabled: false`. A
separate launch packet must be derived from it with the same public inputs,
`packet_role: "launch"`, and `execution.enabled: true`. Its digest is the one
the reviewer signs; the readiness packet is never edited into a launch packet.

The packet to review is the exact combination of:

- `third_party/UPSTREAMS.lock.json`
- `configs/integrated-stack-v2.json`
- `configs/dependency-lanes.json`
- `configs/corpus-manifest.v2.json`
- `docs/integrated-experiment-v2.md`
- `configs/review-packet.v2.json`, which binds the source, upstream, corpus,
  configuration, and custody identities as the readiness artifact

The reviewer must check source commit identity, license compatibility,
dependency-lane separation, exact runtime versions, model and asset hashes,
corpus author-disjoint splits, retention and custody permissions, evaluator
definitions, intervention controls, power and multiplicity assumptions, lock
ordering, and the claim ceiling. The reviewer must verify the exact final
launch packet and all runtime/source attestations, then return a signed
Ed25519 receipt whose `purpose` is `qualification`. The signature covers the
canonical UTF-8 JSON bytes produced by
`verified_review.signed_message(packet_digest, reviewer_id, public_key_id,
"qualification")` under domain `ahcl.external-review.v2`; the legacy ASCII
`ahcl.review.packet.v1:<packet_digest>` message is invalid. The reviewer key
ID must be provisioned in a registry outside operator control, and the
reviewer ID must differ from the operator ID. The execution gate uses
`verified_review.verify_launch_receipt` as its sole verifier.

The private givemeanode artifact `art-rkjx9` is transport custody only. It
cannot provision reviewer authority, create a trusted registry entry, or
serve as an independent signature.

The operator is not an independent reviewer. A local test run, a green CI
run, or a self-created signature cannot change this record to `ACCEPT`.

Execution authorization remains closed while this record is pending.
