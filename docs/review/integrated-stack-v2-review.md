# Integrated stack v2 review record

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

Status: `PENDING_INDEPENDENT_REVIEW`

Current packet: `configs/review-packet.v2.json`

Packet digest: `c5d0c678d97fe306d55070b22a301ca37f3e5b8a22d038db452edf1637c5cdb8`

The packet to review is the exact combination of:

- `third_party/UPSTREAMS.lock.json`
- `configs/integrated-stack-v2.json`
- `configs/dependency-lanes.json`
- `configs/corpus-manifest.v2.json`
- `docs/integrated-experiment-v2.md`
- `configs/review-packet.v2.json`, which binds the source, upstream, corpus,
  configuration, and custody identities

The reviewer must check source commit identity, license compatibility,
dependency-lane separation, exact runtime versions, model and asset hashes,
corpus author-disjoint splits, retention and custody permissions, evaluator
definitions, intervention controls, power and multiplicity assumptions, lock
ordering, and the claim ceiling. The reviewer must return a signed Ed25519
receipt bound to the canonical message
`ahcl.review.packet.v1:<packet_digest>`. The execution gate requires the
reviewer key ID to be present in an external trusted registry and requires an
independent Ed25519 verifier; a non-empty string or arbitrary callback is not
accepted as proof.

The operator is not an independent reviewer. A local test run, a green CI
run, or a self-created signature cannot change this record to `ACCEPT`.

Execution authorization remains closed while this record is pending.
