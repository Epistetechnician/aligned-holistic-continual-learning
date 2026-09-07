# Integrated stack v2 review record

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

Status: `PENDING_INDEPENDENT_REVIEW`

Current packet: `configs/review-packet.v2.json`

Packet digest: `c388216333960f3a48e37d9f4e590964a0d3638ada9d8bed49e97dd5dddaa0ff`

The packet to review is the exact combination of:

- `third_party/UPSTREAMS.lock.json`
- `configs/integrated-stack-v2.json`
- `configs/dependency-lanes.json`
- `configs/corpus-manifest.v2.json`
- `docs/integrated-experiment-v2.md`
- `configs/review-packet.v2.json`, which binds the source, upstream, corpus,
  configuration, and custody identities

The reviewer must check source commit identity, license compatibility,
dependency-lane separation, model and asset hashes, corpus author-disjoint
splits, retention and custody permissions, evaluator definitions, intervention
controls, power and multiplicity assumptions, lock ordering, and the claim
ceiling. The reviewer must return a signed receipt bound to the packet digest.

The operator is not an independent reviewer. A local test run, a green CI
run, or a self-created signature cannot change this record to `ACCEPT`.

Execution authorization remains closed while this record is pending.
