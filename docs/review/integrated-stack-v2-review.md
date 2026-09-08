# Integrated stack v2 review record

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

Status: `PENDING_INDEPENDENT_REVIEW`

Current packet: `configs/review-packet.v2.json`

Packet digest: `d627749e9f22ba45f0be82c7baa8110459e3c1dcc2edf1b31ca3e58f0c510c0d`

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
receipt bound to the packet digest. The execution gate also requires an
independent verifier callback for that signature; a non-empty string is not
accepted as proof.

The operator is not an independent reviewer. A local test run, a green CI
run, or a self-created signature cannot change this record to `ACCEPT`.

Execution authorization remains closed while this record is pending.
