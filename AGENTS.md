# Integrated upstream research rules

All mutations in this repository belong to the named state slice
`aligned-holistic-continual-learning-integrated-stack-v2`.

The clean-room synthetic baseline is immutable at commits `2109889` and
`6057c8f`. This slice explicitly authorizes immutable upstream source
submodules under `third_party/`, with exact commit pins and license records in
`third_party/UPSTREAMS.lock.json`. Do not edit vendored submodule contents in
this repository.

Upstream source may be called only through adapters under
`src/ahcl/adapters/`. Adapter imports are lazy, credentials are never stored,
and each adapter must expose parity, repeatability, no-op, and intervention
reach checks before it can be enabled.

Model weights, feature assets, and fresh corpus bytes must live outside Git
under an owner-only custody root with a manifest and retention policy. The
current slice authorizes configuration and preflight code, but model-bearing
execution, provider calls, and assessment remain closed until an independent
protocol/execution review accepts the exact packet. The runner cannot
self-sign that review.

Keep fit, tune, and assessment identities disjoint. Lock the predictor and all
thresholds before assessment. Publish aggregate metrics only. A result may not
claim mechanistic understanding, alignment, continual-learning superiority, or
production readiness beyond its recorded evidence ceiling.

Measure twice, cut once. Keep the codebase clean: no temporary files, dead
code, dead files, or untracked generated artifacts.

## Public release readiness

Explicit public-release readiness is authorized under named state slice
`public-release-readiness-v1`: additive `LICENSE`, `NOTICE`, `SECURITY.md`,
`CONTRIBUTING.md`, `RESEARCH.md`, README navigation, replacement of one
machine-specific path with a placeholder in documentation, and a CI workflow
that runs the existing hermetic tests. It changes no source behavior,
thresholds, protocols, results, or claim ceilings and authorizes no model
execution or claim escalation. Every mutation in this phase names state slice
`public-release-readiness-v1`.
