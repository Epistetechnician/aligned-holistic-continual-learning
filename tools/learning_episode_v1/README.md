# Synthetic Learning Episode Receipt V1

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`

Governance slice: `global-experiment-assurance-lifecycle-v1`

Protocol: `ahcl.synthetic-learning-episode.v1`

## Purpose and boundary

This is a model-free end-to-end integration contract for one deterministic
synthetic learning episode. It connects a model-state-conditioned selector,
bounded update runner, known-coordinate mechanism adapter, tune-time prediction
lock, and one-use held-out evaluator. It demonstrates digest and lifecycle
binding across those stages; it does not test real continual learning.

The mechanism adapter operates on a three-weight linear toy simulator whose
features, outcome function, and update rule are all known. Its no-op and
feature-ablation checks validate the adapter wiring in that simulator only.
They are not mechanistic interpretability evidence about a learned neural
network.

The fixed episode uses 24 synthetic fit candidates, 16 disjoint tune examples,
32 disjoint assessment examples, eight maximum updates, five predeclared tune
checkpoints, and a fixed-ID replay control. The selector can inspect labels
only in the fit candidate pool. Tune chooses a checkpoint separately for the
selector and control; both weights and tune summaries are locked before the
assessment rows are generated. Assessment emits aggregates only.

## Run and verify

Write the receipt outside the repository so generated outputs are not added to
Git:

```sh
PYTHONPATH=src:tools/learning_episode_v1 python3 -m learning_episode \
  --output /tmp/ahcl-learning-episode-v1.json
PYTHONPATH=src:tools/learning_episode_v1 python3 -m validate_learning_episode \
  /tmp/ahcl-learning-episode-v1.json
```

The verifier does not import the producer. It regenerates the cohorts and
reconstructs selection scores, updates, mechanism checks, tune locks, and
held-out metrics. This is local contract verification, not an independent
reviewer decision. The receipt records the repository HEAD plus exact runner
and validator source digests and the Python runtime. It reports
`INTERNAL_PASS`; repeatability was checked only by the same owner in one local
environment, not by a clean second environment.

The assessment gate is an in-process ordering check: assessment rows are
generated only after both locks validate, and the evaluator is one-use. This
does not provide cryptographic secrecy against the local process or code owner.
The receipt contains synthetic IDs, aggregate data digests, tiny synthetic
weights, scores, and metrics; it stores no raw examples or prompts.

## Current evidence record

- Assurance: `INTERNAL_PASS` for the deterministic local contract.
- Reviewer: `NOT_REQUIRED_FOR_LOCAL_SYNTHETIC_SCOPE`; no review task was
  submitted.
- Reproducibility: `SAME_OWNER_SINGLE_ENVIRONMENT_REPEATABILITY_CHECK`; no
  clean second-environment reproduction was performed.
- Custody: fixtures are generated in process; no model, provider, remote
  worker, user data, or raw trace is used. Keep generated receipts outside
  this repository unless a separately named packet requires their retention.
- Claim ceiling: `LocalDevelopmentSyntheticLearningEpisodeReceiptV1`.
- Nonclaims: no real model, mechanistic understanding, alignment, continual
  learning superiority, production readiness, or independent acceptance.
- Next authority gate: any model-bearing extension needs a fresh packet and
  independent protocol/execution review before execution.

## Open-source integration boundary

The existing upstream projects remain pinned and accessed through the
repository's lazy adapters. This protocol intentionally does not import them:
the current integrated-stack slice keeps model-bearing execution closed until
its exact external packet receives independent review. A future adapter can
replace the toy mechanism only in a separately authorized model-bearing slice
with exact model, runtime, feature asset, corpus, custody, and review bindings.

This synthetic utility is isolated under `tools/learning_episode_v1/` and is
outside the existing model-backed readiness packet's source roster. Its
runner/verifier bytes are bound directly in each receipt; it does not alter or
refresh the pending model-review packet.
