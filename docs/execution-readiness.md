# Execution readiness implementation

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

This pass uses Luna exclusively for subagents. Its task list is: inspect the
live checkout; implement durable attempt custody, document-level power, and a
concrete signature/predictor boundary; run adversarial fixtures; preserve
concurrent edits; deliver unresolved qualification prerequisites.

## Implemented boundary

`verified_review.verify_launch_receipt` verifies a real Ed25519 signature over
packet, reviewer, key, decision, purpose, and domain. It validates the canonical
packet digest and exact configuration. The host must provision a reviewer-key
registry independently; accepting an operator-chosen registry would defeat that
trust boundary. Test-generated keys never authorize a real experiment.

`SealedLinearPredictor` copies numerical weights into immutable tuples, uses
identity feature transformation, and computes directly from the serialized
parameters. It excludes arbitrary predictor callbacks and mutable model state.
An arbitrary feature extractor is not covered by its parameter digest: the
reviewed caller must bind extraction code, scaling, feature order, and inputs.

`assessment_boundary.assess_once` composes receipt verification, predictor
identity checks, and a durable exclusive attempt claim before evaluator access.
`assessment_custody` checks root ownership/permissions, uses a directory file
descriptor and exclusive creation, and synchronizes the marker and directory.
Restart, concurrent calls, or evaluator failure do not release the claim.
The directory owner can delete markers. External custody is still required for
protection against a malicious operator and for preventing a new packet from
querying the same holdout. This is a local guard, not adversarial isolation.

These APIs are additive. The legacy model loader/lifecycle does not yet route
all execution through this boundary; no claim of universal enforcement is made.
No model-bearing callback was invoked: tests use arithmetic and deliberate
exceptions only. Callers must also attest evaluator code and raw retention.

## Statistical design

`power_design` treats each paired document contrast as one independent unit.
The standardized effect is defined at that document level. Repeats are metadata
and cannot increase effective N. Bonferroni per-contrast alpha conservatively
supports Holm testing; a union bound distinguishes marginal from joint power.
The normal approximation is a planning tool, not finite-sample assurance. Final
power requires reviewed effect-size assumptions and the exact analysis model.
See `power-design.md`. No current six-document assessment is declared powered.

## Qualification status and handoff

Status: **not qualified; no independent execution ACCEPT**.

The exact readiness packet has now been regenerated after the source and
configuration bytes settled. Read-only preflight passes with status
`READY_FOR_INDEPENDENT_REVIEW`; model execution remains disabled. The known
external custody path is validated explicitly rather than inferred from an
unset environment variable. Host qualification is complete on the dedicated
Linux x86_64 givemeanode lane, but no model execution or assessment has been
performed.

After any authorized source, configuration, lane, or protocol edit, rebuild
the tracked packet only after the checkout is stable:

```sh
PYTHONPATH=src python3 -m ahcl.refresh_packet
AHCL_CUSTODY_ROOT=/owner-only/path pnpm run lint
```

The refresh command validates the public manifests and source identity before
writing `configs/review-packet.v2.json`. It does not inspect model bytes, sign
review receipts, install dependencies, or open execution.

When an operator has created a pinned environment, produce one lane document
per declared execution lane without loading a model:

```sh
PYTHONPATH=src python3 -m ahcl.runtime_attestation \
  --lane modern --environment /owner-only/runtime/modern
PYTHONPATH=src python3 -m ahcl.runtime_attestation \
  --lane circuit_tracer --environment /owner-only/runtime/circuit-tracer
PYTHONPATH=src python3 -m ahcl.runtime_attestation \
  --lane acdc --environment /owner-only/runtime/acdc
```

The command runs the attestation inside the selected interpreter, hashes its
installed distribution files and imported module bytes, and emits canonical
JSON to stdout. Each distribution roster is rooted at the selected environment
so executable files outside `site-packages` are covered too. Retain those
documents in the external review/custody system; do not commit them. The
qualified ACDC lane is Python `3.11.11` with Torch `1.13.1`, and the qualified
NNsight/circuit-tracer lane is Python `3.11.11` with the pinned NNsight source
commit and a native extension built from that checkout. The current lane
attestation digests are ACDC
`3f95e7e684c2ff81e04785fa6552f08b50cfbd74c518b2ba12fe408c4f5a7cd3` and
NNsight/circuit-tracer
`a939a5b8e72fa9d733f52ba6adacf1dd862c41bc726019bee83434d9f484fb41`.
Source attestation digests are ACDC
`f226d5d54454f1e4db0e0b0431b52579b896c244be78e79c94936ace0ad067ca` and
NNsight
`1906a2580d4a0475719643c7cf9df6d43547f3ab4fe37218567db24e6e7eb687`.
The integrated preflight still reports the family-wise planning outcome as
`NoCandidate`/`unvalidated` until the reviewed cohort redesign is adopted.

Before qualification: freeze the checkout and installed runtime bytes; bind
model, asset, tokenizer and the fresh family roster; keep the fixed power
numbers; derive the immutable launch packet from readiness; and obtain
externally provisioned reviewer trust plus signed acceptance over that launch
packet. Only then execute offline parity/repeat/no-op/reach and reconstruction
qualification. A failure closes that qualification without threshold or asset
shopping. Fit/tune/assessment remain downstream.

An independent Luna agent produced `review/luna-readiness-audit.md`; that is a
technical snapshot and must not be represented as an external signed ACCEPT.
