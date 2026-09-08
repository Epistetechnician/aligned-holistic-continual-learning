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

The exact packet has now been regenerated after the source and configuration
bytes settled. Read-only preflight passes with status
`READY_FOR_INDEPENDENT_REVIEW`; model execution remains disabled. The known
external custody path is validated explicitly rather than inferred from an
unset environment variable. The active interpreter and installed model
packages still do not match the configured runtime lane.

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
modern lane is attested at
`2593b1d04d85f2a652f6374eb18c6bc690452fb5bcbcc1a4495326d8bd04ef6e`, and the
circuit-tracer lane at
`c5cbd67d3869abbfa29d8d36fd8ed3b23c3644d20d3e65440d7934f1d638ac1e`. The
ACDC lane remains blocked on this macOS arm64 host because its locked Torch
1.13.1 requirement has no matching Python 3.11.11 wheel. The integrated
preflight also reports the current family-wise planning outcome as
`NoCandidate`/`unvalidated` until the reviewed cohort redesign is adopted.

The circuit-tracer replacement-model import also passes the exact 130-file
manifest for submodule commit
`6018ed8d35e40f2c50062822e8dde422b8e52e2d`. The NNsight 0.6.1 source manifest
is complete at 170 files, but importing that immutable checkout requires its
native `nnsight._c.py_mount` extension to be built; the current checkout does
not contain that build artifact. The installed NNsight wheel is not accepted
as a substitute for the pinned source checkout, so NNsight local-byte
qualification remains open.

Before qualification: finish integration through one mandatory launch path;
freeze checkout and installed runtime bytes; bind model, asset, tokenizer and
fresh family roster; fix independent unit, contrasts, effect assumptions and
numeric resource caps; run `ahcl.runtime_attestation` in each selected pinned
lane, including the resolved interpreter, distribution, and imported module
bytes; obtain
externally provisioned reviewer trust and signed acceptance. Only
then execute offline parity/repeat/no-op/reach and reconstruction
qualification. A failure closes that qualification without threshold or asset
shopping. Fit/tune/assessment remain downstream.

An independent Luna agent produced `review/luna-readiness-audit.md`; that is a
technical snapshot and must not be represented as an external signed ACCEPT.
