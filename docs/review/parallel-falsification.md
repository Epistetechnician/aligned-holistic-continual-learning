# Parallel technical falsification audit

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

Disposition: **NOT READY FOR MODEL EXECUTION OR SCIENTIFIC ASSESSMENT**.
This is an independent agent technical audit of the inspected working tree,
not an external protocol/execution authorization and not a signed ACCEPT.
No model, provider, corpus download, or assessment was run. Other workstreams
were modifying measurement helpers; those helpers are outside this snapshot.

## Findings and rejection criteria

1. **Critical: execution authority is caller supplied.**
   [review.py](../../src/ahcl/review.py#L32) now requires an Ed25519 verifier
   callback, correcting the earlier signature-presence-only behavior. However,
   the caller supplies that callback and reviewer/key identities. A receipt
   with signature `not-a-signature`, key `unregistered`, and `lambda *_: True`
   passed a hermetic probe. [execution.py](../../src/ahcl/execution.py#L21)
   does not bind its caller-supplied config to the reviewed packet bytes.
   [common.py](../../src/ahcl/adapters/common.py#L16) admits adapters with a
   boolean alone. Reject execution until a concrete cryptographic verifier,
   trusted reviewer-key registry, canonical signed message, packet/config
   equality, and one mandatory launch boundary are implemented and tested.
   A default-disabled boolean is useful accident prevention, not verified
   independent authorization.

2. **Critical: the prediction lock does not seal execution or consume assessment.**
   [lifecycle.py](../../src/ahcl/lifecycle.py#L115) stores a predictor and a
   separately supplied payload without checking that they describe the same
   computation. Config/payload copies are shallow. The tune callback can mutate
   the predictor; assessment does not recompute its identity. At
   [assess](../../src/ahcl/lifecycle.py#L159), phase remains `locked`, permitting
   repeated assessment with new callbacks. No independent tune-lock receipt is
   required. Reject assessment until prediction bytes and executable parameters
   are bound, mutation is detected, a lock-review receipt is verified, and
   success or failure consumes the assessment attempt irreversibly. An iterable
   of already computed assessment outcomes cannot itself prove pre-effect lock
   ordering; generation must occur behind the same boundary.

3. **High: corpus disjointness is metadata-only.**
   [validate_corpus_manifest](../../src/ahcl/preflight.py#L166) checks counts and
   unique author strings but does not validate unique document IDs, canonical
   file hashes, cross-split content duplication, or document-to-family ancestry.
   A probe replacing every document ID with `1342` and every hash with `invalid`
   passed after recalculating the manifest digest. Input binding checks corpus
   counts, not each document hash against custody. Reject until exact document
   and tokenized segment rosters, source byte bindings, duplicate/overlap
   exclusions, and the 18-document to 144-family mapping are executable.
   Author disjointness alone does not establish family independence or absence
   of pretraining contamination.

4. **High: power is calculated but does not establish the declared family-wise power.**
   [achieved_power](../../src/ahcl/stats.py#L99) implements a two-arm normal
   approximation with repeat ICC and unadjusted alpha. It has no multiplicity
   count, primary endpoint roster, paired-contrast covariance, document-level
   dependence, or joint-success requirement. Hermetic calculation gives
   `0.8508387683270562` for its current assumptions; substituting alpha `0.05/4`
   gives `0.69226983948938`. The latter is an illustrative conservative
   four-test comparison, not an estimate of actual Holm joint power.
   Reject an 80% powered-study claim until the exact estimand, independent unit,
   minimally important effect, multiplicity family, and analysis agree with
   an independently reproduced analytic or simulated power calculation.
   Do not increase sample sizes or relax thresholds after scientific effects.

5. **High: aggregate reproduction lacks family-level input validation.**
   [aggregate.py](../../src/ahcl/aggregate.py#L55) never checks family IDs,
   split IDs, duplicate repeats, cohort completeness, or repeat collapse.
   Its constant baseline is estimated from the evaluation outcomes themselves
   (line 80), so it is an evaluation-mean reference, not a deployable fit-only
   predictor. Shuffling is row-level, not family-level. Reject independent
   scientific reproduction until row custody, locked prediction identity,
   family collapse, full roster validation, fit-only baseline parameters,
   uncertainty and multiplicity are validated separately. Not importing the
   production monitor is a useful separation, not proof of independent review.

6. **High: source pins do not yet attest loaded execution bytes.**
   [validate_upstream_lock](../../src/ahcl/preflight.py#L20) checks HEAD and
   license existence but ignores modified/untracked submodule files. Lazy
   imports may resolve installed packages unrelated to those checkouts.
   [source_tree_digest](../../src/ahcl/preflight.py#L101) correctly excludes
   bytecode, preventing ordinary Python cache creation from changing source
   identity. It also excludes AGENTS.md, pyproject.toml, Cargo files/crates,
   and all review documents. Reject exact-execution-identity claims until
   instructions, build manifests, runtime distributions and loaded module
   locations/bytes are attested and checked at launch. Build artifacts need
   separate identities; adding pycache back to source identity is not a fix.

7. **High: pinned inputs are not enforced by loaders.**
   [TransformerLens boot](../../src/ahcl/adapters/transformerlens.py#L23)
   defaults to revision `main` and lacks enforced local-only loading.
   [SAELens loader](../../src/ahcl/adapters/saelens.py#L22) does not accept the
   pinned repository revision/hash or local custody path, and its default
   `layer_6` differs from configured `v5_32k_layer_6`. Reject live qualification
   until these loaders consume validated local bytes and produce the exact
   model/asset/runtime identity. Passing helper-array tests cannot establish
   native/instrumented parity for this actual stack.

8. **Medium: missing statistical values can become valid-looking output.**
   [holm_adjust](../../src/ahcl/stats.py#L85) checks numeric range without a
   finiteness check. A hermetic `holm_adjust([float('nan')])` returned `(1.0,)`.
   Reject a batch containing any missing/nonfinite statistical value rather
   than replacing it with a conservative-looking adjusted value.

## Parallel deliverable and stopping rule

The workstream deliverable is this falsification checklist and independent
reproduction of the four probes above. The stopping rule is continued closure
of model execution and held-out assessment while any critical or high finding
is unresolved. Each resolution requires targeted negative tests and a fresh
review of exact final bytes. This report must not be relabeled as external
ACCEPT, nor used to authorize the operator to sign its own receipt.

Scientific conclusions remain absent. Existing contract tests can establish
local software behavior only; they cannot establish feature stability, causal
scrubbing validity, continual-learning benefit, alignment, or selective repair.

## Local resolution log

The following implementation changes close software-contract portions of the
findings; they do not convert this audit into an independent ACCEPT:

- Finding 1: `review.py` now signs a canonical packet message, requires a
  trusted public-key registry, and `execution.py` binds the reviewed packet
  digest and configuration at launch. A real external reviewer key registry
  and receipt are still absent.
- Finding 2: `lifecycle.py` deep-copies locked inputs, verifies configuration
  and predictor payload digests, and consumes assessment on first entry,
  including failed attempts. A separately reviewed tune receipt is still an
  execution prerequisite.
- Finding 3: the corpus manifest now requires unique IDs, canonical hashes,
  and eight explicit split-bound family IDs per document; the complete
  48-family-per-split roster is checked. `validate_corpus_custody` binds every
  document file and byte hash to the external manifest. Tokenized segment
  ancestry, contained-passage exclusion, and contamination analysis remain
  open scientific work.
- Finding 5: aggregate rows now require a complete family/repeat roster,
  constant baselines must be supplied from fit-only parameters, and the
  shuffled control permutes donor families while preserving repeat identity.
  Uncertainty and multiplicity aggregation remain outside this helper.
- Finding 8: non-finite and empty Holm inputs fail closed.

Findings 4 and 6 remain open. Finding 7 is partially resolved: the
TransformerLens and SAELens adapters now require a caller-supplied source
checkout, exact revision, local custody root, and weight digest before loading;
the NNsight and circuit-tracer model paths still require the same local-byte
boundary. `runtime_attestation.py` now hashes installed distribution files and
the execution gate requires its canonical evidence, but this host has not
produced an attestation for the pinned Python 3.11.11 lanes. The declared
power is still a single-comparison approximation. Model execution and
assessment remain closed until those items receive a fresh independent review
of the exact final bytes.
