# Luna readiness audit

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

Disposition: `NOT READY FOR QUALIFICATION`.

This is a read-only technical audit by an implementation agent. It is not an
independent external protocol/execution review, does not provide a signed
Ed25519 `ACCEPT`, and does not authorize model execution or assessment. No
model was loaded, no provider was called, and no corpus or feature asset was
downloaded.

The audit records the checkout state observed before concurrent additions to
the verified-review, assessment-boundary, and power-design surfaces. Those new
surfaces may close software-contract findings after their own validation, but
they do not retroactively refresh this audit, regenerate the packet, or supply
independent external authorization.

## Evidence collected

- The repository's upstream lock passes: `c7545c15404b911ca50ae7f00e369bd82e0439ebac81e698dd0d2a75c5536816`.
  All ten submodules resolve to their locked commits and have clean submodule
  worktrees (`src/ahcl/preflight.py:20-37`). Dependency-lane validation also
  passes with digest
  `e5dac1ec0fb40e835c4be5218dd845ce86420c01dce2f97b1cc328f98a8d71c6`.
- The public corpus manifest validates and has digest
  `5961c550e1f39cbfc4a55e11a7a8f7783c38401357fa6332f6cf88ab298a90c6`.
  This validates the 18-document metadata roster and six-document split
  counts (`src/ahcl/preflight.py:192-247`); it does not prove that the
  external corpus bytes are present or match custody.
- The recorded packet is internally self-consistent: packet digest
  `182cc2dbd1959c97059d1f55e14ecf50d8f2b5c1d2c27d13fca6b34caa9832bc`.
  It is stale against the live checkout. The live source digest is
  `7fd4e0a789a196aa80043c8e6c0c97f8e4b7ac0fcce69fb3ffd91e36884ec874`, versus
  packet field `26b6dd17aa879d17eb23b6d4961d96491b782dbeff1dd2707e3155226e58eb71`.
  The live config digest is
  `32e5bc6ae7af948dc347402e94a94b55ee0b7e3b1ad0fad1fef252b52e139766`, versus
  the packet's config digest
  `1790468f91ed86ca23387616ab3c2350c82cf1cc4c66e56b2bf7abba2518d121`.
  `configs/integrated-stack-v2.json` and `configs/review-packet.v2.json` are
  therefore not the exact same reviewed input.
- `execution.enabled` remains `false` in `configs/integrated-stack-v2.json`.
  `require_execution_open` fail-closes disabled execution, missing receipts,
  packet/config mismatch, and missing cryptographic verification
  (`src/ahcl/execution.py:18-39`). The review record remains
  `PENDING_INDEPENDENT_REVIEW` (`docs/review/integrated-stack-v2-review.md:5-35`).
- `AHCL_CUSTODY_ROOT` is unset in the audited shell, but the known external
  fallback `/Users/shaanp/.codex/custody/ahcl-integrated-stack-v2` exists with
  mode `0700`. Its `manifest.json` exists with mode `0600`, declares the correct
  state slice, manifest digest
  `059babda7993b9cb2625e1acba8c836ff1e2d8539eef9ebac1f10b29e9ba27a1`, and 29
  entries matching the configured roster. This audit inspected manifest
  metadata and directory/file permissions only; it did not read or hash model,
  SAE, or corpus bytes. The root must be explicitly bound through the launch
  environment and then fully validated, including every file hash and identity
  binding (`src/ahcl/preflight.py:141-189`; `src/ahcl/custody.py:65-103`).
- The interpreter is Python `3.14.5`, while the packet requires Python
  `3.11.11`. Metadata-only availability checks found no installed `torch`,
  `transformers`, `transformer_lens`, `sae_lens`, `nnsight`, `pyvene`, `acdc`,
  or `circuit_tracer`. This check used `importlib.util.find_spec` and package
  metadata only; it did not import or load a model. The focused hermetic port
  suite passes locally (`21 passed`), which establishes software contract
  behavior only.

## Remaining blockers

1. A fresh packet must be generated from the exact final source/config/docs
   bytes after all authorized edits settle. The current packet cannot authorize
   this checkout because its source and config identities differ.
2. An independent reviewer must inspect that exact packet and return a
   packet-bound signed Ed25519 `ACCEPT`. The registry and receipt must be
   provisioned outside the operator process; the local audit is not that
   acceptance. `verified_review.py` verifies the receipt shape, reviewer/key
   binding, purpose, packet/config equality, and signature, but it does not
   create external trust (`src/ahcl/verified_review.py:24-51`).
3. Bind the known external `0700` custody root explicitly (or provision a
   fresh root if the packet requires a new identity), then validate exact model
   revision/bytes, SAE revision/bytes, every corpus document hash, split roster,
   and retention policy before any model load. Manifest metadata and configured
   hashes alone are declarations until the validator hashes the files.
4. Recreate the pinned isolated environments, beginning with Python 3.11.11,
   and attest distribution hashes and imported module locations at launch.
   Current adapters enforce local model/SAE bytes only for TransformerLens and
   SAELens (`src/ahcl/adapters/transformerlens.py:31-70`,
   `src/ahcl/adapters/saelens.py:22-51`); NNsight, pyvene, ACDC, and
   circuit-tracer still accept caller-owned model objects or public identifiers
   and need lane-specific launch evidence (`src/ahcl/adapters/nnsight.py:28-39`,
   `src/ahcl/adapters/pyvene.py:20-37`, `src/ahcl/adapters/acdc.py:20-47`,
   `src/ahcl/adapters/circuit_tracer.py:22-49`).
5. Independently reproduce the declared family-wise power and analysis,
   including the Holm family, paired/repeated-unit covariance, and the exact
   primary endpoint. The current design record identifies the concern as open:
   `docs/review/parallel-falsification.md:52-63`.
6. After review and custody gates pass, run qualification only in the pinned
   offline lane. It must pass native/instrumented parity, deterministic
   repeatability, exact no-op identity, nonzero intervention reach, finite
   reconstruction, and held-out causal scrubbing (`src/ahcl/qualification.py:52-98`).
   Qualification output remains local feasibility evidence under the configured
   ceiling; it does not open assessment. Fit, tune, prediction lock, separate
   lock review, assessment, and aggregate validation remain subsequent gates
   (`src/ahcl/lifecycle.py:108-205`).

Until blockers 1–4 are closed and the exact packet receives an independent
signed `ACCEPT`, qualification must remain closed. No scientific or production
claim follows from the passing hermetic tests.
