# Integrated model-backed experiment

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

This document is the executable design for the next slice. The current
repository has the adapters, custody manifest, and contract checks, but
execution remains closed until the packet is independently reviewed.

## Pinned inputs

The first model is `openai-community/gpt2` at Hugging Face revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. It is small, ungated, and exposes
the same causal language-model objective across the selected backends. The
runtime target is Python 3.11.11 and Rust 1.87.0. The source commits and license
identities for all ten upstream submodules are in
`third_party/UPSTREAMS.lock.json`.

The modern lane is pinned to Torch 2.6.0 and Transformers 5.9.0. The
circuit-tracer lane is pinned to Torch 2.6.0, Transformers 4.57.3, NNsight
0.6.1, and TransformerLens 2.16.1. It uses the exact NNsight v0.6.1 and
TransformerLens v2.16.1 submodules. Torch 2.6.0 is required by TransformerLens
v2.16.1 on Python 3.11; lower Torch pins would make this lane unsatisfiable.
The ACDC lane is pinned to NumPy 1.26.4, Torch 1.13.1, Transformers 4.35.2,
and TransformerLens 1.6.1. These exact versions are recorded in
`configs/dependency-lanes.json` and in the review packet.

The feature asset is SAELens release
`gpt2-small-mlp-out-v5-32k`, repository
`jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs`, layer path
`v5_32k_layer_6`, at repository revision
`123b02d81b5b2b542afa1503b2fb5f1c70b98681`. Its weights are custodied with
SHA-256
`3105f1072fa405f49c442269f43c553fbdc3e6526b6bc0445eaa6c5cd683838b`.
The complete file list is bound by the external manifest digest recorded in
`configs/integrated-stack-v2.json`.

The corpus is 18 Project Gutenberg documents listed in
`configs/corpus-manifest.v2.json`. The six fit, six tune, and six assessment
documents have disjoint authors. Each document owns eight explicit family IDs;
the validator requires the complete `fit-000` through `assessment-047` roster
and binds each document byte hash to custody. The manifest must record the
exact download URL, normalized bytes hash, tokenizer revision, tokenization
policy, and owner-only custody path. The corpus bytes never enter Git. The
external custody manifest currently binds 29 files and has digest
`059babda7993b9cb2625e1acba8c836ff1e2d8539eef9ebac1f10b29e9ba27a1`.

## Dependency lanes

The upstreams cannot safely be installed as one unconstrained environment.
The pinned TransformerLens source requires Transformers 5.9 or newer, while
circuit-tracer requires Transformers 4.57.3 and ACDC requires Transformers
4.35 plus Torch below 2.0. The monorepo therefore uses isolated lanes and a
shared aggregate schema:

- **Modern lane:** TransformerLens bridge, SAELens, NNsight, and pyvene. This
  lane owns activation capture, SAE encode/decode, and representation
  interventions.
- **Circuit-tracer lane:** circuit-tracer with NNsight 0.6.1, TransformerLens
  2.16.1, Transformers 4.57.3, and Torch 2.6.0. This lane owns feature
  interventions and attribution graphs.
- **ACDC lane:** ACDC with TransformerLens 1.6.1, Transformers 4.35.2, and
  Torch 1.13.1. This lane owns automated circuit discovery.
- **Visualization lane:** Neuronpedia's local API or static feature URLs. It
  receives aggregate feature metadata only; it is not an evidence authority.

The adapters in `src/ahcl/adapters/` prevent these lanes from sharing Python
module state. Model and SAE loading requires an exact revision, a caller-owned
local custody root, a matching weight digest, and an imported module location
inside the pinned checkout. NNsight and circuit-tracer additionally require a
caller-supplied exact source-file roster with SHA-256 digests; the loader
rejects symlinks, roster drift, path escapes, and unlisted module bytes before
constructing a tokenizer or model. `ahcl.runtime_attestation` hashes the resolved
Python interpreter, every installed distribution file, and each imported
module's bytes, and binds imported module locations to the selected lane; the
execution gate rejects an enabled packet without that attestation. Trial
packets exchange canonical IDs, tensors during one run, and aggregate metrics
after validation.

## Update and behavior endpoints

Each family starts from an immutable base model. The incoming stream is split
into ordered batches. The primary update is a bounded PyTorch gradient policy;
the first retention comparator uses an EWC penalty with a frozen reference and
fisher diagonal. A plain SGD update is the behavioral baseline. Adapter or
weight changes are allowed only inside the update policy and are hashed before
the next evaluation checkpoint.

For every checkpoint, the evaluator records next-token cross-entropy,
exact-match accuracy, expected calibration error, and a policy-violation rate
over a predeclared forbidden-token test set. The protected document families
are measured again after every incoming-task segment. The primary tradeoff is
incoming-task improvement at a fixed ceiling on protected-family degradation.
The monitor-controlled policy rejects an update only when the locked predictor
crosses the retention ceiling without meeting the adaptation floor. The policy
records applied and skipped counts and fails a trajectory that rejects every
update. Ordinary fixed updates, EWC, and the behavior-only guard remain
predeclared comparators.

The model-free summaries in `src/ahcl/research_tracks.py` make the remaining
claims measurable without opening model execution. `feature_stability`
compares fixed-ID causal-effect vectors across checkpoints without rematching;
`alternative_predictor_features` supplies hand-calculable gradient, update,
loss, and representation baselines; `horizon_summary` requires a complete
ordered checkpoint trajectory and reports cumulative positive harm; and
`selective_repair` requires measured damage, recovery, and preservation of new
learning. These functions summarize caller-owned measurements and are not
scientific evidence by themselves.

## Interpretability measurements

For the same ordered batch and checkpoint, the modern lane captures named
TransformerLens or NNsight activations. SAELens encodes the selected MLP-out
activation and records normalized reconstruction error. pyvene provides a
serializable replacement intervention. circuit-tracer runs a separate graph
or feature intervention when its compatible lane is available. ACDC is used as
a discovery comparator, never as a causal proof by itself. Neuronpedia is
optional visualization of already validated feature metadata.

Every backend must pass, in order:

1. native versus instrumented parity;
2. deterministic repeatability on the same batch;
3. exact no-op identity;
4. nonzero intervention reach at the measured output;
5. finite normalized reconstruction error;
6. held-out causal scrubbing against the locked target.

The shared numerical checks are implemented in `src/ahcl/checks.py`.
`src/ahcl/qualification.py` composes them into one aggregate qualification
record, so a backend cannot satisfy one check while bypassing another.

## Frozen statistics and controls

The family roster is 48 fit, 48 tune, and 48 assessment families with three
repeats per family. Fit rows select the monitor and any feature candidates.
Tune rows lock feature maps, layers, thresholds, update hyperparameters, and
the predictor. Assessment rows are unread until the lock digest exists.

The family-wise alpha is 0.05 with Holm correction. The design records a
minimum standardized effect of 0.50 and a single-comparison two-sided normal
approximation after the ICC 0.50 design effect for three repeats. The
independent unit is explicitly a document with eight families per document;
the four-endpoint Holm union-bound lower bound is recorded separately and the
power design status is `unvalidated`. This is a design target, not a validated
family-wise power claim; the joint-power calculation remains an independent
review blocker. Fixed controls are
activation-only, text-only, exact-copy no-op, shuffled-donor,
constant-donor, and matched-donor controls. Missing or non-finite rows fail
closed; there is no attrition replacement.

The machine-readable version of this contract is in the `study` and
`controls` fields of `configs/integrated-stack-v2.json`. `src/ahcl/lifecycle.py`
enforces phase order and cohort disjointness without retaining raw rows.
`src/ahcl/aggregate.py` independently recomputes row-level R², baseline R²,
sign agreement, and the fixed family-level shuffled control for validator use
during the 72-hour custody window. The constant baseline is supplied as a
fit-only parameter vector and validated against the outcome width; it is never
estimated from assessment outcomes. Its prediction-bound entry point accepts
stored predictions directly, so the validator does not need to import the
fitted monitor or model runtime.

## Review and execution gates

The review packet binds the upstream lock, model revision, asset revision and
hashes, corpus manifest, runtime lock, source digest, module registry,
estimands, controls, thresholds, retention policy, and validator digest. An
independent reviewer must return a packet-bound signed `ACCEPT`; the operator
cannot provide that receipt. The tracked readiness packet keeps
`execution.enabled: false`. After the readiness inputs are frozen, derive a
separate launch packet with `build_launch_packet`; that immutable artifact
sets `execution.enabled: true`, receives a new digest, and is the only packet
that may be signed and used at launch. Never mutate a signed packet.

The execution order is protocol review, custody verification, qualification,
fit, tune lock, independent lock review, assessment, independent aggregate
recalculation, and claim writing. Any failed gate closes this state slice
without retuning. The ceiling before a passing held-out result is
`LocalDevelopmentIntegratedStackFeasibilityOnly`.
