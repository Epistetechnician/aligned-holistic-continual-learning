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
identities for all seven upstream submodules are in
`third_party/UPSTREAMS.lock.json`.

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
`configs/corpus-manifest.v2.json`. The six fit, six tune, and six
assessment documents have disjoint authors. The manifest must record the exact
download URL, normalized bytes hash, tokenizer revision, tokenization policy,
and owner-only custody path. The corpus bytes never enter Git. The external
custody manifest currently binds 29 files and has digest
`767392572ad831d43e19f7a7d3ad485237d2532971233e72cc4b0fbaa281406c`.

## Dependency lanes

The upstreams cannot safely be installed as one unconstrained environment.
The pinned TransformerLens source currently requires Transformers 5.9 or
newer, while the pinned circuit-tracer source caps Transformers at 4.57.3.
The monorepo therefore uses isolated lanes and a shared aggregate schema:

- **Modern lane:** TransformerLens bridge, SAELens, NNsight, and pyvene. This
  lane owns activation capture, SAE encode/decode, and representation
  interventions.
- **Legacy circuit lane:** circuit-tracer and ACDC with their compatible
  TransformerLens/Transformers constraints. This lane owns attribution graphs
  and automated circuit discovery.
- **Visualization lane:** Neuronpedia's local API or static feature URLs. It
  receives aggregate feature metadata only; it is not an evidence authority.

The adapters in `src/ahcl/adapters/` prevent these lanes from sharing Python
module state. Trial packets exchange canonical IDs, tensors during one run,
and aggregate metrics after validation.

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

## Frozen statistics and controls

The family roster is 48 fit, 48 tune, and 48 assessment families with three
repeats per family. Fit rows select the monitor and any feature candidates.
Tune rows lock feature maps, layers, thresholds, update hyperparameters, and
the predictor. Assessment rows are unread until the lock digest exists.

The family-wise alpha is 0.05 with Holm correction. The design target is 0.80
power at ICC 0.50. Fixed controls are activation-only, text-only, exact-copy
no-op, shuffled-donor, constant-donor, and matched-donor controls. Missing or
non-finite rows fail closed; there is no attrition replacement.

## Review and execution gates

The review packet binds the upstream lock, model revision, asset revision and
hashes, corpus manifest, runtime lock, source digest, module registry,
estimands, controls, thresholds, retention policy, and validator digest. An
independent reviewer must return a packet-bound signed `ACCEPT`; the operator
cannot provide that receipt. Only then may the execution flag be enabled.

The execution order is protocol review, custody verification, qualification,
fit, tune lock, independent lock review, assessment, independent aggregate
recalculation, and claim writing. Any failed gate closes this state slice
without retuning. The ceiling before a passing held-out result is
`LocalDevelopmentIntegratedStackFeasibilityOnly`.
