# Aligned Holistic Continual Learning

This clean-room monorepo tests a narrow, falsifiable question: can a fixed
causal feature monitor predict several behavior-relevant consequences of a
continual-learning update while retaining a locked base model? The first
implementation is a deterministic synthetic qualification. It exercises the
full research lifecycle without downloading a model or retaining raw traces.

The integrated upstream slice adds pinned source submodules and lazy adapters.
Install them with `git submodule update --init --recursive`; do not edit the
submodule contents. The model-backed design is recorded in
`docs/integrated-experiment-v2.md` and is fail-closed until independent review
accepts its packet. State slice:
`aligned-holistic-continual-learning-integrated-stack-v2`.

The repository has three bounded layers:

- `src/ahcl/` contains the synthetic stream, fixed feature interventions,
  ridge monitor, fit/tune lock, assessment metrics, aggregate output, and
  vendor-neutral adapter ports. The integrated lifecycle and independent
  aggregate calculator are fail-closed until the review gate opens. The
  monitor controller is compared with fixed-update, EWC, and behavioral-guard
  policies. `research_tracks.py` adds model-free summaries for feature
  stability, alternative predictors, horizon harm, and selective repair. The
  public backend registry is quarantined.
- `crates/admission-core/` contains a dependency-free Rust policy kernel. It
  can quarantine a candidate until independent held-out evidence and a locked
  prediction packet exist.
- `docs/` contains the protocol, architecture, and public literature register.

Run the experiment and validate its aggregate result:

```sh
PYTHONPATH=src python3 -m ahcl.run --output results/synthetic-result.json
PYTHONPATH=src python3 -m ahcl.validate results/synthetic-result.json
```

Run the hermetic checks:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
cargo test --workspace
```

Run the full repository gate with an explicitly bound owner-only custody root:

```sh
AHCL_CUSTODY_ROOT=/owner-only/path pnpm run lint
```

Run the read-only integrated preflight against the external custody root:

```sh
PYTHONPATH=src python3 -m ahcl.integrated_preflight \
  --custody-root /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2
```

It reports `READY_FOR_INDEPENDENT_REVIEW` while leaving model execution
disabled.

The synthetic status is a local implementation contract. It is not evidence
of mechanistic understanding, alignment, continual-learning superiority, or
generalization to a real model.

The integrated packet is currently `READY_FOR_INDEPENDENT_REVIEW`, with
execution disabled. Its declared power is a single-comparison approximation;
Holm joint power remains an unresolved review item. NNsight and circuit-tracer
loaders now require exact caller-supplied local source manifests in addition
to their existing local model and transcoder custody checks. A launch-time runtime
attestation implementation covers the pinned interpreter, distribution, and
imported module bytes, but this host has not produced one for the
pinned Python 3.11.11 lanes. No model-backed result is reported.
