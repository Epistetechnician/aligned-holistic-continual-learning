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
accepts its packet.

The repository has three bounded layers:

- `src/ahcl/` contains the synthetic stream, fixed feature interventions,
  ridge monitor, fit/tune lock, assessment metrics, aggregate output, and
  vendor-neutral adapter ports. The public backend registry is quarantined.
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

The synthetic status is a local implementation contract. It is not evidence
of mechanistic understanding, alignment, continual-learning superiority, or
generalization to a real model.
