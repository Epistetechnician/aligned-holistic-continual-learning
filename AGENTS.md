# Clean-room research repository rules

All mutations in this repository belong to the named state slice
`aligned-holistic-continual-learning-clean-room-v1`.

This repository is a clean-room implementation. It may implement public,
documented methods and cite public literature, but it must not copy code,
datasets, prompts, traces, generated outputs, private details, or hidden
artifacts from another repository. Every external method has a provenance
entry in `docs/literature.md`.

The initial experiment is synthetic only. Model-bearing execution, data
acquisition, provider calls, production traffic, and publication claims are
closed until a new reviewed state slice explicitly opens them. Aggregate
outputs are reproducible and contain no raw prompts, activations, or traces.

Keep the base implementation immutable for an experiment run. Prediction
locking happens before assessment metrics are computed. The validator is a
separate contract check and may reject malformed or self-inconsistent output.
No result is promoted by the runner itself.

Measure twice, cut once. Keep the codebase clean: no temporary files, dead
code, dead files, or untracked generated artifacts.
