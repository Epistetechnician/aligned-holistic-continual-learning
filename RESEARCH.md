# Research report and findings

## Question

Can a fixed causal feature monitor predict several behaviour-relevant
consequences of a continual-learning update (adaptation gain, retention harm,
calibration harm, policy-violation harm) while the base model stays locked?

## What has been run

Only a deterministic **synthetic qualification** has been executed. No model
has been downloaded or run, no raw traces are retained, and no model-backed
result exists. The model-backed "integrated" experiment (v2) is fully designed
and fail-closed pending independent review.

### Synthetic result

Reproduce with the commands in the README (`ahcl.run`, then `ahcl.validate`).
Status: `SyntheticCandidate`; validator: valid; all primary and control gates
passed (gates: R² ≥ 0.25, R² over baseline ≥ 0.20, sign agreement ≥ 0.80,
shuffled-control R² ≤ 0.20).

| Target | Assessment R² | Sign agreement |
| --- | ---: | ---: |
| Adaptation gain | 0.830 | 0.979 |
| Retention harm | 0.853 | 0.979 |
| Policy-violation harm | 0.830 | 0.979 |
| Calibration harm | 0.900 | 0.938 |

Shuffled-label controls scored negative R² on every target (-0.27 to -0.51), as
required. Tune-split numbers are close to assessment numbers (R² 0.82–0.91),
so the locked prediction did not degrade out of sample. The monitor is compared
with fixed-update, EWC, and behavioural-guard policies; the per-arm table is in
the result JSON. Some baseline arms score poorly (for example a retention-guard
arm had negative calibration-harm R²), which is expected behaviour of those
baselines on this fixture.

## Integrated (model-backed) experiment status

- Packet state: `READY_FOR_INDEPENDENT_REVIEW`, execution disabled.
- Open review item: declared power is a single-comparison approximation; Holm
  joint power is unresolved.
- Source loaders require exact local source manifests; launch-time runtime
  attestation covers the pinned interpreter, distribution, and imported modules.
- Modern and circuit-tracer lanes are attested; the ACDC lane is **blocked**
  (Torch 1.13.1 has no matching macOS arm64 wheel for Python 3.11.11).
- Pinned upstream sources are git submodules (TransformerLens, SAELens,
  circuit-tracer, nnsight, pyvene, ACDC, Neuronpedia). Do not edit them.

## What we conclude

The experimental lifecycle (stream, interventions, fit/tune lock, assessment,
validator, quarantining admission kernel) works end to end on a synthetic task
where the answer is known by construction. That qualifies the harness, not the
hypothesis.

## What this does not establish

No evidence of mechanistic understanding, alignment, continual-learning
superiority, or generalisation to a real model. A synthetic pass says nothing
about whether real-model features predict real update consequences.

## Reproduce

```sh
git submodule update --init --recursive      # only for the integrated stack
PYTHONPATH=src python3 -m ahcl.run --output results/synthetic-result.json
PYTHONPATH=src python3 -m ahcl.validate results/synthetic-result.json
PYTHONPATH=src python3 -m unittest discover -s tests   # 108 tests (2 skipped)
cargo test --workspace                                  # Rust admission kernel
```
