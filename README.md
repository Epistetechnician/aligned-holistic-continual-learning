<!-- readme-v2:begin -->
# Aligned Holistic Continual Learning

[![CI](https://github.com/Epistetechnician/aligned-holistic-continual-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/Epistetechnician/aligned-holistic-continual-learning/actions/workflows/ci.yml) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) ![visibility: private staging](https://img.shields.io/badge/release-private%20staging-lightgrey) ![claims: bounded](https://img.shields.io/badge/claims-bounded%20local-orange)

> This clean-room monorepo tests a narrow, falsifiable question: can a fixed causal feature monitor predict several behavior-relevant consequences of a continual-learning update while retaining a locked base model? The first implementation is a deterministic synthetic qualification. It exercises the full research lifecycle without downloading a model or retaining raw traces.

## Contents

- [At a glance](#at-a-glance)
- [Key findings](#key-findings)
- [Verification and CI/CD acceptance](#verification-and-cicd-acceptance)
- [Quick start](#quick-start)
- [Detailed documentation](#detailed-documentation)
- [Governance, claims, and non-claims](#governance-claims-and-non-claims)
- [Security, contributing, license](#security-contributing-license)

## At a glance

| | |
|---|---|
| Stack | Python 3.12 + Rust workspace |
| Release stage | Private staging for research release; not yet public |
| License | Apache-2.0 (see [LICENSE](LICENSE), [NOTICE](NOTICE)) |
| Research report | [RESEARCH.md](RESEARCH.md) |
| Last full acceptance run | 2026-10-14 |
| Evidence ceiling | Bounded local replay. Nothing here is independent, benchmark, or production evidence. |

## Key findings

This is a summary of [RESEARCH.md](RESEARCH.md). Every result is a bounded local or synthetic observation, and negative results are retained rather than retuned away.

### Question

Can a fixed causal feature monitor predict several behaviour-relevant
consequences of a continual-learning update (adaptation gain, retention harm,
calibration harm, policy-violation harm) while the base model stays locked?

### What has been run

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

### Integrated (model-backed) experiment status

- Packet state: `READY_FOR_INDEPENDENT_REVIEW`, execution disabled.
- Open review item: declared power is a single-comparison approximation; Holm
  joint power is unresolved.
- Source loaders require exact local source manifests; launch-time runtime
  attestation covers the pinned interpreter, distribution, and imported modules.
- Modern and circuit-tracer lanes are attested; the ACDC lane is **blocked**
  (Torch 1.13.1 has no matching macOS arm64 wheel for Python 3.11.11).
- Pinned upstream sources are git submodules (TransformerLens, SAELens,
  circuit-tracer, nnsight, pyvene, ACDC, Neuronpedia). Do not edit them.

### What we conclude

The experimental lifecycle (stream, interventions, fit/tune lock, assessment,
validator, quarantining admission kernel) works end to end on a synthetic task
where the answer is known by construction. That qualifies the harness, not the
hypothesis.

### What this does not establish

No evidence of mechanistic understanding, alignment, continual-learning
superiority, or generalisation to a real model. A synthetic pass says nothing
about whether real-model features predict real update consequences.

## Verification and CI/CD acceptance

GitHub Actions workflow `ci.yml` runs on every push and pull request with read-only repository permissions (`contents: read`).

| Job | Runner | Gate |
|---|---|---|
| test | ubuntu-latest | `PYTHONPATH=src python3 -m unittest discover -s tests` |
| test (cont.) | ubuntu-latest | `cargo test --workspace` |

**Gate intent:** Python contract tests plus Rust workspace tests; includes review-packet digest check.

**Local acceptance (2026-10-14):** Python: 108 tests run, 106 passed, 2 skipped, 0 failed; Rust workspace: 6 tests passed, 0 failed.

Run the same gate locally:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests
cargo test --workspace
```

### Acceptance criteria

- The gate above exits 0 on a clean checkout of the default branch.
- No claim in this repository exceeds the evidence ceiling stated in [At a glance](#at-a-glance).
- A failing, skipped, or ignored check is reported as such; it is never reclassified as a pass.
- Commits to this repository are authored by the maintainer only.

## Quick start

```sh
git clone https://github.com/Epistetechnician/aligned-holistic-continual-learning.git
cd aligned-holistic-continual-learning
PYTHONPATH=src python3 -m unittest discover -s tests
cargo test --workspace
```

## Detailed documentation

The sections below are the repository's original design and implementation documentation, preserved in full.

<!-- readme-v2:detail -->

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
  --custody-root /path/to/owner-only/custody/ahcl-integrated-stack-v2
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
imported module bytes. Modern and circuit-tracer lane attestations are retained
outside Git; the ACDC lane remains blocked because Torch 1.13.1 has no matching
macOS arm64 wheel for Python 3.11.11. No model-backed result is reported.

## Research report and project policy

See [RESEARCH.md](RESEARCH.md) for findings, [SECURITY.md](SECURITY.md) to report
issues, and [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under
[Apache-2.0](LICENSE). Submodules under `third_party/` keep their own licenses.

---
<!-- readme-v2:footer -->
## Governance, claims, and non-claims

Work proceeds under named state slices recorded in [AGENTS.md](AGENTS.md). The claim ceiling stated above is a hard limit: a local pass is not independent verification, a benchmark result, a proof of the whole system, or a statement about production readiness. See "What this does not establish" in [RESEARCH.md](RESEARCH.md).

## Security, contributing, license

- Report vulnerabilities per [SECURITY.md](SECURITY.md).
- Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
- Licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE).
