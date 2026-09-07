# End-to-end experiment protocol

State slice: `aligned-holistic-continual-learning-clean-room-v1`.

This document is the immutable synthetic baseline. The integrated model-backed
continuation is specified separately in `docs/integrated-experiment-v2.md`.

## Question and hypothesis

The question is whether a small, fixed causal feature map can predict several
behavior-relevant consequences of a continual-learning update from the update
itself. The hypothesis is restricted to the synthetic generator: a locked
three-coordinate map plus its three pairwise interactions predicts adaptation
gain, retention harm, calibration harm, and policy-violation harm better than a
constant behavioral baseline and a shuffled-feature control.

This is a qualification of software mechanics, not a claim about a trained
language model or human alignment.

## Frozen design

The roster has 144 independent family identities: 48 fit, 48 tune, and 48
assessment. Every family has three deterministic repeats. The four arms are
`untouched_base`, `fixed_update`, `retention_guard`, and `sham_update`. A family
contains a protected vector and an incoming-task vector. The update uses a
fixed learning rate of 0.18. The guard either applies that exact update or
zeros it when the squared protected-feature cost exceeds 0.0018. The sham arm
sign-flips the first three update coordinates while preserving the remaining
coordinates.

The monitor observes exactly three update coordinates. Its fixed seven-term
 design is `[1, f0, f1, f2, f0*f1, f0*f2, f1*f2]`; ridge strength is 0.1. It is
fit only on collapsed fit rows from `fixed_update`. Tune rows are used once to
create the prediction lock. Assessment rows are evaluated only after that lock
is fixed.

## Endpoints and gates

For each of the four axes, the primary endpoint is held-out coefficient of
determination (`R²`) and sign agreement. The locked gates are sign agreement at
least 0.80, `R²` at least 0.25, and an `R²` improvement over the constant
baseline of at least 0.20. The shuffled-feature control must have `R²` at most
0.20 on every axis. The runner publishes all arm aggregates, but only
`fixed_update` is the primary causal-monitor arm.

The validator checks the roster, canonical configuration digest, metric shape,
finite values, control fields, status logic, and final aggregate digest. It
does not pretend to independently reproduce a model computation. A future
scientific slice must add an independent numerical implementation and fresh
review before any model-bearing assessment.

## Execution order

1. Freeze the protocol and canonical configuration.
2. Generate the deterministic synthetic roster in memory.
3. Verify exact no-op identity and intervention reach.
4. Fit the monitor on the fit split.
5. Compute tune metrics and create the prediction lock digest.
6. Compute assessment and control aggregates without changing the monitor.
7. Write one aggregate JSON result with no raw rows.
8. Run the independent contract validator.
9. Present the status as `SyntheticCandidate` or `SyntheticNoCandidate`.

Any later real-model experiment requires a new state slice. It must specify
model/runtime/asset identities, data custody, raw-retention limits, independent
review, multiplicity and power assumptions, and a claim ceiling before model
execution. The current slice authorizes none of that.
