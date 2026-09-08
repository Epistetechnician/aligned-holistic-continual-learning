# Measurement reliability workstream

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

The evaluator now explicitly shifts causal-LM logits at position t against labels
at t+1. Both source and target attention-mask entries must be active; ignored
labels remain ignored. Padding is passed into the model, and policy-token rates
use the same scored-token denominator. Missing masks and empty scoring fail.
The preexisting `exact_match_accuracy` key denotes token accuracy, not whole
sequence exact match. Forbidden-token argmax rate is a lexical proxy, not a
general safety or alignment endpoint. Low-level metric functions accept already
aligned tensors; only the evaluator calls `align_next_tokens`.

Numerical qualification rejects shape mismatches even with identical element
counts, ragged or empty arrays, nonfinite values anywhere, and invalid thresholds.
Reconstruction NMSE currently uses uncentered total input energy as denominator;
it is not pooled centered variance. The exact estimand must be reviewed before
effects. `causal_scrub_effect` measures output displacement only; it does not
implement a causal abstraction, matched resampling, or held-out causal scrubbing.

## Deliverable and stop rule

Deliver one digest-bound native/instrumented path only after exact runtime locks,
independent protocol/execution acceptance, and actual offline qualification.
Require identical tokenization, masks, precision, model mode, and inputs;
predeclare no-op and intervention operator semantics and reconstruction estimand.
Record parity, repeatability, exact replacement identities, nonzero reach, and
asset reconstruction on the authorized qualification split. Any failed gate
stops qualification; do not shop for thresholds or assets after effects.

Current tests provide hermetic numerical checks and optional tensor fixtures.
PyTorch is absent in the current default Python runtime, so tensor fixtures are
skipped there. No weights were loaded, no native/instrumented model comparison
was run, and no trusted model path or causal-scrubbing result is established.
