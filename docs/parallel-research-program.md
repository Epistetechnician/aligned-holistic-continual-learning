# Parallel research program

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

Status: implementation and model-free contract qualification only. This is a
proposal for scientific review, not execution authorization or empirical evidence.

## Task list and bounded work contract

- [x] Establish current baseline: 27 Python tests pass before this work.
- [x] Assign measurement, baseline, and independent technical audit ownership.
- [x] Implement model-free feature, horizon, alternative-input, and repair measurements.
- [x] Integrate measurement fixes and resource-budget contract after focused tests.
- [x] Record independent rejection findings and remaining execution prerequisites.
- [x] Run combined Python/Rust checks; preserve concurrent commits and stage only owned fixes.

The mechanical target is passing adversarial contract tests for all implemented
measurements, without regressing the baseline. Budget: one implementation pass
per workstream, one integration pass, and focused repairs for observed failures.
No model calls, training, downloads, provider spend, or holdout access occur in
this pass. Existing scientific artifacts and third-party sources are read-only.

## Shared experimental contract

Use one model/runtime/feature identity and one freshly constructed, digest-bound
family roster. Existing document intake is a candidate input, not proof of
pretraining novelty or a completed prompt-family split. Construct families and
deduplicate contained passages before review. Split at the source-document level;
48 families per split do not imply 48 independent documents. Power and resampling
must account for document clusters, family pairing, repeats, and multiplicity.

Every arm starts from the same base and sees the same ordered stream. Account
for candidate probes, rejected updates, replay, Fisher estimation, feature
extraction, dictionary recalibration, evaluation, and optimizer storage. Common
hard caps must be numerically set and enforceable before execution; exceeding
any cap terminates that arm. Report actual consumption as well as caps.

Primary comparison: protected-task loss at matched positive incoming-task gain
and within common compute/memory caps. Predeclare the matching grid and paired
contrasts. Report calibration and policy-task behavior separately; a forbidden
token metric alone is not an alignment endpoint. Refusing all updates fails a
positive learning floor. No weighted holistic score may hide a failing endpoint.

Fit trains predictors and feature matching rules. Tune chooses only a finite,
predeclared set of candidates. Freeze all thresholds, coefficients, feature IDs,
stream schedule, caps, and contrasts before assessment. Hold assessment in a
separate evaluator with one-use authorization. Prediction inputs may include
budgeted candidate-probe effects, but never assessment labels or future outcomes.

## Workstreams and falsifiers

1. **Measurement reliability.** Deliver corrected causal-LM alignment, padding
   semantics, parity, repeat, exact replacement, finite reconstruction checks,
   and tiny adversarial fixtures. Stop before model qualification until runtime
   installation, asset schema, module identity, and independent review are valid.
   Fixture success establishes implementation behavior only.
2. **Strong baselines.** Deliver definitions and hard accounting for SGD,
   replay, EWC, behavioral selection, and causal selection. Reject comparisons
   that exclude monitoring overhead or compare at unequal learning progress.
   See `baseline-benchmark.md` for the budget contract.
3. **Feature stability.** At every locked checkpoint, evaluate reconstruction
   and paired causal-effect vectors on the same development anchor families.
   Compare frozen dictionary with recalibration using fit-only activations and
   a fixed recalibration budget/schedule. Freeze matching on fit; prohibit
   assessment rematching. Report unmatched features and reconstruction error,
   not just cosine similarity of surviving features. Stop using a dictionary
   when its preregistered quality gate fails; retain the failure as an outcome.
4. **Long horizon.** Use a fixed schedule A, B, A-recurrence, C-shift,
   B-recurrence, D-conflict, A-delayed-recall, mixed-recovery. Each segment has
   the same fixed checkpoint count and token budget, specified before execution.
   Conflict tasks must have explicit contextual target rules to avoid scoring
   contradictory labels as forgetting. Evaluate protected tasks at every
   checkpoint. Report gain, peak loss, final loss, cumulative positive harm,
   recovery duration, and resource use. Missing checkpoints fail closed.
5. **Alternative predictors.** Use the same predictor family, fit/tune splits,
   target, feature scaling learned on fit, and tuning budget. Compare loss-change,
   gradient-norm, update-norm, and representation-similarity inputs with causal
   features, plus their combination. Lock a constant predictor to the fit mean.
   Predict delayed behavior at a fixed horizon. Reject causal incremental-value
   claims unless the paired held-out contrast beats the strongest cheap baseline
   after multiplicity correction and overhead accounting.
6. **Independent falsification.** Produce line-linked rejection findings,
   independent arithmetic fixtures, and protocol review criteria. Agent technical
   review is not an external signed execution ACCEPT. No operator-supplied
   always-true signature callback may establish reviewer authority.
7. **Selective repair.** From an identically damaged checkpoint, compare targeted
   repair, matched random repair, behavioral repair, full rollback, and sham.
   Match repaired parameter count, gradient steps, data, and total resources.
   Require both recovery of old performance and preservation of new learning.
   Test on disjoint repair-assessment families; selection of damage or targets
   cannot use their outcomes. Stop if no real damage exists or repair trades away
   the new learning beyond the fixed bound.

`src/ahcl/research_tracks.py` implements diagnostic calculations for streams 3–5
and 7. Its tests use hand-computable fixtures. They do not establish semantic
feature stability, long-horizon performance, or causal repair efficacy.

## Ordering and review boundary

Measurement, baseline design, and falsification proceed in parallel now.
Feature stability and alternative prediction require a qualified measurement
path; long-horizon control and repair require a locked predictor or independently
specified repair rule. Each new hypothesis gets a distinct assessment allocation;
parallel teams cannot repeatedly query a common holdout.

Execution remains closed until the exact protocol, installed runtime, source,
custody, statistical plan, resource caps, evaluator, and authority verification
are accepted independently. Unsupported novelty, breakthrough, or general
alignment claims are prohibited. Negative outcomes remain useful deliverables.

## Integration outcome

Three parallel agents delivered measurement fixes, baseline accounting, and
an independent technical rejection report. The remaining four tracks have
explicit protocols and hand-calculable measurement fixtures. No scientific
experiment ran. The combined suite at integration ran 52 tests, with 50 passing
and two PyTorch-dependent tests skipped; PyTorch is absent from this interpreter.

Integration fixes reject duplicate document IDs and malformed hashes, reject
nonfinite Holm inputs, deep-copy configuration/predictor payloads, check those
identities before assessment access, and consume assessment on success or
failure. Durable assessment-attempt custody now provides an atomic external
one-use marker when configured. Review receipts use a canonical Ed25519
message and trusted-key registry; the launch boundary also binds the exact
reviewed packet and configuration. Corpus document hashes bind to custody,
and independent aggregate reproduction validates family/repeat rosters,
fit-only constant baselines, and family-level shuffling. TransformerLens and
SAELens adapters require local, digest-checked inputs and pinned checkout
locations. A conservative document-level power planner records the gap
between single-comparison power and Holm joint power; the current six-document
per-split panel remains a design blocker. The runtime-attestation helper now
binds interpreter, distribution, and imported module bytes. The NNsight and
circuit-tracer adapters enforce local-byte custody. The required Python 3.11.11 environments and model/feature
bytes are still absent on this host, so no launch attestation has been
produced. The locked study exposes a document-level Holm union-bound power
lower bound and rejects a `validated` status when six documents are
insufficient. A fresh execution review is required after those inputs are
available.

Concurrent commits were observed during integration. Existing commits and
unrelated edits were preserved; no external custody or submodule source was
modified by this parallel pass. The review packet must be rebuilt from a stable
final checkout before submission; a stale packet cannot authorize execution.
