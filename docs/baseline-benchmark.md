# Baseline benchmark contract

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

Status: executable model-free accounting and eligibility checks; no empirical
benchmark, training, or assessment has run under this workstream.

`src/ahcl/baseline_contract.py` defines five mandatory comparison arms:
ordinary SGD, bounded reservoir replay, diagonal-Fisher EWC, behavioral
candidate-update selection, and causal-monitor selection. The selection arms
must share candidate generation, permitted behavioral probes and decision
opportunities. Their scientific difference is access to causal features.

Freeze the optimizer, step size, reservoir capacity and sampling algorithm,
replay ratio, Fisher estimator/sample count/refresh schedule, EWC coefficient,
candidate roster, probe schedule and controller thresholds using fit/tune only.
All arms start from the identical checkpoint and see the identical ordered fresh
stream, tokenizer, batch schedule and seeds. Replay draws only from previously
seen permitted training examples. Assessment examples never enter gradients,
replay, Fisher estimation, feature recalibration or controller decisions.

Every arm receives identical positive hard caps for compute operations, elapsed
time and peak live memory. Charge setup, evaluation, feature measurement,
candidate updates (including rejected ones), rollback, replay and Fisher work.
Peak memory includes concurrently resident checkpoints, optimizer, reservoir,
Fisher tensors, feature dictionary and instrumentation. Memory is a peak, not a
sum across sequential events. Report actual consumption as well as caps;
identical caps do not assert identical actual costs. Hardware, operation-count
convention and device synchronization must be fixed in the execution packet.

`ResourceLedger.preflight` rejects conservative projected costs before work;
`charge` records actual measured costs. Any exceeded bound closes that arm;
there is no reset/refund for rejected candidates. These checks do not enforce
hardware limits themselves. A reviewed runner must reserve conservative bounds
and enforce process/device limits before model execution opens. Fit/tune search
costs and final evaluation costs must also be disclosed and budgeted.

`validate_splits` checks family, document and author identity disjointness.
The execution packet must additionally verify content overlap and custody,
bind all source/input/runtime hashes and lock predictions before assessment.
Document identity alone cannot prove absence of pretrained contamination.

Retention is eligible for comparison only when the arm completes the locked
stream within caps and clears a positive preregistered lower-confidence-bound
floor on new-task gain against its frozen initial checkpoint. Zero accepted
updates always fails. The interval estimator and numerical floor remain review
inputs; this contract does not choose a convenient threshold after effects.
Report learning, retention and resources jointly rather than pooling them into
an adjustable score. Failed gates, missing observations, unqualified measurement,
leakage and resource exhaustion are terminal dispositions, not excluded samples.

Deliverable before execution: independently reviewed numerical budgets,
hyperparameters, endpoints, interval/power/multiplicity methods, lock identity,
and runner wiring to this ledger. Deliverable after authorized execution:
aggregate arm comparisons with uncertainty and all failed dispositions. Until
then, the contract establishes accounting behavior only.
