# Synthetic gate fragility addendum (probe, not a result)

Status: exploratory probe produced by an autonomous open-model research loop and independently re-run by the owner's assistant on pristine code (no source, threshold or packet changes; the loop's own branch edited a threshold and a learning rate, and those runs were discarded). It changes no gate, protocol, or claim ceiling and authorizes no model execution. A named state slice for this addition has not been assigned by the owner; this lives on an unmerged local branch.

Method: `PYTHONPATH=src python3 scripts/synthetic_gate_sweep.py` regenerates the transparent synthetic stream under varied seed labels, noise scales and family-roster sizes and re-evaluates the locked gates from `ahcl.experiment.CONFIG`. Caveat: the script reimplements the stream generation, so it is a probe of gate sensitivity, not a reproduction of `ahcl.run` output.

Observed (32 seed-relabelled runs at 16 families):
- Gate pass rate 0.406; weakest margin -0.30 (seed-17).
- Noise scale barely matters: pass rate stays 0.50-0.56 from noise 0.0 to 5.0.
- Roster size matters a lot: 0% pass at 2-10 families per split, 18.8% at 12, 12.5% at 14, 56% at 16. At 4-8 families the binding gate is `r2`; at 12-16 it is `sign_agreement`.

Reading: the locked synthetic gates pass for the recorded configuration but are sensitive to roster size and seed labelling, not noise. Any claim built on the synthetic pass should carry the roster-size dependence. No threshold was or should be changed on this basis.
