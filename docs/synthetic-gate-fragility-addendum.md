# Synthetic gate fragility addendum (probe, not a result)

Status: exploratory probe produced by an autonomous open-model research loop and independently re-run by the owner's assistant on pristine code (no source, threshold or packet changes; the loop's own branch edited a threshold and a learning rate, and those runs were discarded). It changes no gate, protocol, or claim ceiling and authorizes no model execution. A named state slice for this addition has not been assigned by the owner; this lives on an unmerged local branch.

Method: `PYTHONPATH=src python3 scripts/synthetic_gate_sweep.py` regenerates the transparent synthetic stream under varied seed labels, noise scales and family-roster sizes and re-evaluates the locked gates from `ahcl.experiment.CONFIG`. Caveat: the script reimplements the stream generation, so it is a probe of gate sensitivity, not a reproduction of `ahcl.run` output.

Observed (32 seed-relabelled runs at 16 families):
- Gate pass rate 0.406; weakest margin -0.30 (seed-17).
- Noise scale barely matters: pass rate stays 0.50-0.56 from noise 0.0 to 5.0.
- Roster size matters a lot: 0% pass at 2-10 families per split, 18.8% at 12, 12.5% at 14, 56% at 16. At 4-8 families the binding gate is `r2`; at 12-16 it is `sign_agreement`.

Reading: the locked synthetic gates pass for the recorded configuration but are sensitive to roster size and seed labelling, not noise. Any claim built on the synthetic pass should carry the roster-size dependence. No threshold was or should be changed on this basis.

## Joint pass probability versus roster size (extended probe)

`python3 scripts/synthetic_joint_power.py 48` (48 re-seeded runs per roster size, noise scale 1.0, locked thresholds from `ahcl.experiment.CONFIG`). "Joint" means every gate on every axis passes in the same run. Same caveat as above: the probe reimplements the stream generator.

| families per split | joint pass | Wilson 95% | r2 | r2 over baseline | sign agreement | shuffled control |
|---:|---:|---|---:|---:|---:|---:|
| 12 | 0.083 | 0.033-0.196 | 0.50 | 0.54 | 0.10 | 0.83 |
| 16 | 0.313 | 0.199-0.453 | 0.79 | 0.92 | 0.40 | 0.92 |
| 20 | 0.667 | 0.525-0.783 | 1.00 | 1.00 | 0.71 | 0.96 |
| 24 | 0.750 | 0.612-0.851 | 1.00 | 1.00 | 0.85 | 0.88 |
| 32 | 0.958 | 0.860-0.988 | 1.00 | 1.00 | 0.96 | 1.00 |
| 48 | 1.000 | 0.926-1.000 | 1.00 | 1.00 | 1.00 | 1.00 |
| 64 | 1.000 | 0.926-1.000 | 1.00 | 1.00 | 1.00 | 1.00 |

Reading: joint pass probability rises steeply between 16 and 32 families, and `sign_agreement` is the binding gate at every size below 48. Under this generator, at least 32 families per split gives a joint pass probability of about 0.96 (lower 95% bound 0.86); 48 gives a lower bound of 0.93. This is the empirical joint power that the model-backed packet's open review item ("Holm joint power unresolved") asks for, under synthetic assumptions only; it does not substitute for the packet's own power analysis, and it changes no gate, threshold or claim ceiling.
