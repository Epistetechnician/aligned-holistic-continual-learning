# Matched-Progress Diagnosis

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

## Recorded result

The README reports that the separate matched-progress comparison failed. It
describes that comparison as testing whether lower harm is only a side effect
of learning less. `RESEARCH.md` does not provide a matched-progress result
table, and this checkout contains no `results/` artifacts. Therefore, the
matched-progress failure can be recorded here only as a qualitative outcome;
no matched gain, harm, effect size, uncertainty, or per-arm value is available
to reproduce from the committed files.

The nearby synthetic qualification is a different result. Its recorded
assessment values are R² 0.830-0.900 and sign agreement 0.938-0.979; shuffled
controls range from -0.27 to -0.51 R², and tune-split R² is reported as
0.82-0.91. Those numbers do not quantify the matched-progress failure.

## Testable hypotheses

These are hypotheses for the missing matched-progress analysis, not findings.

1. **Unequal learning progress explains the apparent harm difference.** The
   arms may have been compared at different incoming-task gains despite the
   intended matched-progress design. Test this with arm-by-arm incoming-task
   gain, protected-task loss/harm, accepted-update count, and the exact matching
   grid, including uncertainty for each matched contrast.
2. **The harm endpoint changes across progress levels.** A controller may look
   safer at one gain level but not another. Test this with checkpoint-level
   gain and protected-task loss trajectories for every arm, plus matched
   contrasts at each preregistered gain level rather than one aggregate.
3. **The failure is driven by eligibility or resource differences.** Arms that
   stop early, fail the positive learning floor, or incur different monitoring
   costs may not be comparable. Test this with terminal dispositions, accepted
   updates, completed-stream status, actual compute/time/memory ledgers, and
   the same eligibility rules for every arm.

No hypothesis is selected as the explanation without those observations.
