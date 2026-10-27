# Nested set-decision threshold results

The [fixed protocol](R2_SOURCE_SET_NESTED_THRESHOLD_PROTOCOL_20261027.md)
trained six fresh inner-fold readouts, excluding each outer video's group from
the two models and their threshold selection. The selected threshold pair
was saved before that outer fold's existing joint prediction file was read.
The three membership/none threshold pairs were 0.50/0.35, 0.65/0.35 and
0.50/0.35. The latter thresholds are fixed decisions, not calibrated scores.

| True target stratum | N | Fixed 0.5 thresholds exact | Inner-selected thresholds exact | Difference |
|---|---:|---:|---:|---:|
| All | 2,593 | 23.99% | 25.57% | +1.58 points |
| None visible | 692 | 52.89% | 61.13% | +8.24 points |
| One target | 799 | 23.53% | 23.65% | +0.13 points |
| Multiple targets | 1,102 | 6.17% | 4.63% | -1.54 points |

The paired 320-video bootstrap interval for overall difference was +0.78 to
+2.39 points; for multiple targets it was -2.62 to -0.67 points. Intervals
use 10,000 video-group draws, seed 17. The none improvement and multi-target
loss are both material; selecting by macro stratum accuracy did not prevent
that tradeoff. The previous single-choice baseline on the same development
records was 28.50% exact, so this candidate remains below it overall and is
not a product upgrade. No seed or threshold was selected using the outer fold.

This is repeated work on previously observed train-source videos, with oracle
tracks and possible contributor overlap. The inner readouts have roughly half
the outer-training data, so transferring thresholds to the larger-trained
outer readout is an approximation. The bootstrap measures video-group sampling
within these folds, not independent source or seed uncertainty. No sealed audit
was opened. Scores were not recalibrated and NLL/Brier/ECE are unchanged by
this decision-threshold experiment.

CPU execution took 17.61 seconds; CUDA was not initialized. The six checkpoint
hashes and strict weight loads passed; all three saved inner selection grids
and selected pairs were reproduced from hash-checked inner predictions. Full
prediction replay from weights was not done. The outer 0.5 decoder was replayed
against the existing saved predictions. No old artifact was changed.

Run receipt: `C:/CodexArtifacts/pretrained-reuse-r2/perception-source-set-nested-threshold-20261027-attempt-01/run.json`,
SHA-256 `c36be968ca15dd51375ebbf8c08fade7a50b607e02cdb366e8f4760d7412f545`.
