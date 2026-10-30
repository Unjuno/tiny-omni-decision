# EgoViz fixed-budget GRU temporal readout (2026-10-30)

## Result

The fixed-seed CPU run completed all 5,632 updates from fresh GRU-head weights.
It used the same frozen features, 773 train examples / 48 sessions, and 307
development examples / 20 sessions as the linear coverage comparison. The
predeclared fixed updates were 352, 704, 1,408, 2,816, 4,224 and 5,632; no
checkpoint was selected from development performance. The sealed audit was not
accessed.

The recurrent head did not improve the combined quality measures. At update
5,632, Accuracy was 57.33% versus 55.05% for the matched linear reference,
but the paired session-bootstrap interval for the +2.28 percentage-point
difference was [-4.32, +9.09], including zero. GRU NLL was 4.2479 versus
1.2001; paired difference +3.0477, 95% interval [+2.3010, +3.7460]. Brier
was 0.7547 versus 0.6526 and ECE was 0.3744 versus 0.2091. Mean confidence
reached 94.77% despite only 57.33% Accuracy.

| Fixed update | GRU train Accuracy | GRU dev Accuracy | GRU dev NLL | GRU Brier | GRU ECE | Linear dev Accuracy | Linear dev NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 352 | 79.56% | 53.75% | 1.6714 | 0.6996 | 0.3109 | 56.68% | 1.1790 |
| 704 | 98.71% | 54.72% | 2.3853 | 0.7622 | 0.3619 | 57.65% | 1.1783 |
| 1,408 | 100.00% | 54.07% | 3.1337 | 0.7927 | 0.3925 | 56.68% | 1.1808 |
| 2,816 | 100.00% | 58.31% | 3.8835 | 0.7551 | 0.3618 | 55.37% | 1.1901 |
| 4,224 | 100.00% | 57.33% | 4.0746 | 0.7564 | 0.3742 | 55.05% | 1.1976 |
| 5,632 | 100.00% | 57.33% | 4.2479 | 0.7547 | 0.3744 | 55.05% | 1.2001 |

The GRU training NLL reached below 0.04 by update 704 and effectively zero
thereafter, while development NLL rose at every recorded checkpoint. The
single accuracy increase at 2,816 did not persist as an NLL or calibration
improvement. This is direct overfitting for this readout and task. It does not
show that recurrent models in general cannot help, and it does not establish a
generalization limit for pretrained video or sensor representations.

**Disposition:** The experiment does not support replacing the 81,920-parameter
linear readout with this GRU on the current EgoViz timestamp-matching task.
Further selection on the repeatedly observed 20-session development set is
not justified. The result remains a task-specific exploratory comparison, not
a blind product evaluation or four-modality result.

## Frozen experiment and resources

- Pre-run source commit: `ac6aae443a89b2aa8b8c18c4b85a6480f1b0f722`.
- Seed 17; fresh 123,521-parameter readout: independent 768→64 video and
  512→64 sensor projections, four-way pair features, shared 64D projection,
  unidirectional GRU(64), and shared scalar score.
- Inputs: frozen 8×768 VideoPrism tokens; frozen MOMENT six-channel,
  16×512 sensor features, mean-pooled over channels and paired into eight
  temporal positions. No pretrained encoder weights were loaded or updated.
- AdamW, learning rate 0.003, weight decay 0.01, batch size 32, four CPU
  threads, cross-entropy. No scheduler, dropout, calibration, early stopping,
  extra seed, or task/data changes.
- 5,632 updates; 174,149 sample exposures (225 full passes plus seven batches
  of a partial pass); all 773 unique training examples consumed.
- Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6; Intel i7-12700H CPU.
  Training and scheduled evaluation took 115.55 seconds. Peak RAM was not
  captured. No GPU or cloud compute was used; the separate RTX 3080 remained
  occupied by another process and was not touched.
- C: free space was about 5.30 GiB after the run. The GRU checkpoint is
  498,450 bytes. No existing data, run, or checkpoint was deleted or changed.

## Reproducibility and integrity

Development input and target IDs match the frozen linear comparison exactly.
All six checkpoint and prediction hashes were checked in a fresh process;
reloaded logits differed from saved logits by at most `3.82e-6`. Option
permutation equivariance and sample/group/target identity checks passed at all
six checkpoints. The frozen input hashes and run/configuration details are in
the [machine-readable result](../../manifests/pretrained_reuse/egoviz-gru-temporal-results-20261030.json).

- Runner SHA-256: `a026a5eea67e80e07b952d68ebd5f809ba88105ae21dd856cc98a5813b07c943`.
- Merged-corpus report SHA-256: `e8a451de798f468af65dddb037e12ef132a45062df3dc4f882aa2addb5550635`.
- Linear reference results SHA-256: `fc26c040e61618d64812793fef4fbc10863c33b89174ea24f05e77ab31ac1a9a`.
- GRU results SHA-256: `5f64a15d0b963c6e041109a1ce1c0012d61c4fe26fbaa92bedb6f8f9265ac95c`.
- Reload verification SHA-256: `c381e25cef28171f53787bd3388290ae04c1c52bc524ac09e860aa8d3e2e3b1e`.
- Final GRU state SHA-256: `ee54065f5161bfb1ead7356ad9d838d2f28cc1d4294597087986ff3d8d8505ec`.
- Final development predictions SHA-256: `f8827057fc7b311b57d0bfa658228b3a7e292160f8e8f4f83a80d81beaeae912`.
- External artifact directory: `C:\CodexArtifacts\egoviz-gru-temporal-budget-20261030-attempt-01`.

The run is derived from the already-observed development set and its selected
linear reference. The intervals are paired by 20 publisher session groups,
2,000 bootstrap draws, seed 17, percentile 95%; they do not correct for prior
checkpoint/architecture selection or unknown participant overlap.
