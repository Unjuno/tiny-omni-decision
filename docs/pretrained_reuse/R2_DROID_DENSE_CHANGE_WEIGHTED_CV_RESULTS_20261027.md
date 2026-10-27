# Dense DROID windows: change-weighted train-fold follow-up

The [fixed follow-up protocol](R2_DROID_DENSE_CHANGE_WEIGHTED_CV_PROTOCOL_20261027.md)
used all 80,998 existing train windows, the same five recording-group folds,
two model structures, optimizer settings, seeds and 50 epochs as the previous
natural-loss run. For each fold, inverse-frequency loss weights were computed
from its training rows only: stable weights ranged 0.549–0.552 and changed
weights 5.308–5.594. No held window was filtered or reweighted for reporting.

The first execution computed an unused weight array for held rows as well as
train rows. Those held weights were never used for loss, but this exceeded the
strict isolation intended by the procedure. Its artifacts are retained in
`droid-dense-change-weighted-cv-20261027-attempt-01/`. The implementation was
corrected and the **same fixed settings and seeds** rerun in attempt 02; all
OOF probability arrays are exactly identical between attempts. The results
below are from attempt 02, whose training weights are populated only for
training indices; a regression test changes held targets and confirms no effect
on those weights.

## Natural-prevalence results

| History + sensor prediction | Accuracy | NLL | Macro NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|
| Previous natural-loss run | 0.90722 | 0.44993 | 0.49308 | 0.17533 | 0.02178 |
| Change-weighted raw probabilities | 0.84734 | 0.82747 | 0.88677 | 0.38132 | 0.30934 |
| Fixed weight-corrected probabilities | 0.90696 | 0.44286 | 0.48758 | 0.17430 | 0.01779 |

| History + sensor prediction | Change detection recall | Change detection precision | Stable false-change rate | Exact new command on changed rows |
|---|---:|---:|---:|---:|
| Previous natural-loss run | 22 / 7,442 (0.30%) | 22.22% | 0.105% | 4 / 7,442 |
| Change-weighted raw probabilities | 1,081 / 7,442 (14.53%) | 16.76% | 7.30% | 444 / 7,442 (5.97%) |
| Fixed weight-corrected probabilities | 33 / 7,442 (0.44%) | 24.44% | 0.139% | 8 / 7,442 |

Change detection means the argmax differs from the previous command; it need
not be the correct new command. The weighted history-only head detected no
changes, with or without correction. The raw sensor head detects more changes
but has many false alarms and much worse natural-prevalence calibration. The
fixed correction improves its natural-prevalence NLL and ECE in this cohort,
while sacrificing nearly all change-detection recall. The correction is an
algebraic inverse of the training weights, not proof of calibrated deployment
probabilities. These trade-offs are not a decision to adopt either operating
point, and no threshold was tuned.

This follow-up was designed after earlier results were inspected and uses the
same train recordings and correlated windows. It is neither blind evaluation
nor independent-site evidence. Dominant-axis changes may be small fluctuations;
the task does not establish audio/video fusion or the product's general
95–98% target. Training losses were still falling at epoch 50; held learning
curves were not collected, so a plateau or absence of overfitting is not shown.
The two heads have 56 versus 840 trainable parameters. No GPU, validation,
sealed audit or final product checkpoint was used. Attempt 02 CPU execution
took 20.50 seconds.

An independent replay verified all ten saved heads, training-only sensor
normalizers and fold weights. Maximum absolute difference from saved PyTorch
probabilities was `7.16e-7` for raw and `4.63e-7` for corrected NumPy replay.

Private attempt 02 artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-dense-change-weighted-cv-20261027-attempt-02/`.
Raw SHA-256 of `summary.json`:
`53fe441c36b15bbb6b3b4f9570775adee62373f365c99278c314c819896c3359`;
`oof-probabilities.npz`:
`20f08c39df29716894501dd5c7f7081d9d09177a5b8c2a834093d9fe30739c85`.
The summary records per-class results, all fold weights, training curves,
source/protocol/code hashes and hashes of ten heads.
