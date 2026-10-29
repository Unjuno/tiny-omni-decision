# Frozen-feature Path A baseline and LSTM result reconciliation

**Reconciled 2026-10-29 from existing 2026-10-16 run artifacts.** This
artifact-level supplement adds selected per-modality metrics and verifies local
receipts referenced by the [2026-10-28 run reconciliation](R2_PATH_A_NATIVE_BASELINE_RECONCILIATION_20261028.md)
and original [temporal-fit report](R2_NATIVE_TEMPORAL_FIT_RESULTS_20261016.md).
No model, media, validation prediction, or sealed-audit evaluation was rerun.

## What was verified

The local `C:\CodexArtifacts\pretrained-reuse-r2` receipts contain a complete
seed-17, 2,048-update fit for the shared frozen-feature Path A Decision readout
and a second fit with a zero-output residual LSTM on speech/video tokens. Both
used the same source commit `72c1eb47dde0c81e503d3a4d783aa5528660186e`, training
config SHA-256
`45954a7240ee8da32e4b5b5dc3283994c40c4849451f27ff4fce13d3cc118f75`, frozen
feature corpus SHA-256
`47a6d2f3495810bac58f57dec1a962f8b490fa409c75671b46bb2999fde87d49`, and query
cache SHA-256
`7f5ffd7362e84770b864a189d50bc29f43be3738f54a07f6dbdc21f1ef7d909e`.

| Receipt | SHA-256 |
|---|---|
| Baseline `result.json` | `aaa1fe96a4be85a71ce146981c6b2e8b58e1308d91fc110e1ece6804b4d8048c` |
| LSTM `result.json` | `47dde479c91ed82cdbf260686f0184ebdf95b53981b28248e0ad1ebf2c6b10c6` |
| Shared validation ID list | `db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30` |
| Shared sampler receipt | `b2647f3a1783fe5909ab4864281522314b694ef9a33851a35cc11b6beabe41fb` |
| Paired comparison `comparison.json` | `7ceb98a7bb9aa9b2a651a0e6903386167ff3bca16109a133a01bb0b6c1469c46` |

The comparison receipt reports identical initial validation logits, identical
source pins, identical sampling and validation IDs, and exact fresh-process
prediction replays for both runs. The paired-comparison audit status is
`PASS_PAIRED_COMPARISON_AUDIT`; its CPU-only focused tests pass (**3 passed**).
The trained weights and large prediction arrays remain outside Git.

Both runs consumed 8,192 microbatches (four modalities balanced at 2,048 each):
6,656 unique questions, 1,536 repeated questions, 5,107 unique observations,
3,114 unique media assets, and 3,382 groups. The validation set contained
24,346 questions, including 200 CLEVRER scenes / 400 video questions. No
encoder weights were updated, and sealed audit was not loaded.

| Variant | Trainable / total parameters | Best / final step | Train / eval seconds | Peak CUDA / process working set |
|---|---:|---:|---:|---:|
| Shared readout baseline | 889,345 / 196,254,874 | 1,792 / 2,048 | 61.4 / 476.1 | 208,541,696 / 3,221,950,464 bytes |
| Residual LSTM | 1,925,761 / 197,291,290 | 1,536 / 2,048 | 100.0 / 943.7 | 275,454,464 / 3,478,687,744 bytes |

Both full parameter counts are below the provisional 200M parameter ceiling.
This is not a packaged-size or device-latency result. Training used FP32 on the
RTX 3080 Laptop GPU with CUDA 12.4. The same hierarchical macro-NLL selector
was applied at scheduled validation points; its best step was retained apart
from the final step.

## Validation learning curve

Values below are fixed 24,346-question development validation metrics. Macro
means equal-weight modalities; they are not sample-weighted overall scores.

| Step | Baseline macro Accuracy | Baseline macro NLL | LSTM macro Accuracy | LSTM macro NLL |
|---:|---:|---:|---:|---:|
| 0 | 15.91% | 2.2227 | 15.91% | 2.2227 |
| 512 | 27.35% | 1.8005 | 27.87% | 1.7789 |
| 1,024 | 29.09% | 1.7558 | 28.89% | 1.7659 |
| 1,536 | 29.90% | 1.6960 | 30.19% | 1.6967 |
| 1,792 | 30.71% | 1.6865 | 30.82% | 1.6994 |
| 2,048 | 30.58% | 1.6910 | 30.71% | 1.7071 |

The baseline is selected by the fixed validation rule (macro NLL 1.6865 versus
1.6967 for the LSTM). The LSTM's selected Video Accuracy is higher, but the
selected Image Accuracy is lower. Both candidates remain far from the target;
their zero minimum task accuracies and per-task weaknesses must not be hidden
by the overall aggregate. This supplements the previous run reconciliation;
it does not newly close or repeat the fit work package.

## Selected-checkpoint metrics

Each cell lists **Accuracy / NLL / Brier / ECE**. These are the selected
validation checkpoints, not sealed-audit results.

| Variant (step) | Overall | Macro | Text | Image | Audio | Video |
|---|---|---|---|---|---|---|
| Baseline (1,792) | 36.31% / 1.5769 / 0.6934 / 0.0578 | 30.71% / 1.6865 / 0.7402 / 0.0842 | 20.89% / 1.9211 / 0.8237 / 0.0742 | 20.12% / 2.1651 / 0.8652 / 0.0311 | 54.28% / 1.1467 / 0.5364 / 0.0707 | 31.50% / 1.3393 / 0.7006 / 0.0842 |
| Residual LSTM (1,536) | 35.57% / 1.5795 / 0.6940 / 0.0572 | 30.19% / 1.6967 / 0.7423 / 0.0750 | 21.16% / 1.9458 / 0.8289 / 0.0786 | 14.70% / 2.2213 / 0.8794 / 0.0463 | 53.25% / 1.1186 / 0.5303 / 0.0539 | 35.75% / 1.3339 / 0.6970 / 0.0522 |

At fixed step 2,048, macro NLL was 1.6910 for baseline and 1.7071 for LSTM.
Video Accuracy was 31.50% and 33.25%, respectively; Image Accuracy was 19.73%
and 17.82%. The LSTM did not establish a broad multimodal gain.

## Paired Video uncertainty and interpretation

The saved audit used 5,000 paired bootstrap resamples of underlying CLEVRER
scene groups (200 groups; seed 17; percentile 95% intervals). For the selected
checkpoints, LSTM-minus-baseline Video Accuracy was +4.25 percentage points
(95% CI -0.25 to +8.75); the temporal-descriptive subset was +2.50 points
(-2.50 to +7.50). Both intervals include zero. At the fixed final step, the
all-video accuracy difference was +1.75 points (-1.50 to +5.00), while NLL
and Brier differences favored LSTM by -0.00697 (CI -0.01332 to -0.00069) and
-0.00407 (CI -0.00722 to -0.00096). On the 200 temporal-descriptive questions,
final accuracy difference was 0.00 points (-3.00 to +3.00); NLL and Brier
intervals included zero. These are development-validation diagnostics after
checkpoint selection, not independent evidence of generalization.

This comparison does not isolate recurrent memory: the LSTM variant has
1,036,416 more trainable parameters, only one training seed was used, and the
full-clip video feature may contain future information. No parameter-matched
nonrecurrent control or temporal-order intervention was part of this fit.
Therefore the data do not justify adopting the LSTM or claiming that recurrence
caused the Video difference. The already observed failure to improve selected
macro NLL, along with the Image trade-off, is sufficient to avoid repeating
this same fit or enlarging its budget without new data or a distinct hypothesis.

The next quality gate remains a task-matched, source-grouped real joint-input
Decision evaluation. RoboHEARD's official development release is scheduled for
2026-10-30, but access requires registration and email delivery, its terms are
CC BY-NC-SA 4.0, and its stated AVSR metric is not yet the project's supplied-
option calibrated Decision contract. Do not register or accept terms on the
user's behalf; inspect public release metadata first and keep it review-only
until rights, labels, split identities, both-input necessity, storage, and
task fit are verified.
