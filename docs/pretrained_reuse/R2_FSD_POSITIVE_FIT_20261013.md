# Consumed-positive replay: adaptation occurs, transfer remains weak

Replayed the same128 consumed whole-clip positive queries with three preserved
checkpoints: original baseline best, augmented best(epoch9), augmented final(24).
Verified media selection, feature order/hash, checkpoint hashes and exactly equal
normalization. Used raw probabilities; no fitting, calibration or new selection.

| Known positive query | Baseline best yes /16 | Augmented best yes /16 | Final yes /16 |
| --- | ---: | ---: | ---: |
| Bell | 1 | 13 | 16 |
| Clapping | 0 | 16 | 15 |
| Door | 0 | 15 | 16 |
| Knock | 0 | 16 | 16 |
| Musical instrument | 0 | 14 | 16 |
| Telephone | 0 | 15 | 16 |
| Footsteps | 1 | 13 | 15 |
| Water tap | 0 | 16 | 16 |
| Total known positives | 2/128 | 118/128 | 126/128 |

Positive-only NLL fell strongly: for example Bell5.214→.393→.068. The saved
readout adapts to the consumed examples. Thus complete failure to fit those
positives is not supported. Despite this, STARSS validation Bell recall remains0
at the selected checkpoint and other rare events also transfer poorly.

These are training positives only, with no verified negatives. Do **not** call
118/128 or126/128 overall classification Accuracy or evidence of calibrated
generalization. Always-yes would answer every such query positively. Memorization,
source-specific shortcuts, clip/window duration, recording conditions and
annotation-policy differences remain possible explanations. This probe cannot
isolate them. Increasing epochs already worsened legacy validation NLL; no
additional run or checkpoint replacement follows.

The next useful data test needs unseen source groups with verified positive
**and negative** event labels under one task definition. FSD missing labels
cannot supply those negatives. This requirement follows from the inability of
a positive-only probe to measure false positives, not from selecting easier
validation examples. Preserve the existing benchmarks and artifacts.

Run: `python -m scripts.replay_fsd_positive_fit`.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/fsd-positive-training-fit-01/`.
Includes all128 raw probabilities per checkpoint and per-event positive-only
NLL/mean probability, with hashes. Ruff passes. GitHub Actions for the preceding
training/report commit e10bbbf both succeeded. No audit or model updates.
