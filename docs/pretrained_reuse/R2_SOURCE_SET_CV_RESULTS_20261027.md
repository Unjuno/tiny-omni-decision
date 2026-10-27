# Source set-decision comparison: fixed seven-epoch results

The [fixed protocol](R2_SOURCE_SET_CV_PROTOCOL_20261027.md) ran all nine
independent readouts: three modality modes on each of three video-group folds,
seed 17 and seven epochs. Each model has 344,834 trainable parameters. The
same frozen training-source features and 2,593 records from 320 video groups
were used. These are previously observed train-source development folds, with
oracle tracks and unresolved contributor grouping; this is not blind, an
end-to-end product test or a measure of all four modalities.

| Mode | Exact-set accuracy | None decision accuracy | Candidate NLL per record | Candidate Brier per record | None NLL | None Brier |
|---|---:|---:|---:|---:|---:|---:|
| Audio | 0.0779 | 0.7782 | 0.3063 | 0.0843 | 0.5151 | 0.1672 |
| Video | 0.2383 | 0.8207 | 0.2263 | 0.0674 | 0.4517 | 0.1347 |
| Audio + video | 0.2399 | 0.8326 | 0.2283 | 0.0682 | 0.4378 | 0.1290 |

The joint-minus-video exact-set accuracy difference is +0.15 percentage
points, paired 320-video percentile bootstrap interval -0.85 to +1.16 points
(10,000 draws, seed 17). Candidate micro ECE was 0.0097 / 0.0154 / 0.0152
for audio / video / joint; none-event ECE was 0.0279 / 0.0588 / 0.0654.
Candidate micro scores weight records with more candidate objects more heavily;
the table averages per record. None scores and candidate marginals are separate
binary events, not a normalized distribution over entire sets. The audio-only
model has tied candidate scores and is not an independent visual identifier.
The small ECE values are not evidence of calibration on independent sources.

On the same records and folds, the old single-choice joint baseline at epoch 7
had 739/2,593 (28.50%) exact sets versus 622/2,593 (23.99%) for this new
set-valued joint model. New minus old was -4.51 points with a paired video
interval -6.25 to -2.84 points. This exploratory comparison changes both loss
and decision rule; it does not establish that multiple outputs cannot help.
The new model can predict multiple targets, unlike the old model, but this
particular fixed rule and seven-epoch schedule did not improve exact accuracy.
No model selection, seed search, threshold tuning or sealed audit was done.

CPU training took 43.80 seconds; CUDA was not initialized. Nine checkpoint
hashes were verified and weights strictly loaded; full prediction replay from
weights has not been performed. Prediction and target IDs were hash-checked
and paired for the comparisons. The run produced no claims of product readiness.

Run receipt: `C:/CodexArtifacts/pretrained-reuse-r2/perception-source-set-cv-20261027-attempt-01/run.json`,
SHA-256 `5df64f799220dbe95c1234556266c92e4aff332c524bc79c048e1a485fe2218b`.
Summary: `C:/CodexArtifacts/pretrained-reuse-r2/perception-source-set-cv-summary-20261027-attempt-01/result.json`,
SHA-256 `73e0cf163fd9c0a58e26c2b454b6dc24ea19c29989fced546ce005bba28cb098`.
Paired old/new comparison: `C:/CodexArtifacts/pretrained-reuse-r2/perception-source-set-vs-single-20261027-attempt-01/result.json`,
SHA-256 `209fe990cd0ca85f5ec443ef63adf76c38aa9cb74bfd7d2e603dbe538a5e8523`.
