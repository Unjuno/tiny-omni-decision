# STARSS23 frozen-feature comparison: measured negative joint result

All three preregistered readouts completed on CPU from source commit `8e2dda7`.
Configuration and split hashes were unchanged. Both frozen feature corpora passed
the input provenance checks; best-checkpoint reload and option-order checks passed.
No retry, new seed, resampling or hyperparameter adjustment followed these scores.

## Validation results

486 five-second windows, 5,832 binary queries, 12 events, two held-out development
rooms. Metrics below are macro-event averages after calibration on the separate
calibration room. Brier sums squared error over both options; ECE uses 15 bins.

| Condition | Best epoch | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Audio only | 10 | .89969 | .27039 | .15136 | .06458 |
| Video only | 6 | .83968 | .41080 | .24535 | .11684 |
| Audio + video | 6 | .86180 | .32901 | .20135 | .11208 |
| Training prevalence prior | — | .85717 | .37398 | .22481 | .09228 |
| Joint, audio zeroed | — | .85219 | .33207 | .20986 | .09309 |
| Joint, video zeroed | — | .87106 | .29779 | .18444 | .07190 |
| Joint, different-recording video | — | .85614 | .33102 | .20306 | .10346 |

Video did not improve the strongest unimodal baseline. The joint readout is
worse than audio-only on all four metrics; removing its video input improves
Accuracy, NLL and Brier. Mismatching video changes NLL only slightly; the paired
room bootstrap interval for that NLL difference crosses zero. This experiment
does not establish useful synchronized-video contribution. Do not promote this
joint model or pursue its compression on the strength of these results.

High binary Accuracy is partly explained by negative-event prevalence: the
training prior alone reaches 85.7%. Per-event balanced Accuracy, positive/negative
counts, sensitivity, specificity, raw/calibrated metrics and every epoch's NLL
are retained in the adjacent JSON. Bell has no positive training examples.
These are development estimates with validation checkpoint selection, not a
blind audit. The 10,000-replicate paired bootstrap uses rooms, seed 17 and
percentile 95% intervals; only two validation rooms make those intervals severely
low-power. No general model-family or multimodal impossibility conclusion follows.

## Execution and artifacts

Video feature extraction: 2,720 x 768, 3,920.33 seconds CPU wall time,
4,251,328,512 bytes peak working set. Audio feature extraction is documented in
`R2_STARSS23_CORPUS_20261011.md`. All readout training/evaluation/bootstrap took
28.67 seconds on CPU. No paid compute was used.

```powershell
python -m scripts.train_starss23_readouts --manifest-dir C:/CodexArtifacts/pretrained-reuse-r2/starss23-all-windows-v1 --audio-features C:/CodexArtifacts/pretrained-reuse-r2/starss23-audio-corpus-features-v1 --video-features C:/CodexArtifacts/pretrained-reuse-r2/starss23-video-corpus-features-v1 --config configs/pretrained_reuse/path_a_starss23_events.yaml --output-dir C:/CodexArtifacts/pretrained-reuse-r2/starss23-readout-comparison-v1
```

That output directory contains separate best/final checkpoints, normalization,
effective config, input hashes, validation IDs, per-window raw/calibrated
predictions, intervention predictions and permutation indices. It must not be
overwritten. The compact Git report includes histories and artifact hashes;
media, features and weights remain outside Git. Legacy Teacher/QAT artifacts
and sealed evaluations were untouched.

The next decision must distinguish task suitability, room shift and loss of
spatial/temporal information in pooled panorama features. These are hypotheses,
not measured causes. Repeated tuning against these same two rooms would not
establish generalization; a further experiment needs a separately fixed protocol.

## Event-level diagnostic from saved predictions

No retraining or new model selection was performed.

| Event | Positive windows | Audio balanced Accuracy | Joint balanced Accuracy | Audio NLL | Joint NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| speech | 311 | 0.6691 | 0.7339 | 0.4349 | 0.6024 |
| clapping | 10 | 0.4989 | 0.5000 | 0.1128 | 0.1367 |
| telephone | 19 | 0.4979 | 0.5000 | 0.1511 | 0.1675 |
| laughter | 56 | 0.6926 | 0.5016 | 0.3513 | 0.4185 |
| domestic sounds | 267 | 0.5918 | 0.4954 | 0.8667 | 1.0063 |
| footsteps | 19 | 0.5092 | 0.5000 | 0.2809 | 0.3051 |
| door | 17 | 0.5535 | 0.5000 | 0.1536 | 0.1818 |
| music | 229 | 0.8857 | 0.5352 | 0.3052 | 0.6397 |
| instrument | 5 | 0.4979 | 0.5000 | 0.0844 | 0.1126 |
| water tap | 0 | undefined | undefined | 0.0543 | 0.0861 |
| bell | 32 | 0.5000 | 0.5000 | 0.4058 | 0.2028 |
| knock | 4 | 0.6240 | 0.5000 | 0.0436 | 0.0886 |

Largest joint-minus-audio NLL degradations: music (+0.3345), speech (+0.1675),
domestic sounds (+0.1396). Speech balanced Accuracy improves while NLL worsens.
Music balanced Accuracy falls from .8857 to .5352. Bell NLL improves despite
zero positive training examples, while balanced Accuracy stays .5; this is
not learned bell discrimination. Water tap has no validation positives.

Each head has 74,753 parameters and a 300,792-byte best checkpoint. Fixed train
versus validation calibrated NLL: audio .0871/.2704, video .1886/.4108, joint
.1798/.3290. These gaps show a generalization problem without identifying its
cause. This post-hoc diagnostic does not alter the preregistered protocol.

## Independent-process reload check

A separate CPU process reloaded all three best checkpoints, saved normalization,
calibration temperatures and complete frozen feature matrices. It verified
checkpoint/feature/manifest hashes, validation IDs and targets, and reproduced
all 5,832 validation queries per condition: maximum absolute difference was
exactly zero for both raw and calibrated probabilities. This is feature-to-score
replay, not a new raw-media decode or blind quality evaluation. The compact
record is `R2_STARSS23_RELOAD_20261011.json`; full local artifact directory is
`C:/CodexArtifacts/pretrained-reuse-r2/starss23-readout-independent-reload-01`.
GitHub checks for commit `8d89617` both completed successfully.
