# IntPhys Path A 16-Frame Follow-up

**Status:** completed one fixed-seed development comparison; no model or dataset
was promoted. This is a repeat diagnostic on previously inspected development
data, not a blind or sealed evaluation.

## Question and fixed change

The 8-frame IntPhys Path A input had exact opposite-label sampled-media
collisions in 9 of 90 scenes. The frozen protocol in
[`R2_INTPHYS_FRAME16_PATHA_PROTOCOL_20261127.md`](R2_INTPHYS_FRAME16_PATHA_PROTOCOL_20261127.md)
therefore tested one change: uniformly sampled RGB frames increased from 8 to
16. The 16-frame positions were 1, 7, 14, 20, 27, 34, 40, 47, 53, 60, 67, 73,
80, 86, 93, and 100 (1-based).

The frozen VideoPrism-B checkpoint and source revision, 264,960-parameter
causal LSTM, seed 17, train/validation IDs and order, optimizer, learning rate,
weight decay, batch size, epoch limit, patience, and `min_delta` were held
constant. The 16-frame candidate used the same minimum-raw-validation-loss
checkpoint rule as the corrected 8-frame reference. No sealed test was opened.

## Data and artifact integrity

The official IntPhys train shard is CC BY 4.0, SHA-256
`5f4ce9b2240bdbd57e32ed2d8315300314d30572f39adb0d9e58afa3b27da83a`. It
contains 3,750 unique videos: 3,004 training and 746 internal validation
examples, with zero cross-split group overlap. The new features contain 3,004
sequences shaped `[16, 768]` and 746 validation sequences shaped `[16, 768]`.
Sample IDs, order, and group IDs match the frozen 8-frame caches exactly; all
features are finite.

| Artifact | SHA-256 |
| --- | --- |
| 16-frame train cache | `d6dd8f45d160ea4ba1743da9ba232af22a23b89d10d41b6f9e363b53c6ab5ca3` |
| 16-frame validation cache | `20042d272fca2550f4a8e974cded7bd6f86c2110774a43c3289d18c2435f5d3b` |
| Train extraction receipt | `805624d793aa835b52cf57496fc1c923ca6a8d4a62a8a228bd13465c9d613af0` |
| 16-frame selected LSTM checkpoint | `1fe68a2b5aa9b6268bd4286a41d2a9843d8554ea0a54fff4efb77338716f2049` |
| Training history | `f093cc9bb9c1a26bde0e59fa57597cf2623ad9149e3f20873c87123c76c17f0d` |
| 16-frame dev cache | `02001e535b7db75240381cac52fbb27428ef544344a0edc531c5a4ec1d952251` |
| Dev extraction receipt | `423daa48cba42819aa3f1df76fc7cb664dc09d0eca2c728e308d9f5c192b7a1b` |
| Dev predictions | `a5554304fa05a8cf3a4603bb680bd090495d6914b6a9ec6eb9f5ad26b323a7df` |
| Paired comparison receipt | `1aa6f7ac3cc6f51943d81afd18dd56cff6e764ac610e684f313843093b66d0a7` |

All generated features, weights, histories, and predictions are outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\intphys-frame16-patha-seed17-attempt-01\`.
Existing 8-frame features, checkpoints, reports, and the sealed test remain
unchanged.

The fixed dev diagnostic reuses 360 videos from 90 scenes, with 180 possible
and 180 impossible labels. The same sample IDs, labels, groups, and order were
used for both models. There were zero cross-split sample, group, or normalized
media identity overlaps. The repeated dev results are not blind evidence.

## Training and compute

The 16-frame frozen feature extraction took 10,563.60 seconds end-to-end
(10,479.96 seconds in feature extraction); forward median/p95 were 2.628/3.224
seconds per video. The JAX CPU run used the pinned VideoPrism-B revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, source commit
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, and checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`. Peak
process working set was 4.16 GB. No GPU or cloud compute was used for this
candidate.

The LSTM trained on CPU for 99.61 seconds, completed 76 epochs, and selected
epoch 75. It has 264,960 trainable parameters. Validation next-feature cosine
loss was `0.00285804` (reloaded checkpoint identical); validation persistence
loss was `0.02085843`. The corrected 8-frame reference selected epoch 81 at
`0.00256564`. Thus 16 frames raised the selected validation loss by
`0.00029240` (about 11.4%). This objective measures next-feature prediction,
not binary physical reasoning or calibrated class probability.

The fixed 16-frame dev extraction took 859.33 seconds end-to-end, with
2.210/2.309-second forward median/p95. It found zero scenes with an exact
opposite-label collision in the sampled normalized RGB inputs, versus 9/90 in
the 8-frame reference. Removing that input collision is useful coverage
evidence, but does not prove that the labels are solvable from the frozen
features.

## Same-dev comparison

Each run used the same internal-validation 95th-percentile anomaly threshold
rule. The 8-frame reference is the corrected epoch-81 checkpoint, not the
preserved earlier epoch-75 checkpoint.

| Frames | Selected epoch | Accuracy | Balanced accuracy | ROC AUC | Impossible sensitivity | Possible specificity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 81 | 0.5000 | 0.5000 | 0.4919 | 0.4667 | 0.5333 |
| 16 | 75 | 0.5056 | 0.5056 | 0.4868 | 0.3722 | 0.6389 |
| 16 minus 8 | — | +0.0056 | +0.0056 | -0.0051 | -0.0944 | +0.1056 |

A paired percentile bootstrap resampled 90 scene groups with replacement,
2,000 replicates, seed 17:

- Balanced-accuracy difference (16 minus 8): mean `+0.00545`, 95% CI
  `[-0.00833, +0.01944]`.
- ROC-AUC difference: mean `-0.00503`, 95% CI
  `[-0.01509, +0.00263]`.

Both intervals include zero. The small accuracy increase is not a reliable
improvement; AUC moved slightly lower and sensitivity to impossible cases fell.
The result does not support a claim that 16 frames improved this anomaly
readout. It also does not establish that more frames, other temporal models, or
supervised temporal reasoning cannot work.

The reproducible comparison receipt is
`C:\CodexArtifacts\pretrained-reuse-r2-local\intphys-frame16-patha-seed17-attempt-01\comparison-8v16-verified-v2.json`.
It hashes both prediction files, evaluation reports, corpus reports, and
training metadata, and fail-closes if those receipts disagree. Two earlier
comparison receipts in the same external directory were superseded because
they used the preserved epoch-75 reference metadata with epoch-81 predictions.
`comparison-8v16-verified.json` contains the corrected metrics and epoch but is
superseded by `comparison-8v16-verified-v2.json`, which additionally embeds the
training-metadata hashes. The evaluator now checks receipt hashes before
pairing results.

## Data-first interpretation and next step

The extra frames remove exact sampled-input collisions but do not improve
measured IntPhys discrimination. This points away from treating frame count as
the only bottleneck. The training shard contains only possible videos, while
the fixed development diagnostic contains 90 labeled scenes. The next useful
step is to improve task/data alignment: identify a license-cleared training
source with labels or paired counterfactuals for the same temporal physical
decision, preserve scene-level split separation, and only then fit a task-matched
Decision readout. This result is not evidence for adding model capacity or
changing the product architecture.

BERT remains a possible **text-task** candidate, not a fix for this video
result. Existing English BERT-Mini and Japanese BERT evaluations are
task-specific and have split/template limitations; see the
[English BERT-Mini result](R2_COMMONSENSEQA_BERT_MINI_FROZEN_RESULT_20261126.md)
and [Japanese text cross-task audit](R2_TEXT_BERT_CROSS_TASK_RESULTS_20261123.md).
A future BERT comparison should use a new, source-disjoint text task with clear
labels and should be reported separately from video metrics.

## Pipeline corrections found

The new trainer wrapper wrote `status=complete_physical_concepts_frozen_feature_next_step_fit`
and `selected_epoch`, while the IntPhys evaluator initially accepted only the
older status and `best_epoch` names. The evaluator now accepts these two
explicit completion statuses and checks either epoch key against the selected
training receipt. Tests cover both formats. The comparison script also verifies
that evaluation report hashes bind the supplied training metadata and corpus
receipt; this caught and prevented use of the stale epoch-75 metadata for the
corrected epoch-81 reference predictions.

The first 16-frame smoke invocation lacked the repository root on
`PYTHONPATH`, so it failed before model import. The next run completed the
VideoPrism forward but report serialization failed because a `getattr` default
eagerly requested absent TensorFlow package metadata. Both issues were fixed;
the pinned single-video CPU forward was rerun successfully and produced finite
`[1, 16, 768]` pooled features.

## Commands

The main extraction, training, dev extraction, and evaluation commands are
recorded verbatim in each external run's JSON metadata. The paired comparison
was rerun with:

```powershell
python scripts/compare_intphys_frame_candidates.py `
  --reference-evaluation-dir <corrected-8-frame-evaluation> `
  --candidate-evaluation-dir <16-frame-evaluation> `
  --reference-training-metadata <corrected-8-frame-run-metadata.json> `
  --candidate-training-metadata <16-frame-run-metadata.json> `
  --reference-corpus-report <8-frame-extract-report.json> `
  --candidate-corpus-report <16-frame-extract-report.json> `
  --output <new-output-json> --bootstrap-replicates 2000 --seed 17
```

The 16-frame candidate is a diagnostic only. Do not replace the 8-frame
reference, promote a product model, or interpret its 50.6% dev accuracy as
general video Decision quality.
