# AVATAR held-category transfer probe — protocol (2026-11-02)

## Question and scope

Does the frozen AVATAR audio/video representation and the already selected
73,857-parameter concatenation readout transfer to AV event categories absent
from all readout fitting and checkpoint-selection examples?

This is a one-seed, train-partition-only development diagnostic. It is not a
new external audit, a raw-media runtime result, or a product-quality claim. It
does not change the existing corrected AVATAR corpus, readout, checkpoints,
validation, or sealed audit. Frozen feature encoders are not run or updated.

## Frozen inputs

Only the corrected AVATAR `cross-source-option-choices-02` TRAIN model input and
TRAIN analysis sidecar are read. The source group and category references are
used to rebuild candidate pools. Feature vectors are selected only for keys
referenced in those TRAIN files. The source cache container also holds prior
validation feature rows, but this probe does not read validation labels or
predictions, and the split builder is given no validation JSONL.

| Input | SHA-256 |
| --- | --- |
| TRAIN model input | `0139f4f59a685c3f83890156b59c42d12aca9db54b22ee5c0a9c70b6ec7d60ef` |
| TRAIN analysis sidecar | `90adbf04450e0854fdb379dbcceb6aaa73790b205c08f1cf6c2cb786b6710a57` |
| audio feature cache | `5736e2c7e094a93aea81fa13f517799292e790197dda6f6282140e3882636a58` |
| video feature cache | `6ab47dcc0f9c8fe691425fa45a6f1394c8520677829a419c51b93a45da068b6e` |

The existing feature dimensions are 384 for EfficientAT audio and 768 for
VideoPrism video. Candidate video assets are restricted to the frozen TRAIN
feature cache; no media download or feature re-extraction occurs.

## Split and choices

1. Sort the 80 TRAIN target categories by
   `SHA256("avatar-category-transfer-v1:17:<category>")`; the first 16 are held
   out. This selection is fixed by seed 17 and is not based on quality metrics.
2. A source group is assigned to held-category test if any TRAIN query or
   candidate-video metadata associates that group with a held category. All
   other groups are eligible for seen-category fitting/selection.
3. Split seen groups by the first 20% ranked by
   `SHA256("avatar-inner-groups-v1:17:<group>")` for inner validation; remaining
   groups are fit-only.
4. Rebuild choices independently inside each source-group/category pool.
   Each row has one same-category positive from another source and three
   different-category candidates. Candidate sources are unique within a row,
   and the audio query source is not a video option source. No source group,
   asset, or query ID crosses fit, inner validation, and held-category test.
5. Use one row per distinct TRAIN audio asset. Retain all eligible rows; do not
   shrink or rebalance the held test after seeing predictions.

Pre-fit TRAIN-only preflight found 7,464 fit, 1,874 inner-validation, and 2,490
held-category test choices, with respectively 1,404, 351, and 513 source
groups. All three cross-split source-group, asset, and query overlaps are zero.
All 16 held categories have at least four held-test query groups; Tuba is the
minimum-support category (23 queries / 4 groups). Per-category counts will be
reported so this limitation remains visible.

## Fit and selection

- Seed: 17, set once in an isolated CPU process.
- Model: per-option `[audio 384 + video 768] -> 64 -> GELU -> 1`, matching the
  AVATAR concatenation MLP; 73,857 trainable parameters.
- Normalize audio and video using fit-only means/stds; floor standard
  deviations at `1e-5`.
- AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 128, at most 50
  epochs, patience 6, minimum improvement 0.001.
- Select the best checkpoint by macro-category NLL on inner validation. Save
  best and last states separately. The held-category test is not used for
  early stopping, checkpoint selection, or further tuning.
- Hide CUDA with `CUDA_VISIBLE_DEVICES=-1` and require PyTorch to report zero
  visible CUDA devices before fitting. Limit Torch to at most two CPU threads.

## Reported outputs

For the held-category test, report Accuracy, NLL, Brier, ECE-15, mean
confidence, macro-category metrics, all 16 per-category metrics and counts,
and a 5,000-replicate seed-17 source-group bootstrap interval for Accuracy.
Keep per-example IDs, option categories/source groups, targets, predictions,
and probabilities outside Git. Record train history, selected epoch, update
count, parameter/checkpoint size and hash, wall time, environment, and split
manifest hash. Uniform four-choice chance is 25%.

No test result will trigger another seed, split, candidate-pool change, or
hyperparameter adjustment. A low result is evidence against this readout's
zero-shot category transfer under this task construction; it is not a ceiling
claim for the model family.
