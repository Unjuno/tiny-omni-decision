# Text-free CLEVRER 256-scene matched control

Status: frozen before the control fit; no control results are included here.

## Why this control is being run

The already completed 512-scene text-free run did not itself contain a
256-scene arm, so it cannot identify a train-coverage effect. Its original
config hypothesis is retained as written and must not be cited as an answered
question. This add-on trains a deterministic 256-scene subset against the same
512-question validation IDs, then compares it to the preserved 512-scene run.

The repository already contains a 256-versus-768 scene coverage comparison
for a different text-conditioned VideoPrism+MiniLM candidate model. This
control addresses the distinct text-free VideoPrism/readout path; it is not a
replacement or duplicate of the earlier result.

## Frozen design

- Seed 17; two existing readouts (`mean_mlp`, `bilstm`) with identical
  initialization to the corresponding 512-scene arms.
- Train set: 512 examples from 256 scenes, selected from the 512-scene parent
  train pool by ascending `SHA256("17:" + scene_group_id)`, retaining every
  row for each selected scene and preserving parent manifest order.
- The selected 256-scene list hash, derived train manifest hash, and original
  512-question validation manifest hash are pinned in the JSON config and
  media-integrity receipt. The validation sample ID order is the same as the
  512-scene arm. Scene/media/sample identities are disjoint across train and
  validation. Repeated normalized question templates are counted.
- Frozen VideoPrism framewise features are reused only for media identities in
  the selected train subset and the fixed validation pool. Their provenance
  points to the larger parent train manifest and the identical validation
  manifest; no new model features are fitted.
- Each readout receives exactly 2,048 optimizer updates and 131,072 training
  example presentations. For 512 train rows and batch size 64, this is 256
  epochs × 8 updates. Evaluate every two epochs (16 updates), matching the
  512-scene arm's 16-update validation cadence.
- Same AdamW, learning rate, weight decay, clipping, model, readout initialization,
  ordered validation IDs, macro question-type NLL selector, and checkpoint
  reload verification as the pinned 512-scene arms. No early stopping.
- Compare selected checkpoints using saved per-example predictions and paired
  scene-cluster bootstrap (5,000 resamples, seed 17, percentile 95% interval).
  Report each question type separately and preserve final checkpoints and
  learning histories.

## Claim limits

This is one seed and a development validation set used for checkpoint
selection. Training scenes are a subset of the larger arm, so it is a paired
coverage contrast for this task/model pipeline, not a new independent test.
Question templates repeat across scenes. A mixed static/temporal result is
uncertain; a null result does not establish a model-family ceiling. Neither
arm is eligible for product promotion based on this comparison.

The earlier 512-scene fit is immutable. This control uses a fresh output path;
the original 512-scene checkpoints, run, cache, and corpus are not overwritten.
