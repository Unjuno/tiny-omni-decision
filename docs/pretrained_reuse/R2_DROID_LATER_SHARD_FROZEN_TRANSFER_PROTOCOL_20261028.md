# DROID later-shard frozen-readout transfer protocol

## Question and scope

Replay the three already-fitted DROID readouts on a one-time, previously
unopened sample of episodes from official-train shards 00466, 00759, and 01632.
The sample was selected before labels were read and has zero source-ID,
recording-group, and sampled-frame overlap with the earlier 20-shard task. The
metadata-only sample audit found no new collection sites, so this is
same-site/group-disjoint transfer evidence, not a new-site or blind audit.

The question is whether the existing VideoPrism-only, raw-sensor-only, and
VideoPrism+sensor frozen logistic readouts retain their earlier behavior on
new episode groups. This does not test a new architecture. The previous
development validation and this transfer sample are not paired because they
contain different episodes.

## Frozen components

- Feature model: `google/videoprism-base-f16r288`, revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`, checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Existing fitted readouts are loaded without modification. Their hashes are
  pinned in `configs/pretrained_reuse/droid_later_shard_frozen_transfer_v1.json`.
- Inputs are 8 uniformly indexed observations from the first quarter of each
  episode, the same three camera views and 14 raw proprioception channels as
  the original probe. The target is the next commanded Cartesian direction in
  the fixed seven-choice vocabulary.
- No readout fitting, checkpoint selection, calibration, threshold selection,
  or additional training is performed.
- The data come from the official training partition. The selected sample was
  not previously opened for labels/media. No sealed audit is read.

## Integrity gates

The task builder verifies the source-only audit receipt, frozen shard indices,
object sizes and SHA-256 hashes, unique episode and recording-group IDs,
non-overlap with the prior task, the 8-observation rule, finite sensors and
targets, and duplicate/overlapping sampled JPEG hashes. It writes one
evaluation-only split. The feature runner verifies the frozen model and source
manifest hashes, runs on CPU, and stores features outside Git. The evaluator
verifies task/cache/readout hashes, IDs, order, labels, and probabilities
before reporting overall, per-site, per-shard, and per-class Accuracy, NLL,
Brier, ECE-10, and confidence.

If any source/media overlap or artifact hash gate fails, do not modify the
sample or tune around it; record the run as blocked and investigate the source
identity gate separately.

## Timing estimate

The earlier full 906-episode CPU VideoPrism extraction took 3,078.37 seconds,
or 17.66 examples/minute. At that measured rate, 137 eligible episodes predict
about 7 minutes 46 seconds of encoder work. Model initialization, task
construction, and file validation add overhead. Re-estimate from the new run's
first progress intervals; do not treat the initial estimate as a guarantee.
No GPU or cloud compute is allocated.

## Commands

Paths and new attempt directories are fixed in
`configs/pretrained_reuse/droid_later_shard_frozen_transfer_v1.json`. The
reproduction commands and effective hashes are recorded in the result report
after execution. Each output directory must be new; no prior artifact is
overwritten.

## Decision rules

Report every arm and all strata regardless of outcome. Describe differences
from the earlier development validation as unpaired descriptive changes. Do
not use this one-time result to modify or select a model. The data are still
from the official training split, sites overlap the earlier cohort, and the
task is narrow robot next-action prediction; the result cannot establish
general Omni quality, site generalization, or that video is a useful
complement to sensors on other tasks.
