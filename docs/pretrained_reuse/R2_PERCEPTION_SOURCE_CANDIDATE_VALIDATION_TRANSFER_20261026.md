# Perception Test source-candidate validation transfer (2026-10-26)

## Result

The sound-localisation-only validation archive cannot support this task because
it omits object tracks. The official full validation annotations archive does
include tracks, so records can be joined without borrowing training labels.
This permits a fixed validation corpus; it does not make the cohort blind or
provide model metrics by itself.

The official full validation archive was downloaded from the Perception Test
release and inspected read-only:

- Source revision: `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`
- Archive: `valid_annotations.zip`, member `all_valid.json`
- Archive SHA-256: `eeefeb43955b1850b5e521ad774fe66675baac0a2a06617bd83ffbe43aaf03d7`
- Video records: 5,900; all declare `split=valid`
- Each row contains metadata, sound localisation, and object tracking
- Official sound-only archive SHA-256:
  `4f0e602ba8f97313aca7d84805334a73edd94fa6f56d80b945954cd32327c141`; its
  5,359 rows contain sound events but no object tracks

The cut map SHA-256 is
`3cd9ae2d56f28ba70fa80ccca04aca6481ebab1bf1099f3aabe92afad89a67f7`. The
preselected one-event-per-video cohort SHA-256 is
`10cdc277cf417acba7e50245d8e0d189a6f60ce467c8d87dade42a41cb05e6d9`.
All 200 selected video IDs exist in the full validation archive. Applying the
existing candidate-construction and consolidation gates yielded 138 eligible
selected events, 61 held events, and 1 duplicate. The selection had already
been inspected for visibility in another diagnostic; therefore this is a
development-validation transfer, not a blind or sealed audit. The previous
attempt-01 corpus (built from the sound-only archive) remains preserved outside
Git and is not used for model evaluation.

## Integrity handling

The builder is pinned to `all_valid.json` and fails before creating output if a
selected video lacks usable object-track annotations. It requires the official
validation split and matching video identities, and emits explicit
`official_validation` provenance. The existing train-only default and its
serialized train records are unchanged.

The official full archive gives an independent annotation route within the
same validation split: sound `parent_objects` refer to object track IDs in the
same video. In the selected 200-video cohort, all selected videos joined; among
the 9,858 nonnegative parent-object references in the 915-video overlap between
the sound-only and single-object-tracking validation subsets, every reference
matched a track ID. This cross-check is not required to fabricate missing
labels: the full `all_valid.json` already contains both annotation types.

No model was loaded, no inference or GPU work was performed, and neither the
official test split nor sealed audit was opened. Existing training corpora,
checkpoints, and source-candidate readout artifacts were not changed. The next
step is to freeze the corpus from the pinned full archive, verify media and
split integrity, then inspect the existing training/readout artifact and
preprocessing so the same frozen checkpoint can be evaluated without tuning.

## Source

The official [Perception Test repository README](https://github.com/google-deepmind/perception_test#download-the-data-and-annotations)
lists the full `valid_annotations.zip` separately from the sound-task-only
validation annotations and documents the annotation fields. The source code is
Apache-2.0; dataset materials are CC-BY-4.0 per the repository README.
