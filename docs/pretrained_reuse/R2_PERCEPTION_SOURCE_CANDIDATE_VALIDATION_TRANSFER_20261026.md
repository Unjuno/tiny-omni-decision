# Perception Test source-candidate validation transfer (2026-10-26)

## Result

The planned transfer evaluation could not be run. The pinned official validation
sound-localisation annotation archive contains sound events but no object-track
annotations, which the existing audio-event-to-visual-source task requires to
construct candidate boxes and distractors. This is a validation-data/schema
compatibility blocker, not a model-quality result.

The archive was inspected read-only:

- Source revision: `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`
- Member: `challenge_sound_localisation_valid.json`
- Archive SHA-256: `4f0e602ba8f97313aca7d84805334a73edd94fa6f56d80b945954cd32327c141`
- Video records: 5,359
- Sound events: 35,625
- Rows with usable `object_tracking`: 0 / 5,359
- Every validation row has only the top-level keys `metadata` and
  `sound_localisation`.

The pinned validation cut map has SHA-256
`3cd9ae2d56f28ba70fa80ccca04aca6481ebab1bf1099f3aabe92afad89a67f7`.
The preselected 200-video cohort selection has SHA-256
`10cdc277cf417acba7e50245d8e0d189a6f60ce467c8d87dade42a41cb05e6d9`.
Its previously generated diagnostic corpus is preserved outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-candidate-validation-transfer-20261026-attempt-01/corpus/`;
all 200 selected rows were held, with zero eligible candidates. That attempt
loaded no model and computed no metrics. The cohort had also been inspected for
visibility in a separate diagnostic, so it is not blind or a sealed audit.

## Integrity handling

The validation-corpus builder now fails before creating an output directory if
any selected validation row lacks a nonempty `object_tracking` list. Running it
against the pinned real inputs failed closed and identified all 200 selected
videos as missing usable tracks. This prevents a structurally unusable corpus
from being mistaken for a completed transfer evaluation. The train path and its
existing split behavior are unchanged; validation rows retain explicit
`official_validation` provenance.

No model was loaded, no inference or GPU work was performed, and neither the
official test split nor sealed audit was opened. The existing training corpus,
checkpoint, and source-candidate readout were not changed.

## What would unblock this evaluation

An official validation object-tracking annotation source must be identified and
verified to join to the sound annotations by the same video IDs and parent
object IDs. If such a source is unavailable or incompatible, this specific
source-candidate transfer evaluation remains unavailable; do not fabricate
tracks, relax candidate gates, or substitute train annotations. A different
validation task may be designed from available annotations, but it would need
its own pinned data contract and must not be described as this transfer test.
