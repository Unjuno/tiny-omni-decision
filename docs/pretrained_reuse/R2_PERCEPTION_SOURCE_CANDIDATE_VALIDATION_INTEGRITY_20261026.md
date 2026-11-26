# Perception Test source-candidate validation integrity — 2026-10-26

## Scope and result

This CPU-only check validates the frozen official-validation transfer corpus,
its media, and the already-selected source-candidate readout checkpoint before
any evaluation. It does not compute model metrics, extract features, fit or
select a model, or open official test/sealed-audit data.

**Integrity result: PASS. Evaluation remains pending.** The selected seed-17
readout is the existing epoch-7 checkpoint. The cohort is an exposed,
visibility-balanced development-validation selection and must not be described
as blind or sealed.

## Verified inputs

- Source code snapshot: `codex/r2-perception-source-validation-transfer-20261026`
  at `595e4c31`; upstream dataset revision
  `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
- Frozen validation record JSONL SHA-256:
  `4a15269531ef9a2a1bccdab731fa23e30cd3754850fc72fd35028b9986bd0844`.
- Corpus manifest SHA-256:
  `f85afede73fe1d90bfed0dca7fe69dd7381b8a1102d06be531d7d4d27bcc3d5f`.
- Fixed 200-video selection SHA-256:
  `10cdc277cf417acba7e50245d8e0d189a6f60ce467c8d87dade42a41cb05e6d9`.
- Existing 200-video media-validation report SHA-256:
  `8c4b4e30988edf23c365f80175480d20d90f7533a8f97c8725665b46ec10d67d`.
- Existing epoch-7 readout checkpoint SHA-256:
  `63390ba7532348bbd600f64ed796e92a3daf13b1f44ba0b18e83b1fd304780bf`.

The corpus has 200 selected rows: 138 eligible candidates, 61 held rows, and
one duplicate. The 138 eligible rows map to 138 distinct official-validation
video IDs. Each eligible row has a distinct video group. The 138 candidate IDs
have zero overlap with the 320 training and 80 development video groups used
to fit/select the checkpoint.

All 138 candidate videos were present in the previously acquired 200-video
media cohort. The SHA-256 for every corresponding MP4 and WAV matched the
existing media-validation manifest: **276/276 files verified**. The audit also
rechecked the corpus, manifest, media report, and checkpoint hashes.

## Selection and interpretation limits

The 200-video cohort was selected and inspected in an earlier visibility
diagnostic, with 100 visible and 100 not-visible videos. The source-candidate
labels were not used to fit or select the readout, but the overlapping
visibility target means this is still a non-blind, label-stratified development
transfer. It can provide a fixed-checkpoint transfer diagnostic only; it is not
an independent product benchmark, not natural-prevalence calibration evidence,
and not a sealed audit.

The earlier source-candidate checkpoint was selected on 606 development events
from 80 different official-training videos. Its reported 56.93% set Accuracy
and 1.2748 NLL are the internal development reference, not the result for this
official-validation cohort.

## Next execution gate

The fixed checkpoint may be evaluated once on these 138 records only after the
same candidate-specific video ROI and audio-window preprocessing is reproduced.
The inspected extraction scripts currently accept only `train` and `development`
splits; do not relabel this corpus as either one. Add an explicit
`official_validation` path with tests, keep labels out of feature extraction,
and save the predictions and metric computation separately. Do not fit,
calibrate, or select from this cohort. Preserve the selected checkpoint and all
prior outputs.

The integrity verifier and report are outside Git:

- `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-validation-transfer-20261026-attempt-06\verify_integrity.py`
  SHA-256 `996e3310cc12890c0e76c4c58e44fae13cc8d7c64f3db45ad1b7bb1b31a305ae`.
- `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-validation-transfer-20261026-attempt-06\integrity-audit.json`
  SHA-256 `9b958f9c9ef2966c2e8ef1b743ba9d504fee719f4127dd0536a175dccab7dcec`.

No model, corpus, media, checkpoint, or active process was modified. No GPU,
cloud compute, training, inference, official test, or sealed-audit access was
used.
