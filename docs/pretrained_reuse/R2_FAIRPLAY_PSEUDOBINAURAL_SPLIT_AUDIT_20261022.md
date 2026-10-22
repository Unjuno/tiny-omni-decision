# FAIR-Play / PseudoBinaural split reconciliation — 2026-10-22

Status: **row-level metadata passes; not admitted for quality-selected AV
training or product evaluation.** This audit used only the already-saved
train/validation ID metadata and a train-only media preflight receipt. It did
not read test content, media bytes, annotations, features, or model outputs.

## Sources and procedure

- FAIR-Play repository revision:
  `facebookresearch/FAIR-Play@2666d6caf2d126ae43a84841e94e0c9ae24b3e2d`.
- PseudoBinaural code revision:
  `SheldonTsui/PseudoBinaural_CVPR2021@87f7be08d41748995c561ad1dc7bda453d4ee553`.
- The pinned [CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Visually_Informed_Binaural_Audio_Generation_without_Binaural_Audios_CVPR_2021_paper.pdf)
  describes reconstructing long recordings, grouping by scene, and creating
  five splits. The paper reports averages across those splits; it does not
  publish a machine-readable scene-group key alongside the split lists audited
  here.
- Parsed the saved metadata-only screen and the existing 8-ID train media
  preflight receipt. The audit normalizes `.wav`/`.mp3` IDs to their numeric
  asset stems, checks list row counts and duplicates, verifies each
  PseudoBinaural train/validation pair is disjoint, compares split variants,
  reconciles the original FAIR-Play random split 1 against PseudoBinaural split
  1, and confirms the preflight IDs are train-only.
- The source artifacts explicitly mark test content, media access, model loads,
  feature extraction, and fitting/evaluation as false. Their SHA-256 hashes are
  recorded in the result.
- Reproduction:

  ```powershell
  py -3.11 scripts/audit_fairplay_pseudobinaural_splits.py `
    --metadata C:/CodexArtifacts/pretrained-reuse-r2/fair-play-split-audit-attempt-01/metadata-only-screen.json `
    --preflight C:/CodexArtifacts/pretrained-reuse-r2/fair-play-train-group-preflight-attempt-01/decode-grouping-receipt-v2.json `
    --output NEW_OUTPUT_DIR/report.json
  ```

## Results

- Each of the five PseudoBinaural split variants has **1,497 train IDs and 187
  validation IDs**, with zero normalized train/validation asset-ID overlap.
- All eight preflight media IDs belong to PseudoBinaural split 1 train; none is
  in its validation list.
- The validation set has only 187 records. Even if every record were its own
  independent scene, it cannot reach the project's proposed 200-independent-
  group minimum. The actual independent scene count is **UNKNOWN** because the
  saved lists have no explicit original-recording or scene-group key.
- PseudoBinaural's repository says its five splits are non-overlapping; the
  FAIR-Play README describes them as having “no or less scene overlap.” The
  current metadata proves row-level separation within each variant, but does
  not independently reconstruct that scene grouping. Sources: [pinned
  PseudoBinaural repository](https://github.com/SheldonTsui/PseudoBinaural_CVPR2021/tree/87f7be08d41748995c561ad1dc7bda453d4ee553),
  [pinned FAIR-Play README](https://github.com/facebookresearch/FAIR-Play/blob/2666d6caf2d126ae43a84841e94e0c9ae24b3e2d/README.md).
- **Correction to attempt 04:** overlap of the full train+validation universes
  (1,510–1,607 IDs pairwise) is expected when training partitions overlap and
  is not by itself evidence about validation independence. The direct
  validation-to-validation audit found 13–35 shared IDs per pair. Across the
  five lists there are 935 validation rows and 744 unique asset IDs. For each
  fold, 134–161 of its 187 validation IDs also occur in another fold's train
  list. These are overlapping resampling assignments, not five independent
  evaluation replicates or seeds. The current lists do not identify original
  scene groups, so 744 unique asset IDs cannot be reported as 744 independent
  scenes.
- Of the 744 unique validation IDs, 504 are in split 1's train list and 53 are
  outside split 1's exposed train+validation lists. A descriptive split-1
  reallocation would leave 993 split-1 train IDs after excluding the 504 known
  IDs, but it is **not admitted**: scene identity is unresolved, and the 53
  outside IDs cannot be reconciled against split 1's sealed test without
  opening that split. The test list was not accessed. Do not pool the five
  validation sets or construct this reallocation until source-level group and
  sealed-test reconciliation is possible without exposing test records.
- The old FAIR-Play random split 1 cannot be mixed with PseudoBinaural split 1.
  Both train+validation lists contain 1,684 IDs, but only **1,518 IDs match**;
  each list has 166 IDs absent from the other. Also, **148 of 187 original
  random validation IDs appear in PseudoBinaural train**, while 144 original
  random train IDs appear in PseudoBinaural validation. Keep one split protocol
  fixed; do not recover a larger validation set by combining the two.
- Results, input hashes, split-order hashes, validation overlap matrices and
  preflight membership are in the preserved [attempt-06 report](C:/CodexArtifacts/pretrained-reuse-r2/fair-play-split-audit-attempt-06-named-overlap-metrics/report.json)
  (SHA-256 `4ebe262d9fea14b8d88f909a7a476ae6007a2512a960a86c5a556f14dacd9c9a`).
  Attempts 01–05 remain preserved; attempt 06 uses semantically explicit metric
  names and does not access test lists or records.

## Decision and remaining gates

Per-fold row-level train/validation separation is confirmed. The paper gives a
scene-aware split-generation method, but the local manifest does not expose
group IDs to independently verify it or count groups. The five validation
lists can be audited as overlapping resampling assignments; a pooled holdout
cannot be admitted from their union because 53 IDs are outside split 1's
exposed train+validation lists and their status relative to its unopened test
set is unknown. No task examples or hard negatives were constructed, and no
evidence shows that matching audio to one of several video candidates requires
both modalities. Therefore do not fit a quality-selected readout yet or claim
AV complementarity or general AV reasoning.

The next admissible step is source-side scene-identity reconciliation and a
frozen candidate-matching protocol with matched audio-only, video-only, and
question/option controls. Use a group-aware out-of-fold protocol only after
the five split assignments and original-scene identities are verified; retain
an untouched final set. The provisional 200-group floor applies to unique,
independently supported groups, not the sum of correlated fold rows. The
source/task proposal remains research-only until rights, group count, and
both-input necessity pass.

All local attempts are preserved without overwrite. Attempt 02 produced a
preliminary report; attempt 03 clarified the group-count field; attempt 04 adds
the exact cross-source union-size reconciliation; attempt 05 separately audits
validation-to-validation overlap and corrects attempt 04's overbroad fold
independence conclusion; attempt 06 uses explicit metric names for those
overlaps.
