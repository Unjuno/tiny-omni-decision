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
- The five split variants are **not independent folds**. Their pairwise exact
  asset-ID overlaps range from 1,510 to 1,607 of 1,684 IDs. Do not average them
  as five independent evaluations or use them as five independent seeds.
- The old FAIR-Play random split 1 cannot be mixed with PseudoBinaural split 1.
  Both train+validation lists contain 1,684 IDs, but only **1,518 IDs match**;
  each list has 166 IDs absent from the other. Also, **148 of 187 original
  random validation IDs appear in PseudoBinaural train**, while 144 original
  random train IDs appear in PseudoBinaural validation. Keep one split protocol
  fixed; do not recover a larger validation set by combining the two.
- Results, input hashes, split-order hashes, and preflight membership are in the
  preserved [attempt-04 report](C:/CodexArtifacts/pretrained-reuse-r2/fair-play-split-audit-attempt-04/report.json)
  (SHA-256 `1b2119a7a92d0922e7387cc6f0dbe237392fec458c8aa5cc79e877ff14172b17`).

## Decision and remaining gates

The row-level split makes PseudoBinaural split 1 a better development starting
point than FAIR-Play's random split, but the current validation is underpowered
for the project's product gate and its scene grouping is not independently
verified. No task examples or hard negatives were constructed, and no evidence
shows that matching audio to one of several video candidates requires both
modalities. Therefore do not fit a quality-selected readout on this split yet,
and do not claim AV complementarity or general AV reasoning.

The next admissible step is a source-side scene-identity reconciliation and a
frozen candidate-matching protocol with matched audio-only, video-only, and
question/option controls. It must use at least 200 independently supported
validation groups; the old random split's validation rows cannot fill that gap.
The source/task proposal is research-only unless those gates are satisfied.

All local attempts are preserved without overwrite. Attempt 02 produced a
preliminary report; attempt 03 clarified the group-count field; attempt 04 adds
the exact cross-source union-size reconciliation.
