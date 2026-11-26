# Clevr-4 100k internal-validation exposure audit — 2026-11-10

## Disposition

The candidate validation pool is **not yet admitted for Decision-quality
evaluation**. Exact byte hashes and canonical 64x64 RGB pixel hashes show no
matches against the preserved 10k Clevr-4 images. The perceptual-hash screen
produces many close pairs and is not calibrated for this synthetic image
domain. We conservatively froze a smaller eligible pool after excluding all
low-pHash candidates and all prior ID-string collisions. This narrows known
exposure risk but does not prove semantic novelty or replace a full
cross-worktree exposure review. No model was trained or scored.

## Frozen split and source evidence

- Dataset: official Clevr-4 100k archive, publisher metadata records CC BY 4.0.
- Archive: `clevr_4_100k_v1.full.zip`, 36,057,238,315 bytes. Its existing
  preflight records SHA-512
  `7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`.
  This audit confirmed size and read selected members; it did not recompute
  the 36 GB archive hash.
- The official archive split has 80,012 train and 19,988 validation images.
  The frozen seed-17 index holds out 8,002 images from official TRAIN for
  internal validation and uses 72,010 for internal training; asset overlap is
  zero. The previously observed official 19,988-image validation is excluded
  from this candidate pool and remains off-limits for tuning.
- Candidate internal-validation manifest SHA-256:
  `ee702a6846759aab6b818ff061dad4e96c7e49194f292d168cdaeeff92400f1b`.
  Its ID-order hash is
  `417a961a7c72dc74bd82f5cf9284243b941b61eec9b40917efc09b6baf70df6e`.
- The prior 10k Clevr-4 train/validation manifests contain 10,531 images.
  Their manifest hashes are recorded in the machine-readable result.

## Identity and exposure results

Every candidate member was read from the pinned archive and its raw SHA-256
matched the media manifest. Against the preserved 10k images:

| Check | Result |
|---|---:|
| Candidate assets | 8,002 |
| Reused string IDs | 576 |
| Exact media SHA-256 overlap | 0 |
| Exact RGB pixels after canonical 64x64 Lanczos resize | 0 |
| Nearest 63-bit pHash distance | min 0, median 8, p95 12 |
| Candidate images with nearest pHash distance <=2 / <=4 | 337 / 1,222 |

The ID strings are not stable across the 10k and 100k archives: for example,
`CLEVR_new_000000` has different recorded attributes in the two archives.
Therefore ID overlap alone does not establish asset identity. Conversely, the
zero exact hashes do not rule out semantic or near-duplicate scenes. The
pHash thresholds are an uncalibrated screen; the many low-distance pairs need
independent disambiguation before calling the holdout content-disjoint.

The identified 100k VideoPrism transfer evaluation used the official
19,988-image validation manifest, with training and checkpoint selection
disabled. The prior Clevr-4 fit used the separate 10k assets. This review did
not find evidence that the 8,002-image internal holdout was previously scored,
but it is not a proof over every cache or model log in all worktrees.

## Reproduction

Script: `scripts/audit_clevr4_content_overlap.py`.

```powershell
python -m scripts.audit_clevr4_content_overlap `
  --archive C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-v0\clevr_4_100k_v1.full.zip `
  --candidate-manifest C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-index-seed17-media-hashed-v1\validation-media-manifest.jsonl `
  --previous-train-images C:\CodexArtifacts\pretrained-reuse-r2\clevr4-full-coverage-seed17\train\images `
  --previous-validation-images C:\CodexArtifacts\pretrained-reuse-r2\clevr4-full-coverage-seed17\val\images `
  --expected-archive-bytes 36057238315 `
  --expected-archive-sha512 7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c `
  --output C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-phash-overlap-20261110-attempt-02
```

Run result:
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-phash-overlap-20261110-attempt-02\result.json`,
SHA-256 `e30e90fdd217bcc019feae1d9e6216538a4be0f1c02b7d2e7a1392e003e6f726`.
No media files were written; the archive and prior media were read only. The
first attempt completed media comparison but failed while resolving the old
manifest paths; the corrected second attempt completed and saved the result.

Ruff passes and four focused CPU tests pass. Tests cover deterministic pHash,
normalized-pixel hashing, exact fixture overlap, and manifest-path resolution.

## Next gate

Do not use this split to select or claim a model result yet. First inspect or
replace the pHash similarity screen with an identity method suited to these
procedurally generated scenes, and complete a read-only cross-worktree scan of
all relevant sample-ID/media-hash/cache ledgers. If these gates pass, freeze a
single task and one evaluation protocol before fitting a small frozen-feature
Decision head. Never access the already observed official validation for
selection.

## Follow-up: near-pair and scene-identity audit

The 1,222 candidates with nearest pHash distance <=4 were compared directly
against their pHash-near prior images after canonical 64x64 RGB and edge
transforms. For each candidate, the reference with the lowest RGB MAE was
retained. Across those pairs, RGB MAE min/median/p95 was
1.427/4.308/7.697; edge MAE was 0.309/0.621/1.394 on a 0–255 scale.
816/1,222 RGB comparisons had MAE <=5 and 1,015/1,222 edge comparisons had
MAE <=1. These are close visual similarities, not identity proof. The
label-only crosswalk had all four categorical attributes equal for 3 pairs;
the individual equal-attribute counts were color 165, count 438, shape 247,
texture 128. Because no image-level identity threshold is calibrated here,
every candidate within the pHash threshold remains quarantined.

Separately, the archive's `whole_scene_data.json` was streamed and canonical
scene configurations were compared for the frozen internal train and
validation IDs only. All 80,012 records matched. There were zero exact
configuration collisions and zero collisions after rounding numeric
configuration values to 3 decimals. The run independently replayed the
seed-17 ID hash split and matched the current label and media-manifest orders.
Official validation records were skipped and not hashed. Archive SHA-512 was
not recomputed in this audit; the previously pinned value remains an
unrecomputed preflight value.

The prior index report's label hashes/ID-order hashes do not match the current
label files. The current label files do match the current media manifests and
the replayed deterministic split. Therefore, this follow-up records hashes of
the actual current files as authoritative and does not claim the stale index
report reproduces them.

## Frozen conservative validation candidate pool

To avoid choosing a visual-similarity cutoff after looking at model behavior,
the frozen rule excludes every internal-validation ID that either (a) has
nearest pHash distance <=4 to a prior 10k image or (b) has an ID string
collision with any prior 10k image. Of 8,002 candidates, this quarantines
1,706 (1,130 pHash-only, 484 ID-only, 92 both) and leaves 6,296. IDs were
ranked by a fixed SHA-256 rule into a 1,000-example checkpoint-selection
subset and a 5,296-example reporting subset; source order is retained in both
files. These are only internal validation candidates, not a sealed or blind
audit set. No model metrics have been computed on either subset.

The manifests and labels were written without extracting or copying media to
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-clean-validation-20261110-attempt-01`.
Receipt hashes: selector labels
`c08af3839089638d081dd31c1118be7256418e6c2440a70a55f79e50456fb920`, selector
media `4c551e91d4e02bd896380476d0e59f1348a01ed3f95a5eed7e0d39cb9a311dee`,
report labels `184427b36d10d637b0f6f3d35e4d7dd6137fe2062bf0b5aa1f6560849e3a14b6`,
report media `0dd87a1303cb25bbc87ed3d69a2acde5ecef02f27bec16985297e9d685b09525`.
The exact quarantine and split rule is implemented in
`scripts/freeze_clevr4_internal_validation.py` and covered by focused CPU tests.
