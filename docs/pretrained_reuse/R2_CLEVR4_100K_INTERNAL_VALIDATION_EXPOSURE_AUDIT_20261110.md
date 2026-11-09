# Clevr-4 100k internal-validation exposure audit — 2026-11-10

## Disposition

The candidate validation pool is **not yet admitted for Decision-quality
evaluation**. Exact byte hashes and canonical 64x64 RGB pixel hashes show no
matches against the preserved 10k Clevr-4 images, but the perceptual-hash
screen produces many close pairs and has not been calibrated for this
synthetic image domain. This audit did not train, score, tune, or select a
model.

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
