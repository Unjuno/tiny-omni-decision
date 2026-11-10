# R1 AVMIT feature-producer provenance audit — 2026-11-10

## Disposition

**AVMIT remains a research-only, precomputed-feature diagnostic. It is not admitted as a product runtime path or as evidence that the selected Path A encoders reproduce its feature pipeline.** This audit is read-only: it did not load feature arrays, raw media, sealed data, or model weights and did not rerun validation.

The official AVMIT extractor at pinned source revision [`e3c06ec0e215b1934d141f86008e92e567366d6c`](https://github.com/mjoannou/audiovisual-moments-in-time/blob/e3c06ec0e215b1934d141f86008e92e567366d6c/feature_extractor/extract_features.py) specifies:

- Audio: TensorFlow Hub `https://tfhub.dev/google/yamnet/1`, mono audio resampled to 16 kHz; expected embedding width 1,024.
- Video: Keras `EfficientNetB0(include_top=False, weights="imagenet", pooling="avg", input_shape=(224,224,3))`; RGB frames sampled every 480 ms and resized to 224×224; expected width 1,280.
- The repository's [extractor README](https://github.com/mjoannou/audiovisual-moments-in-time/blob/e3c06ec0e215b1934d141f86008e92e567366d6c/feature_extractor/README.md) gives broad dependency requirements but no immutable environment lock or feature-generation receipt.

The local cache report is `D:\CodexArtifacts\pretrained-reuse-r2\avmit-metadata-gate-20261110-attempt-01\features-v1\cache-report.json`, SHA-256 `4b88888b09676c854a6272edb136ef91b3d2240468a0a596a80f6b203d6f8e7f`. It binds the cache to AVMIT source revision `e3c06ec0e215b1934d141f86008e92e567366d6c`, source metadata SHA-256 `0a4c607b99acb248e30d3c3eb29ab074ac668989e02518459530b4de6994ded9`, archive SHA-256 `9a76ce7fd43bcac714019c08d6c39177e7901f3cd9794fe1b34775c2c30c1127`, and widths 1,024 / 1,280. It records `raw_media_opened=0` and `sealed_payloads_opened=0`.

## Findings and unresolved provenance

| Question | Finding |
|---|---|
| Does the checked-in AVMIT extractor identify the intended producer families? | Yes: YAMNet Hub version 1 and Keras EfficientNetB0 with ImageNet weights. |
| Does the archive/cache receipt identify exact YAMNet and EfficientNet checkpoint hashes or generation IDs? | **No. UNKNOWN.** The feature cache report has no producer checkpoint hash, model artifact generation, or extraction environment receipt. |
| Can the exact Keras/TensorFlow version used for the released AVMIT embeddings be established? | **No. UNKNOWN.** The extractor accepts a broad TensorFlow range and the dependency file is not a lockfile. |
| Do official Keras docs establish the terms for the specific ImageNet-derived no-top weights used in this archive? | They document `weights="imagenet"` as the initialization choice, but do not by themselves identify the historical weight bytes or settle separate rights for that artifact. Treat product clearance as **REVIEW/UNKNOWN**, not as established. See the [official Keras EfficientNet API](https://keras.io/api/applications/efficientnet/efficientnet_models/). |
| Are the original Moments in Time source media included or cleared by the AVMIT feature/annotation record? | No. The AVMIT paper describes obtaining raw clips separately; media rights remain separate from the released annotation/precomputed-feature license. This experiment opened no raw media. |
| Do AVMIT's feature producers match current Path A producers? | No. Path A uses different pinned encoders/preprocessing. The AVMIT result is a task/fusion feasibility diagnostic only. |

The official extractor establishes what the published code requests, not which exact binaries generated the downloaded feature archive. The local feature widths and shapes are consistent with the documented family, but dimensions are not cryptographic producer provenance.

## Existing result boundary

The frozen-seed SAVC development result is recorded in the source experiment branch `codex/r2-avmit-metadata-gate-20261110` (PR #61, source commit `d63228c74baf6d6369ee4b5269b4778007f1fd13`) and the external `run-v1/fit-v1/results.json`. Its readouts were trained on precomputed features only. On the frozen development validation split, the mean readout had Accuracy 0.671943, NLL 0.586653, Brier 0.404667, and ECE-15 0.023219; the GRU readout had Accuracy 0.661481, NLL 0.600030, Brier 0.415760, and ECE-15 0.035225. These metrics are cited solely as the already-recorded result; this audit did not reopen predictions or rerun evaluation.

Those scores do **not** support the requested 95–98% quality target, do not establish general audio/video reasoning, do not demonstrate raw-input runtime capability, and do not establish four-modality performance. The SAVC task is closed-set AVMIT action-label correspondence.

## Gate decision and next evidence required

- Keep AVMIT's released annotations/precomputed features in the research-only lane allowed by the existing source record and repository gate. Attribute AVMIT and its authors.
- Do not use AVMIT as proof of a production-ready audio/video tower, exact producer parity, or clearance of original media.
- Before any future product/runtime admission, require a producer receipt that pins exact model artifact revisions and SHA-256 hashes, exact TensorFlow/Keras/runtime versions and preprocessing, and a rights review that separately covers source media and pretrained weights.
- Do not rerun the exposed development validation to resolve provenance questions. Provenance must come from producer-side receipts or newly authorized data, not from additional scoring.

## Scope and integrity

Source branch: `codex/r2-avmit-metadata-gate-20261110`, commit `d63228c74baf6d6369ee4b5269b4778007f1fd13` (PR #61 worktree). Audit branch starts at roadmap commit `e07e049ff4387484fb754e0fff963631f3cc7c85`. No model was loaded, no GPU was used, no features or media were read, no sealed or published-test payload was opened, and no validation was rerun. Teacher v1 and other Teacher artifacts were not changed.
