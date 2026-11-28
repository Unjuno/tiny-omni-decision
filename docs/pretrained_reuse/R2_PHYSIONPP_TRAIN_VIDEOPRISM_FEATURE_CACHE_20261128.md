# Physion++ train-only frozen VideoPrism feature cache — 2026-11-28

## Execution

Extracted VideoPrism-B features for all 652 rows in the existing Physion++
TRAIN manifest. The 2-row CPU smoke passed first. An initial direct-script
launch exposed a Python import-path issue before the output directory or model
was created; the extractor now supports both module and direct-script launch,
and the 2-row smoke then completed. The full run used the fixed 8-frame,
288-pixel profile, endpoint-inclusive uniform frame selection, frozen model
weights, mean pooling over VideoPrism spatiotemporal tokens, and JAX CPU only.

No validation, publisher test, or sealed data was read. No readout was fit or
selected. The feature cache deliberately contains sample IDs, media hashes and
features only; labels are not stored in the cache. Every video SHA-256 and
decoded frame count was checked against the existing train manifest before
feature extraction.

## Integrity and measured runtime

| Check | Result |
|---|---:|
| Train rows processed | 652 / 652 |
| Unique sample IDs / unique media hashes | 652 / 652 |
| Feature shape | 652 × 768 float32 |
| Non-finite feature values | 0 |
| Cache order | exact train-manifest order |
| Labels in feature cache | no |
| Runtime | 827.65 s (13 min 47.65 s) |
| RSS before model load / after extraction | 346,742,784 B / 1,928,216,576 B |
| Device | `TFRT_CPU_0` only |
| Cache size | 1,901,918 B compressed |

The runtime includes model initialization, first-call compilation, per-video
media hashing and decoding. It is a single CPU run, not a GPU throughput
benchmark or a general service latency estimate.

## Pinned provenance and artifacts

- Train JSONL SHA-256: `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`
- Source dataset revision: `0c53b983ca138962ba636c0936e735490ad4c326`
- VideoPrism revision: `be719a406d563b66f0ac969e7c94bab8e997c81a`
- Upstream VideoPrism source commit: `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`
- Checkpoint SHA-256: `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`
- Cache SHA-256: `32846b085df7642d9190df8f553a58496d275f8580efe6c6a8adfa3a83217360`
- Progress JSONL SHA-256: `07cdc06e70c3303b4494ffc37439c2557ae90d064a1d385d0e91ce7b75aaf11e`
- Run report SHA-256: `8502d674a92cda35b42292bddb19767c22904c9943a2b680c7902442570643de`
- Extractor SHA-256: `1a3c797debd533797ba99d39679a8e18f668e083f1b5e040db2c02569c74822c`
- Focused extractor tests: `tests/test_extract_physionpp_train_videoprism_features.py` — 4 passed
- Output: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-videoprism-features-20261128-attempt-02`

## Status and limits

This is the full frozen-feature extraction stage of a narrow video Path A
candidate, not a Decision result. It establishes that the pinned encoder can
process this train corpus under the current local CPU runtime. The human
blinded label-observability review remains pending; this cache does not resolve
that data-quality gate. Readout fitting and development-validation extraction
remain unstarted, so no Accuracy/NLL/Brier/ECE or product-quality claim exists.
