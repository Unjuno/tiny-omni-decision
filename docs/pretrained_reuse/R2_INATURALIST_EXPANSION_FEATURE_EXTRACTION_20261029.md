# iNaturalist CC0 expansion: streamed VideoPrism features (2026-10-29)

## Execution result

The frozen pages 11–30 expansion was processed end to end with the pinned
VideoPrism-B checkpoint using a persistent CPU worker. The earlier media-size
preflight projected retaining about 0.49 GiB of PNGs; this run instead
normalized one CC0 image at a time, extracted and verified its feature, then
deleted that temporary image. Only feature vectors and media/hash receipts were
retained. The earlier 2-GiB disk-reserve check therefore did not apply to the
streaming output, while a 128-MiB minimum-free guard remained active before
each download.

| Measure | Observed |
|---|---:|
| Train / validation rows | 921 / 446 |
| Unique underlying media assets | 1,367 |
| Feature tensor | `float32[1367, 768]`, all finite |
| Total elapsed | 3,465.31 s (57 min 45 s) |
| Mean VideoPrism forward time | 1.1697 s / image |
| End-to-end throughput | 2.54 s / image |
| GPU | Not used; worker reports `TFRT_CPU_0` |
| Retained feature tensor bytes | 4,199,552 |
| Train/validation exact or pHash≤4 overlap | 0 / 0 |
| Prior-media exact or pHash≤4 overlap | 0 / 0 |

The first three-image pilot had measured 1.401 s/image for download and image
normalization; the VideoPrism smoke measured 1.249 and 1.213 s for its two
post-JIT images. Their sum projected roughly one hour for the full stream. The
actual 57:45 wall time was close to that estimate. The first 25 rows took
78.1 s including worker startup/JIT; from row 50 onward the observed rate
settled near 2.5–2.6 s/row. No image payloads remain in the run's staging
directory.

Class counts (train / validation):

| Taxon ID | Train | Validation |
|---:|---:|---:|
| 3017 | 158 | 80 |
| 4956 | 108 | 51 |
| 6930 | 75 | 39 |
| 7089 | 107 | 56 |
| 9083 | 49 | 25 |
| 12727 | 93 | 41 |
| 13858 | 156 | 80 |
| 144455 | 175 | 74 |

This is feature preparation only. No readout was trained, no quality score was
computed, and it does not establish broad image or Decision-model quality.
The validation split remains a development split, not a sealed audit.

## Reproduction and artifact integrity

- Branch: `codex/r2-inat-expansion-roadmap-20261029`
- Source commit: `e22639d299effce89ce3426b1eb6ccd86bc76128`
- CI: run 3518 succeeded for that commit.
- Command: `python scripts/stream_inaturalist_videoprism_features.py` with
  metadata directory
  `C:\CodexArtifacts\inat-birds-cc0-expansion-filtered-pages-11-30-20261029-attempt-02`,
  the pinned checkpoint/source/worker paths in the local run config, and the
  three prior-media manifests listed there.
- Run directory:
  `C:\CodexArtifacts\inat-birds-cc0-expansion-videoprism-features-20261029-attempt-01`
- Model: `google/videoprism-base-f16r288`, revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`; upstream source commit
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- Checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Frozen train manifest SHA-256:
  `362a562568910676a15b71d42233fbeb8f6811116afcb6ef22f846bcf83e8618`.
- Frozen validation manifest SHA-256:
  `c3f846c677516b5f08c82f6ea12a39146bf9a890be66306985f202622d6ddcc0`.
- Canonical run config SHA-256 (the value embedded in `result.json`):
  `34c3b570b9eb5d4dfcbb5ec32325d46001416f0b4fa5741954915f3826e2c964`.
- Serialized `run-config.json` file SHA-256:
  `bbfa174c8ea3651b1fdaa431437f85f66ba6c10bb1d91abd2d9e0c6226a1fd6f`.
- `features.npy` SHA-256:
  `f198750082a71c92f939ddfc03f8df1f2fc362a59757d50e8dfb8f01ba065ecc`.
- `feature-rows.jsonl` SHA-256:
  `d772a67381f37dbda25c77d7a2ee4e841773573023dfa920082e4f4be1d309ae`.
- `media-receipts.jsonl` SHA-256:
  `5aa874138344176d0b80581050e561b9ec33ea208a2f3d644511f53eb8006160`.
- `result.json` SHA-256:
  `914d7d97eb8b0b25021a0d4af90c4658fd126b69ff65c4acd17ec49fc26dabd9`.

An independent post-run check confirmed `features.npy` shape, dtype, finiteness,
all 1,367 row/receipt counts, and result-file checksum. The run records zero
within-split, cross-split, and prior-media exact/perceptual-hash overlaps. The
sealed audit was not accessed. The pre-existing iNaturalist cache and all
Teacher artifacts were left unchanged. The separate Orbitune GPU process was
not modified; GPU occupancy was monitored while this CPU-only run proceeded.

## Next step

Use this immutable cache in a small, separately identified frozen-image
readout experiment. Select its checkpoint on the frozen validation split and
report only development metrics. Keep the one-time transfer result and this
expansion as distinct cohorts; do not merge their validation records or tune
on a sealed audit.
