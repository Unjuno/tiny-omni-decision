# AVATAR TRAIN point-grid support audit — 2026-11-03

## Purpose and decision

This CPU-only, metadata/annotation audit checks whether a fixed spatial point
grid can represent AVATAR Mixed-Sound masks as independent binary Decision
queries. It is a label-support feasibility check, not a feature experiment,
model result, native-task score, or candidate admission. AVATAR remains
`REVIEW_NOT_ADMITTED` for fitting: rights for source videos, exact official
geometry parity, the point/feature coordinate contract, and a defensible
training-query policy remain open.

The candidate output semantics are one Bernoulli decision per supplied point:
`[background, sound-source-present]`; a point is positive when it lies inside
the union of the frame's visible instance masks. Different points can be
positive simultaneously. Probabilities must not be softmax-normalized across
points. This is closer to the official multi-positive localization target than
converting each frame to a single-winner choice, but does not by itself define
the model input or evaluation protocol.

## Data and split boundary

- Dataset: `mipal/AVATAR` revision
  `a7f0146ff704799d044133e61137836cacc1c233`.
- The global source-group assignment was replayed with seed 17 and validation
  fraction 0.4 over previously unexposed groups: 1,015 total, 609 train, 406
  validation; group-set hashes are recorded in the report and manifest. The
  two group sets have zero overlap.
- Geometry counts below use only Mixed-Sound records assigned to the global
  train side: 370 source groups, 426 clips, 1,962 frames, 60 categories, all
  at 640×360. No frame lacked a visible union mask.
- The internal validation-group annotation JSON was parsed only to replay the
  split. Its masks and dimensions were not used. Official validation and
  sealed-audit annotations were not read. No media payload, feature, or model
  was opened.
- Annotation-tree SHA-256:
  `4c17a0bdbd7d488bced018404ce71be5659d27273efcfdec9e1cdfffd16eb651`.

## Fixed-grid center coverage

| Grid | Queries/frame | Frames with ≥1 positive | Frames with >1 positive | Mean positive cells | Positive query fraction | Queries total |
|---:|---:|---:|---:|---:|---:|---:|
| 14×14 | 196 | 1,949 / 1,962 (99.34%) | 1,921 | 27.70 | 14.13% | 384,552 |
| 16×16 | 256 | 1,949 / 1,962 (99.34%) | 1,932 | 36.30 | 14.18% | 502,272 |
| 28×28 | 784 | 1,962 / 1,962 (100.00%) | 1,955 | 111.05 | 14.16% | 1,538,208 |

On these training masks, 28×28 adds 13 covered frames relative to 16×16,
while producing about 3.06× as many point queries. This is a support/cost
tradeoff only. It does not establish that the pretrained visual feature map
has a matching 28×28 coordinate system, nor that a denser grid improves
localization or Decision quality. No grid has been selected using validation
labels.

## Geometry limitation and provenance

Attempts 01 and 02 are preserved. Attempt 01 encountered one- and two-point
degenerate polygons that a three-vertex-only rasterizer could not handle;
attempt 02's pure-Python point-in-polygon loop was manually stopped for poor
throughput. Attempt 03 used a Pillow 1-bit raster mask and completed. This
attempt 01 on 2026-11-03 adds a per-grid count of frames with multiple positive
queries and reproduces the same split and aggregate support counts.

The support mask uses Pillow polygon rasterization with integer-truncated
coordinates, `draw.line` for two-point masks, and `draw.point` for one-point
masks. It has **not** been validated against the official OpenCV `fillPoly`
implementation. Boundary cells, integer conversion, and degenerate masks may
therefore differ. The result must not be used as an official localization
metric. Before any corpus construction or fitting, add fixture-level and
representative-mask parity checks against the pinned official rasterizer.

The report is
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-point-grid-support-20261103-attempt-01\report.json`
(SHA-256 `4d6b7e2449a60631e1791c425a7b2010489092ba3b00a8f82be3ce24f99c19ca`).
The report was produced by `scripts/audit_avatar_point_grid_support.py` from
source commit `ff26621b158290e0202067cd9261f7d23f6168bc` plus the uncommitted
script revision whose hash is recorded in the manifest. Python 3.11.9 and
Pillow 10.4.0 were used; wall time was 17.69 seconds on CPU. GPU time and
cloud cost were zero.

## Next gate

Keep the candidate unadmitted until all of these are resolved:

1. Confirm source-video rights separately from the dataset-card license.
2. Verify rasterizer parity and define how point coordinates map to the exact
   frozen visual feature grid.
3. Freeze frame/point sampling, the multi-positive loss and any negative
   sampling or weighting from training data only.
4. Freeze native localization metrics and matched audio-only, video-only,
   missing-stream, shuffled-stream, and time-offset controls before fitting.
5. Recheck local storage and GPU availability before a separate, small
   train-only feature smoke. Keep the held-out project audit unopened.

No model was trained, no checkpoint was created, and no quality or product
claim follows from this support audit.
