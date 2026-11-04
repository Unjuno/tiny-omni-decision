# AVATAR official rasterizer parity — 2026-11-03

## Result and disposition

The TRAIN-only comparison found that the Pillow mask rasterizer used by the
point-grid support audit is not pixel-identical to AVATAR's pinned OpenCV
rasterizer. Across 1,962 Mixed-Sound TRAIN frames (640×360), 219,084 of
452,044,800 pixels differed (0.04847%); 1,820 frames had at least one pixel
difference. Center-query labels differed on 194/384,552 (14×14), 214/502,272
(16×16), and 754/1,538,208 (28×28) queries.

This is a geometry-implementation result, not a model-quality result. It does
not measure localization performance or establish that any grid is suitable
for the frozen feature coordinates. Use the pinned OpenCV behavior if a future
TRAIN corpus needs official-mask parity; do not describe the prior Pillow
support counts as exact official rasterization. AVATAR remains
`REVIEW_NOT_ADMITTED` for fitting: source-video rights, feature-coordinate
alignment, negative-query/loss policy, and matched native-task controls remain
open. No model, feature, media payload, official validation annotation, or
sealed audit was loaded for this comparison.

## Pinned reference and method

The dataset revision is `a7f0146ff704799d044133e61137836cacc1c233`. Its
`code/evaluation/utils.py` (SHA-256
`cef3a03a5b3befff7d1859526ed075a32e7d3ebfba70b51ae3e9f3b94b8d549b`) converts
each polygon with `np.array(p).reshape(-1, 2).astype(int)` and rasterizes it
with `cv2.fillPoly(mask, [p], 1)`. The pinned evaluator
(`code/evaluation/evaluator.py`, SHA-256
`0560545251519b5d0bce20ca1d3c8e2d57b5545f7fe08c85302d99f4f300827b`) unions
visible instance masks. The audit mirrors those operations and compares them
to the existing Pillow union-mask implementation.

The comparison replayed seed 17 and the existing global source-group split
over 1,015 previously unexposed groups: 609 train / 406 validation, zero group
overlap. Pixel and point measurements use only 370 Mixed-Sound TRAIN groups
(426 clips, 1,962 frames, 60 categories). Internal validation-group JSON was
parsed only to reproduce assignment; validation masks/dimensions, official
validation labels, and sealed-audit data were not used. The TRAIN annotation
tree SHA-256 is
`4c17a0bdbd7d488bced018404ce71be5659d27273efcfdec9e1cdfffd16eb651`.

OpenCV 4.10.0 came from the isolated `opencv-python-headless==4.10.0.84`
wheel (SHA-256
`afcf28bd1209dd58810d33debb62b325d3cbe49dcd7a43a902982c33e5fad05`); it was
installed into the attempt artifact directory rather than the project Python
environment. Python 3.11.9 and NumPy 2.4.6 were used. Runtime was 15.288 seconds
on CPU; no GPU or cloud compute was used.

## Measurements

| Center grid | Queries | Pillow positives | OpenCV positives | Different labels | Difference fraction |
|---:|---:|---:|---:|---:|---:|
| 14×14 | 384,552 | 54,345 | 54,539 | 194 | 0.05045% |
| 16×16 | 502,272 | 71,219 | 71,433 | 214 | 0.04261% |
| 28×28 | 1,538,208 | 217,886 | 218,640 | 754 | 0.04902% |

For grids 14 and 16, both implementations found at least one positive center
in 1,949 frames; for grid 28, both did so in all 1,962 frames. Similar support
counts do not mean that the individual query labels match. The point-label
differences are directly relevant to any future supervised point-query
corpus, so the prior Pillow counts remain approximate geometry evidence only.

## Reproduction and artifact integrity

The run used `scripts/audit_avatar_rasterizer_parity.py` with
`--expected-opencv-version 4.10.0`, seed 17, validation fraction 0.4, and grids
14, 16, and 28. The full machine-readable result is kept outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-rasterizer-parity-20261103-attempt-01\report.json`,
SHA-256
`8b22556f6aaa7493fe73e61b65daa01836d1badaf166b1a3ebd39e8a7880c242`.
The committed manifest records the command, split and data hashes, tool
versions, wheel hash, report hash, and explicit non-access assertions.

Implementation hashes: `scripts/audit_avatar_rasterizer_parity.py`
`26e11c01190cd5f71b61042b24d8354071b79da82aad86d5b636acba1c05a8f7`;
`tests/test_audit_avatar_rasterizer_parity.py`
`050b074f1e34da3427df200ddcce2a754b0623b52688374fa3a1db538f449c4c`.

Focused tests pass: `ruff check scripts/audit_avatar_rasterizer_parity.py
tests/test_audit_avatar_rasterizer_parity.py` and
`python -m pytest -q tests/test_audit_avatar_rasterizer_parity.py
tests/test_audit_avatar_point_grid_support.py` (7 passed). No source media,
features, model weights, or checkpoints were changed.

## Next gate

Keep AVATAR unadmitted until source-video rights are resolved, feature-grid
coordinates are matched to the pinned encoder, and TRAIN-only negative-query
and loss rules plus native localization and unimodal/input-perturbation
controls are frozen. Only then consider a separate small feature smoke. This
audit does not justify fitting, validation tuning, or product promotion.
