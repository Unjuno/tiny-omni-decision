# R2 CATER temporal-task feasibility screen — 2026-11-13

## Result

The already downloaded CATER scene metadata supports a plausible video-only
temporal Decision task with enough selector examples for a bounded Path A
evaluation. This is a **metadata feasibility result only**. CATER remains
`REVIEW_NOT_ADMITTED`: no video was acquired or inspected, no model was run,
and this does not resolve possible visual near-duplicates or qualify a model.

The fixed task candidate asks: **which object had the greatest net displacement
between the endpoints of one specified 75-frame interval?** Each source scene contributes at most one query
per fixed, non-overlapping interval: `0–75`, `75–150`, `150–225`, and `225–300`.
Every object in the scene remains an option (5–10 choices in the eligible
cohort); no option-count reduction was used. A scene/query is eligible only if
all objects have unique `(shape, color, material)` descriptors, the largest
annotated displacement is at least 0.20 world units, and it exceeds the
runner-up by at least 0.10 world units. These thresholds are now frozen for
this candidate before any model evaluation.

| Split candidate | Candidate video groups | Eligible groups | Distinct video-interval queries | Options/query |
|---|---:|---:|---:|---:|
| `train_subsetT` after conservative Clevr-4 numeric-ID exclusions | 1,679 | 1,196 | 4,372 | 5–10 |
| `train_subsetV` after the same exclusions | 439 | 308 | 1,120 | 5–10 |

The selector has 308 independent scene groups but 1,120 correlated queries;
uncertainty estimates must resample by video/scene, not by query. The four
intervals are separate temporal questions, not repeated copies of one question.
Per-interval eligible selector counts are 287, 281, 277, and 275 respectively.
These counts pass the provisional 1,000-query / 200-group feasibility screen,
but do not establish task quality or generalization.

## Reproduction and integrity

Reproduce from the exact local inputs with:

```powershell
python scripts/audit_cater_temporal_task_feasibility.py `
  --scenes-zip D:\CodexArtifacts\pretrained-reuse-r2\cater-source-screen-20261113-attempt-01\all-actions-scenes.zip `
  --lists-dir D:\CodexArtifacts\pretrained-reuse-r2\cater-source-screen-20261113-attempt-01 `
  --collision-receipt D:\CodexArtifacts\pretrained-reuse-r2\cater-source-screen-20261113-attempt-01\clevr4-index-collision-screen.json `
  --output D:\CodexArtifacts\pretrained-reuse-r2\cater-temporal-task-feasibility-20261113-attempt-01\feasibility.json
```

The script fails closed unless the audited CATER scene archive, train/selector
lists, and Clevr-4 collision receipt match their pinned SHA-256 values. It
checks train/selector video-ID disjointness and scene `image_index` identity.
The machine-readable receipt is stored outside Git at
`D:\CodexArtifacts\pretrained-reuse-r2\cater-temporal-task-feasibility-20261113-attempt-01\feasibility.json`
(SHA-256 `f28444999f0c469b3c990733d41360f1597956825dcacd9fb1719ecf2ab48618`).
The script SHA-256 is
`a6c7ce5f85cb5012bcb9b92a2c46e086450876f0cc5bff147094cd838f9c7b6d`.

Pinned inputs:

- CATER source commit: `13a19643f1a2fb24e931df25abd74353e4f2fdcb`.
- `all-actions-scenes.zip`: `2855bc5d765516b113dc3bf1cd8c43c7923d3a6350450af816ac6790f4d4f51d`.
- `train_subsetT.txt`: `62b1733632a7f447bcc1fcb1a4e1cb4472fabf14d756fe5cabd5df4afddd23cb`.
- `train_subsetV.txt`: `e320cfe0af43d8c1256885096952a4a0bb4e31475332f720a6b530cccd425b81`.
- Prior Clevr-4 numeric-ID collision receipt: `94bf68dc1a56fea3f227a3552a22f6d8e49ecd76b916dd3c4069f91a2b2b6f50`.

The separate coarse CATER/Clevr-4 signature audit found no eligible CATER
scene with Clevr-4's image-wide uniform shape/color/texture signature. This
narrows the exact-identity concern but does not exclude arbitrary visual
near-duplicates. No media, features, model outputs, sealed audit, or GPU were
used for this screen. Official CATER validation/test list contents remain
unopened. Rights interpretation and acquisition scope are recorded separately
in [the license follow-up](R2_CATER_LICENSE_SCOPE_FOLLOWUP_20261113.md).

## Next gate

Before downloading selected CATER videos or fitting a readout:

1. Resolve the remaining cross-source visual-identity risk with a bounded,
   non-sealed train-only media check, or document why the existing exact-ID and
   scene-attribute evidence is sufficient under the R2 gate.
2. Freeze a group-level train/selector manifest from the eligible IDs; never
   split intervals from one video across partitions.
3. Inspect a small, fixed sample of selected videos to verify object
   descriptions match rendered appearance and that annotated displacement is
   visible over the fixed interval. Record hashes and do not use this sample to
   tune the model.
4. Only then run one frozen video Path A baseline with temporal-order,
   frame-shuffle, position-prior, and option-order controls. Report both query
   and scene-group counts, and bootstrap by scene.

This candidate is video-only M1 evidence. It cannot close the roadmap's
both-input-required multimodal M2 gate or support a claim about general Omni
quality.
