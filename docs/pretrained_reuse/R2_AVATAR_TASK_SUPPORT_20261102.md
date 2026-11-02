# AVATAR TRAIN task-support follow-up — 2026-11-02

## Scope

This is a metadata-only follow-up over the already extracted AVATAR TRAIN
annotation tree at dataset revision
`a7f0146ff704799d044133e61137836cacc1c233`. Prior model-use source groups were
collected from the inventoried local fit/evaluation sidecars. No media bytes,
features, model weights, official validation annotations, or sealed-audit data
were opened. Source group IDs are not emitted.

The machine-readable output is
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-task-support-20261102-attempt-01\report.json`
(SHA-256 `6f98db1103db7640c140dab8e937740179f094bcfab05ab4789a6426ca1df5ac`).
Its annotation-tree digest is
`4c17a0bdbd7d488bced018404ce71be5659d27273efcfdec9e1cdfffd16eb651`; the
prior-exposed-group set digest is
`4f0fbd6dc342c8c485c4c187bc102a988def78fdaeef11879ed11221ab4adf3a`.

## Aggregate support

| Scenario | Unexposed source groups | Illustrative train / validation (80/20) | Unexposed annotations / frames |
|---|---:|---:|---:|
| Single-Sound | 0 | 0 / 0 | 0 / 0 |
| Mixed-Sound | 635 | 508 / 127 | 6,453 / 3,386 |
| Multi-Entity | 452 | 362 / 90 | 2,517 / 2,347 |
| Off-Screen | 75 | 60 / 15 | 213 / 213 |

The split is a deterministic support illustration using seed 17, not a frozen
benchmark split. Mixed-Sound spans 70 categories, but only 23 have at least
four source groups in both illustrative folds. Of its 3,386 annotated frames,
1,305 contain more than one annotated category; 1,371 contain exactly one
Mixed-Sound annotation. These counts do not establish that unannotated
categories are valid negatives.

The existing 3,009-file AVATAR media cache has zero source-group overlap with
the 635 previously unexposed Mixed-Sound groups. This was a filename-only
identity scan; media contents were not read. No media acquisition or feature
extraction was attempted.

## Task interpretation and decision

The official AVATAR card defines Mixed-Sound as multiple overlapping sources
and describes localization metrics (CIoU/AUC); the paper says the task must
discriminate and associate multiple concurrent sources with visual sources.
This is inherently multi-positive localization. It cannot be directly
converted into a one-target four-choice example without changing the task.
Possible binary candidate-region questions would still need a defensible
negative-label rule and matched audio-only, video-only, missing-stream, and
shuffled-stream controls before they could satisfy the both-input gate.

Under the current 80/20 support screen, validation has 127 independent source
groups, below the provisional 200-group quality floor. The candidate is
therefore not admitted for media processing or fitting. Do not adjust the split
merely to pass the floor. First define a task-preserving multi-positive Decision
representation and a negative-label policy; then evaluate whether the fixed
group budget can support a clean validation set. Current C: free space is below
0.02 GiB, also below the 2 GiB reserve.

Official references: [AVATAR dataset card](https://huggingface.co/datasets/mipal/AVATAR)
and [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Choi_Whats_Making_That_Sound_Right_Now_Video-centric_Audio-Visual_Localization_ICCV_2025_paper.pdf).
