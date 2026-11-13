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

The first 80/20 screen had only 127 Mixed-Sound validation groups. A follow-up
support-only screen uses one deterministic 60/40 assignment over the union of
previously unexposed source-video groups across all task types. It yields 609
train and 406 validation groups globally, with no overlap; Mixed-Sound has 370
train and 265 validation groups (55 categories represented in both folds, 32
with at least four source groups per fold). The validation count clears the
provisional 200-group floor, but this is not a frozen benchmark split and
category support remains uneven (the weakest shared category has one validation
group). Exact report and hashes are recorded in the follow-up section below.

The task remains multi-positive localization. One candidate Decision encoding
is a binary query per fixed, mask-independent spatial point: synchronized
audio/video plus timestamp and normalized point coordinates, with supplied
options `[background, sound source]`. A point is positive when it falls inside
the official union of visible-source masks. Queries are independent across
points; probabilities are not normalized over the whole frame. Ground-truth
boxes/masks must not be model inputs or proposal generators. The grid and point
bank are not yet frozen; first verify coordinate alignment and freeze them from
TRAIN-only evidence. Matched audio-only, video-only, shuffled-stream,
missing-stream, and coordinate-prior controls are still required.

The provisional group-count gate passes, but AVATAR remains
`NOT_ADMITTED_FOR_FITTING`. No media payload, features, model, official
validation annotations, or sealed-audit data were opened; no quality result
exists. Source-video rights, task/grid freeze, media verification, and matched
input controls remain gates. C: had about 2.87 GiB free when the screen ran and
about 2.60 GiB after CPU validation; both are above the 2 GiB reserve, but
extracting the 0.49 GiB expansion would leave too little margin for subsequent
artifacts, so extraction was not started. The local RTX 3080 Laptop was occupied
(15,821 MiB used / 355 MiB free at the initial check), so no GPU feature
extraction or fitting was started.

## Follow-up evidence — global split and local archive inventory

The follow-up re-used the pinned TRAIN annotation tree at revision
`a7f0146ff704799d044133e61137836cacc1c233` and the inventoried prior-use
sidecars. A seed-17 hash ranking assigned the union of 1,015 previously
unexposed source-video groups once with validation fraction 0.40. The split is
global across task types, so a source video shared between tasks cannot cross
the illustrative folds. The result is 609 train / 406 validation groups with
zero overlap. It was selected without model scores and is still an illustrative,
not frozen, development split.

The updated Mixed-Sound support is 370 train / 265 validation groups across 70
categories, with 55 categories in both folds and 32 categories having at least
four groups per fold. Minimum validation support among shared categories is
one group. The annotation tree remains 19,159 JSON files with digest
`4c17a0bdbd7d488bced018404ce71be5659d27273efcfdec9e1cdfffd16eb651`; the
prior-exposed group-set digest is
`4f0fbd6dc342c8c485c4c187bc102a988def78fdaeef11879ed11221ab4adf3a`.

The reproducible aggregate report is
`C:\\CodexArtifacts\\pretrained-reuse-r2\\avatar-task-support-20261102-attempt-02\\report.json`
(SHA-256
`6ecce0f7684b3d97a1a1744e94e5d71eca13fb81847fe4f7804b710856097433`). Its
global train/validation group hashes are
`2ab6aad8eb73e3c85f4529406068187244cf619e1edcddef2a7795739834d9be` and
`9ee1a810ad8dd7032c2d46c55359fc80f19e7c73a8d8d58851386d4a0b55a65b`.

The media inventory read only `video.zip`'s central directory. All 738
annotated Mixed-Sound clips were present (517,372,409 compressed bytes;
523,793,811 expanded bytes); zero member payloads were opened or decoded. The
existing 3,009-file extracted-media cache has zero filename-identified overlap
with the fresh Mixed-Sound source groups.

### Official task and label semantics

The dataset card identifies CC BY 4.0 and defines Mixed-Sound as multiple
overlapping sources. The official evaluator at the pinned revision
`a7f0146ff704799d044133e61137836cacc1c233` unions masks for all visible
instances, then evaluates localization maps using CIoU/AUC. Pinned files:
[`evaluator.py`](https://huggingface.co/datasets/mipal/AVATAR/blob/a7f0146ff704799d044133e61137836cacc1c233/code/evaluation/evaluator.py)
(SHA-256 `0560545251519b5d0bce20ca1d3c8e2d57b5545f7fe08c85302d99f4f300827b`)
and
[`utils.py`](https://huggingface.co/datasets/mipal/AVATAR/blob/a7f0146ff704799d044133e61137836cacc1c233/code/evaluation/utils.py)
(SHA-256 `cef3a03a5b3befff7d1859526ed075a32e7d3ebfba70b51ae3e9f3b94b8d549b`).
The official label is multi-positive localization, not one winning source
category. Unannotated categories cannot be treated as category-level negatives;
the candidate binary target is only whether a point lies in the official union
mask.

This task representation is a candidate only. Before fitting, freeze the
mask-independent point bank and spatial resolution using TRAIN-only evidence,
verify frame/mask coordinate alignment, and run the matched unimodal and
input-perturbation controls on the same source-group split. The dataset-card
license does not replace review of underlying source-video rights or
redistribution conditions. No feature extraction, fitting, official validation
or sealed-audit access occurred in this follow-up.

Official references: [AVATAR dataset card](https://huggingface.co/datasets/mipal/AVATAR),
[ICCV 2025 project page](https://hahyeon610.github.io/Video-centric_Audio_Visual_Localization/),
and [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Choi_Whats_Making_That_Sound_Right_Now_Video-centric_Audio_Visual_Localization_ICCV_2025_paper.pdf).
