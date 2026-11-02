# R1 AVATAR source-rights follow-up — 2026-11-03

## Scope

Read-only reconciliation of the exact pinned AVATAR dataset card and VideoPrism provenance records. No archive, annotation payload, video/audio bytes, official validation labels, or sealed-audit data were opened; no model was loaded and no features or metrics were generated.

## Findings

- The AVATAR dataset repository at revision `a7f0146ff704799d044133e61137836cacc1c233` reports `cc-by-4.0` in its pinned Hub metadata and README. It describes `video.zip` and `metadata.zip` as dataset contents and identifies VGGSound as the source family for the training IDs.
- The official VGG-Sound page says the dataset is available for commercial/research use under CC BY 4.0, while also stating that copyright remains with the original video owners. The AVATAR card does not provide per-YouTube-video rights receipts or clarify that the annotation license grants rights in every underlying source video. Therefore, the annotation/material license is not treated as proof that each source video is cleared for product-directed feature extraction or redistribution.
- VideoPrism-B remains pinned to `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`, checkpoint SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`, implementation commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`. The local artifact was rehashed at 457,475,296 bytes; both preserved runtime receipts report 114,365,184 float32 parameters and identical finite-output SHA-256 `df185755af2fabc6c8dcf0765743edd5cbb01cf6bb6bc7d968e469a929b9796f` on the deterministic eight-frame CPU fixture.
- The pinned VideoPrism materials disclose large pretraining collections, including YT-Temporal-180M, InternVid, VideoCC and WTS-70M, but no checkpoint-specific item-level training-video ID manifest. The AVATAR TRAIN input sidecar's 2,268 YouTube source groups therefore cannot be compared exactly to VideoPrism's training items. Overlap is **UNKNOWN**; neither zero overlap nor an observed collision is established.

## Disposition

Keep AVATAR **`REVIEW_NOT_ADMITTED_FOR_MEDIA/FEATURE FITTING`** for product-directed model work. The CC BY metadata supports attribution-aware use of AVATAR's author-provided annotations as such; it does not settle rights in each linked video. The unknown VideoPrism pretraining overlap also prevents describing AVATAR source-group holdout as independent-source transfer. Existing AVATAR development results remain diagnostic only; do not reopen their validation groups for tuning. No additional model-quality experiment is authorized by this audit.

Machine-readable record: [`avatar-videoprism-rights-followup-20261103.json`](../../manifests/pretrained_reuse/avatar-videoprism-rights-followup-20261103.json).

## Sources

- [Pinned AVATAR Hub metadata](https://huggingface.co/api/datasets/mipal/AVATAR?revision=a7f0146ff704799d044133e61137836cacc1c233) and [pinned README](https://huggingface.co/datasets/mipal/AVATAR/blob/a7f0146ff704799d044133e61137836cacc1c233/README.md).
- [Official VGG-Sound source and rights statement](https://www.robots.ox.ac.uk/~vgg//data/vggsound/).
- [Existing VideoPrism overlap audit](R1_AVATAR_VIDEOPRISM_OVERLAP_AUDIT_20261102.md), [pinned official README](https://github.com/google-deepmind/videoprism/blob/c6eba0ece3dd49a24ffc44582966997a4a59f1d8/README.md), and [pinned official license](https://github.com/google-deepmind/videoprism/blob/c6eba0ece3dd49a24ffc44582966997a4a59f1d8/LICENSE).
