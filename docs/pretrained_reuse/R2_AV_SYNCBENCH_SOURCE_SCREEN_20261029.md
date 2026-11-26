# AV-SyncBench source screen — 2026-10-29

## Decision

`REVIEW_NOT_ADMITTED`. AV-SyncBench has a useful task shape for the R2-M gate:
choose whether an audio/video pair is synchronized or whether its audio was
temporally or semantically perturbed. The task is pairwise and directly
multimodal. However, the public source metadata does not establish rights to
the underlying media or the independent-source split needed for this project.
The roughly 68.19 GB payload also exceeds current local free space. No data or
model files were downloaded.

## Pinned source evidence

The author repository was screened at commit
[`82e1353`](https://github.com/Duck-Boss/AV-SyncBench/tree/82e13536074377b71a51d81b2f5559a43cf4a122).
Its README reports 3,269 videos and 38,390 evaluation pairs across ten
scenarios, covering temporal offset/jitter/speed and voice/instrument timbre
perturbations. The [paper](https://arxiv.org/abs/2607.00726) describes the
benchmark as separating temporal from semantic AV consistency.

The [ModelScope dataset repository](https://modelscope.cn/datasets/coming245/AVSyncBench)
is the linked data source, but its current card/license could not be
independently verified in this screen. The author repository explicitly says
the media may retain the original creators' rights. Its README also says
Synchformer weight terms are not stated separately and identifies ImageBind as
CC BY-NC-SA 4.0. The repository's reproduction instructions estimate about
68.19 GB for the full dataset. Code or repository metadata does not clear
source-media rights for product-directed evaluation or training.

The benchmark's task could isolate a real joint-input skill. A valid project
study would still require video/source-group disjoint splits and matched
video-only, audio-only, joint, and corruption controls. For example, a
video-only scorer sees the same video on both sides of an audio-perturbation
pair, while an audio-only scorer cannot establish audiovisual synchronization
without an external temporal reference; each specific challenge still needs
its own leakage/shortcut audit.

## Reproducibility and disposition

Metadata-only inspection of the pinned author repository and paper metadata on
2026-10-29. No dataset archive, source media, model checkpoint, training, or
sealed-audit access. Reconsider only after explicit media-rights confirmation,
split/source grouping verification, and a storage-safe subset plan. See the
[machine-readable screen](../../manifests/pretrained_reuse/av-syncbench-source-screen-20261029.json).
