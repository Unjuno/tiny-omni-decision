# HarmonySet / RealSync AV source screen — 2026-10-22

Status: **task-design leads only; neither source is admitted for fitting or
quality evaluation**. This screen used public project, dataset-card, paper, and
repository pages only. No annotations, dataset payloads, source videos, or
sealed evaluation data were downloaded or opened.

## HarmonySet

The official project page describes 48,328 video-music pairs with annotations
for rhythm/synchronization, theme, emotion, and culture. This is a useful
semantic AV Decision task lead: candidate construction could ask which of
several supplied music tracks best matches a video on a named dimension, and
temporal questions could be derived from synchronization annotations.

The public Hugging Face card's example points to a YouTube video and presents
an instruction/response annotation. The card reports 144 MB of files; its file
listing shows `HarmonySet_Train.json` and `HarmonySet_Test.json`, without a
visible validation split. The card does not declare a dataset license, and the
reviewed materials do not grant rights to the underlying video/music media or
establish video-group separation, source-media identity, or a usable untouched
validation split. The public project page's CC BY-SA 4.0 footer applies to the
website, not by itself to the underlying media or dataset annotations.

**Disposition:** do not download annotations/media, fit, or evaluate. Revisit
only if the authors supply explicit annotation and source-media permissions,
group-level split documentation, and a distinct authorized development split.

## RealSync / StreamSync

The official StreamSync repository describes a focused AV synchronization
task: classify temporal offset between audio and video, with 21 offset classes
from -2 to +2 seconds. Its linked RealSync metadata release reports disjoint
source-video counts of 511 train, 74 validation, and 85 test sources, while
also warning that the public metadata release differs from the paper's
headline collection/protocol and that exact full-video cutting and timestamp
registration remain unresolved. The repository says users obtain source
videos independently and that the release does not redistribute them.

This is a better-defined temporal task than generic event-presence prediction,
but it tests synchronization/offset classification rather than the full
multimodal decision objective. The public pages reviewed do not establish
rights to acquire/use the underlying YouTube videos for model training or
commercial derivatives. The split metadata alone is not permission to fetch
or process those videos.

**Disposition:** keep as a task-design and protocol reference only. Do not
retrieve source media, fit, or score. Revisit only after the original media
rights are clear and the selected metadata revision, exact media/timestamp
mapping, split hashes, and Decision-option conversion are frozen.

## Next action

Do not start another AV architecture fit on the currently exposed synthetic,
Perception Test, HoloAssist, or Greatest Hits development populations. Continue
the R2 source/task admission gate. Prioritize sources that provide (1) explicit
model-training and derivative-use terms for the actual media, (2) a fixed
group-disjoint train/validation split, and (3) labels/controls proving that the
decision benefits from both modalities. If no such source is available, keep
the AV quality gate open rather than converting a source-rights or split
failure into a model result.

## Official references

- [HarmonySet project page](https://harmonyset.github.io/)
- [HarmonySet dataset card](https://huggingface.co/datasets/Zzitang/HarmonySet)
- [HarmonySet CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_HarmonySet_A_Comprehensive_Dataset_for_Understanding_Video-Music_Semantic_Alignment_and_CVPR_2025_paper.html)
- [StreamSync / RealSync repository](https://github.com/jvoas655/StreamSync)
- [RealSync WACV 2025 paper](https://openaccess.thecvf.com/content/WACV2025/papers/Voas_Temporally_Streaming_Audio-Visual_Synchronization_for_Real-World_Videos_WACV_2025_paper.pdf)
