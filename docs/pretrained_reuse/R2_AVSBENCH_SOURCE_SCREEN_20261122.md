# AVSBench source screen - 2026-11-22

## Decision

**Status: `NOT_ADMITTED_FOR_PRODUCT_QUALITY`.** AVSBench is a relevant research benchmark for fine-grained audio-video correspondence, but its publisher limits dataset use to noncommercial purposes and its native target is pixel-level sound-source segmentation rather than option-conditioned JSON decisions. No dataset, annotation, media, model, or checkpoint was downloaded. No email or access request was sent, and no GPU was used.

## Source and rights

The publisher's project page describes AVSBench-object with a single-source (S4) and multi-source (MS3) setting, with pixel masks for objects producing sound. It also describes a semantic extension (AVSS). The project page makes the dataset available for research and states that the dataset is under CC BY-NC 4.0. Its code repository separately has an Apache-2.0 license; that code license does not replace or broaden the dataset terms.

The publisher links a CSV of video IDs and annotated segmentation masks through Google Drive/Baidu. It asks users to email their name and institution to receive processed video/audio. No such request was sent. The publisher links to source video IDs, so rights in the underlying media also need to be considered independently of the annotation license.

## Fit to the project

AVSBench directly tests whether audio helps localize the sound-producing object within video, which is a stronger audiovisual-interaction target than activity labels that either modality may predict alone. The official description explicitly includes scenes with multiple visible objects where only some are sounding.

However, the native output is a dense segmentation mask, not a probability distribution over caller-supplied options. Converting masks into choices would create a different evaluation task and would not establish the product's general JSON decision quality. Its noncommercial dataset license is also incompatible with admitting it as a reusable product-training source under the current product-directed roadmap.

## Scope and re-entry

This was a publisher-page and repository-license screen only. No data/media payload, annotations, user information, or model was accessed. No split was created and no benchmark result is claimed.

Reconsider only if the project explicitly establishes a noncommercial research-only scope for this source and defines a separate sound-source-grounding benchmark that preserves its spatial target. Do not fold such a benchmark into product Decision accuracy or use its results to claim four-modality quality. For the product track, continue seeking a rights-compatible source with verifiable group splits and supplied-option labels whose correctness demonstrably depends on both audio and video.

## Publisher sources

- [AVSBench publisher project page](https://jasongief.github.io/AVSBench/)
- [AVSBench official code repository](https://github.com/OpenNLPLab/AVSBench)
- [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/)
