# LVOmniBench source screen — 2026-11-22

## Decision

**Status: `NOT_ADMITTED_FOR_PRODUCT_QUALITY`.** LVOmniBench is a strong task
match for long-form audio-video reasoning: the official paper describes 275
videos and 1,014 manually written four-choice questions designed to require
both audio and video. The official paper explicitly releases the benchmark
under CC BY-NC-SA 4.0, so it is not admitted to this product-directed path. The
public annotation mirror's Apache-2.0 metadata does not override that conflict
or license the underlying video assets. No payload was accessed.

## Pinned metadata and rights

- Official project repository: `KD-TAO/LVOmniBench`, Git HEAD
  `68d5e3db2f9762b29f90c585fb59fd27f6607a68` at screen time.
- Official paper: arXiv `2603.19217`, which states in Appendix 0.C that the
  dataset is CC BY-NC-SA 4.0. The paper says source videos came from YouTube
  and that all selected videos adhere to Creative Commons licenses, but the
  public metadata reviewed here did not provide an auditable per-video license
  ledger with exact variants.
- Original Hugging Face source `KD-TAO/LVOmniBench` revision
  `587494e7abcf1c1e3db6be3055db052363bf3636` is automatically gated.
- Repacked annotation-only dataset `xxayt/LVOmniBench` revision
  `ae9d203ca347b8195d3448e2528ff00d858c3e0f` tags itself Apache-2.0 but states
  that the upstream release does not specify a license and the repack therefore
  defaults to Apache-2.0. This statement conflicts with the official paper's
  explicit benchmark license. The repack contains no videos and cannot grant
  rights to the underlying media.

## Task and split fit

The official project describes 10–90 minute videos (about 34 minutes on
average) and audio types including speech, sound, and music. Its MCQ format
matches the supplied-option interface, and the authors state that questions
require joint audio-visual reasoning. The published resources reviewed expose
one benchmark corpus, not a separately documented train/validation split for
iterative model development. The set is therefore not suitable for tuning a
candidate even apart from the rights restriction. Its long duration also makes
local cost and temporal coverage a distinct feasibility question; no media
size was estimated or downloaded.

## Scope and re-entry

This was a source-page, paper, repository-revision, and dataset-card metadata
screen only. No annotations, video/audio media, model, or checkpoint was
loaded; no access request was submitted; no split, feature extraction,
training, or evaluation was performed; and no GPU was used. Do not use this
source for product training or product-quality claims. Reconsider only if the
publisher clarifies product-compatible rights for both the annotations and
each underlying media asset, and provides an independently grouped
development split. A website/code license or the annotation mirror's default
license is insufficient.

## Sources

- [Official LVOmniBench project page](https://kd-tao.github.io/LVOmniBench/)
- [Official paper, arXiv:2603.19217](https://arxiv.org/abs/2603.19217)
- [Official dataset repository](https://huggingface.co/datasets/KD-TAO/LVOmniBench)
- [Annotation-only repack and its license note](https://huggingface.co/datasets/xxayt/LVOmniBench)
- [Official evaluation repository](https://github.com/KD-TAO/LVOmniBench)
