# TCA-Bench and FAVE source screen — 2026-11-12

## Decision

Neither source is admitted for product-directed Path A fitting or evaluation.
Both are relevant to audio-video temporal reasoning, but their annotation/data
terms include a non-commercial restriction. No media or labels were downloaded
or opened, no model was loaded, and no training or evaluation was run.

## TCA-Bench

The dataset publisher's Hugging Face card describes an audiovisual temporal
alignment benchmark with 459 short videos and stage-2 audio-video binding and
stage-3 temporal-relation annotations. The visible dataset viewer exposes a
326-row `test` split; no independent train/validation split or source-group
protocol is declared in the card. Its stated license is CC BY-NC-SA 4.0, which
does not meet the project's product-directed rights gate. The task is also
primarily caption evaluation; adapting it to supplied-option Decision output
would require a separately frozen protocol.

## FAVE

The CVPR 2026 supplemental material describes FAVE's AV alignment and temporal
multiple-choice tasks, a useful task shape for this project. It states that
FAVE annotations are released under CC BY-NC-SA 4.0. The underlying QVHighlights
source being MIT-licensed does not change the annotation license. This source
therefore fails the same product-directed rights gate and was not acquired.

## Disposition

Keep both candidates `REVIEW_NOT_ADMITTED`; do not use them for fitting,
checkpoint selection, or product evaluation. Reconsider only if a rights-holder
provides compatible terms and a versioned, group-disjoint train/validation/test
protocol is available. This screen does not change R2-M's data-admission gates
or establish any model-quality result.

## Primary sources

- [TCA-Bench publisher dataset card](https://huggingface.co/datasets/26TCA/TCA-Bench)
- [FAVE CVPR 2026 supplemental material](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Lu_FAVE_A_Structured_CVPR_2026_supplemental.pdf)
