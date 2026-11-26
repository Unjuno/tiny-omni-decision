# D2E synchronized desktop source screen — 2026-10-31

## Disposition

`REVIEW_NOT_ADMITTED` for this product-directed workstream. Do not download or
fit D2E-480p under its current terms. This was a metadata-only check of the
official project repository and Hugging Face dataset card; no media, event
records, hidden labels, or model weights were downloaded or opened.

## Pinned evidence

- Official code/repository revision checked: `worv-ai/D2E@1adbc09f81810412413ff046a80e8333df80647d`.
- Official Hub dataset revision checked: `open-world-agents/D2E-480p@f075f7e25df6f6d385840a836f86bf92dfb877ff`.
- The Hub card labels the dataset `CC-BY-NC-4.0`, exposes one `train` split
  with 459 rows, and reports 194 GB total file size. Non-commercial terms do
  not meet the current product-directed data gate; the dataset is also far
  beyond the available bounded acquisition plan.
- The project describes synchronized game video, audio, and timestamped
  keyboard/mouse/window events across 29 PC games. This is potentially relevant
  for state-to-action research, but the card does not provide a ready-made
  supplied-option Decision benchmark, independent train/validation/audit
  split, or game/session group manifest adequate for this project.
- The released game recordings also raise source-game/media rights questions
  separate from the dataset-card license; this screen did not resolve them.

## Gate impact

D2E is not an admitted route for the required screen/video/audio or joint
Decision gates. Its synchronized event structure makes it a useful research
lead only if the data terms and underlying rights change or are clarified, and
if a bounded, group-disjoint option-decision task is specified. No training,
feature extraction, validation, or inference is authorized by this screen.
Continue R2 source admission on rights-cleared, size-bounded sources; do not
reuse D2E labels or media in another branch unless its admission gate changes.

## Primary sources

- [Official D2E repository](https://github.com/worv-ai/D2E/tree/1adbc09f81810412413ff046a80e8333df80647d)
- [Pinned D2E-480p Hub card](https://huggingface.co/datasets/open-world-agents/D2E-480p/tree/f075f7e25df6f6d385840a836f86bf92dfb877ff)
