# UnAV-100 source/task screen — 2026-10-25

## Decision

**Status: REVIEW; do not fetch, train, or evaluate.** UnAV-100 is relevant to
audio-visual event localization, but the public source does not establish
training rights for the underlying videos. Its event labels also do not show
that both audio and video are required for a decision.

## Task fit

The official project describes more than 10,000 untrimmed YouTube videos, over
30,000 audio-visual events, and 100 event categories, with event time bounds.
This is a useful task-design lead for timestamped event selection or
localization. It is not already a calibrated supplied-option Decision task.
The current task description does not specify counterfactual audio-only and
video-only controls proving that a selected event requires both streams. A
future conversion would need a frozen candidate construction, time-window
policy, source-group split, and matched audio-only, video-only, shuffled-pair,
and timing-offset controls.

## Rights, access, and split evidence

- The official project page says raw videos are retrieved from a CSV of
  YouTube IDs and start/end times. It also lists annotations and audio/visual
  features on an external file host.
- The page states that the dataset is offered under CC BY 4.0 for
  commercial/research use, while **copyright remains with the original video
  owners**. The page does not document a separate grant from those owners for
  using the videos or their audio in model training. CC BY's broad permissions
  apply only to material the licensor can license; therefore the source-owner
  rights question remains unresolved.
- The CSV is described as carrying train/validation/test labels, but this
  screen did not inspect rows or establish that all clips from the same source
  video remain in one split, nor did it test normalized-content or media
  identity overlap.
- No CSV, annotation, feature, or video files were accessed or downloaded.

## Reconsideration gate

Re-screen only after explicit rights covering the underlying media, model
training, evaluation, derived weights, and intended distribution are available
from the relevant owners or authorized licensors. Then independently audit
video-level split grouping and media/content overlap before admitting any
records. Freeze a Decision conversion with the modality-necessity controls
before model selection. Do not treat the dataset-level CC BY statement or the
authors' pretrained benchmark results as proof of source-video rights or
cross-modal Decision quality.

## Primary sources

- Official UnAV-100 project page, task, source format, split fields, and
  copyright statement: <https://unav100.github.io/>
- Creative Commons Attribution 4.0 deed:
  <https://creativecommons.org/licenses/by/4.0/>
