# AVSD / Charades source and task screen — 2026-11-15

## Disposition

**`REVIEW_NOT_ADMITTED`.** No AVSD data, audio, video, annotations, model,
features, or gated downloads were accessed. No registration or terms were
accepted. This source is not eligible for product-directed Path A fitting under
the evidence currently available.

## Pinned public source evidence

- Author-maintained DSTC7 AVSD challenge repository, observed `master` commit
  `6a5ee8542132ad6634ee02896d7c935b8c447d78`:
  [README at pinned commit](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge/blob/6a5ee8542132ad6634ee02896d7c935b8c447d78/README.md)
  and [Data README at pinned commit](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge/blob/6a5ee8542132ad6634ee02896d7c935b8c447d78/Data/README.md).
- The pinned README reports audio, video, and dialog inputs, about 10 QA turns
  per video, and official split sizes of 7,659 train, 1,787 validation, and
  1,710 test dialogs. These are repository documentation counts, not a fresh
  inventory of released files or independent groups.
- The same README says AVSD data are shared by registration and distinguishes
  registered challenge participants from others who may not access the data.
  The repository's MIT `LICENSE` applies to repository code; it does not
  establish an unrestricted license for AVSD data or source video.
- The AVSD authors describe videos as coming from the Charades dataset. The
  current linked AI2 Charades page and legacy `license.txt` URL redirect to the
  AI2 homepage, so the source owner's current primary license text could not
  be verified from those URLs. Independent dataset documentation reports
  Charades as non-commercial use; retain the source as blocked on rights rather
  than assuming that AVSD's code license covers the video. See the
  [Charades terms recorded by the Hugging Face dataset loader](https://huggingface.co/datasets/HuggingFaceM4/charades)
  and the [NeurIPS 2024 dataset-license table](https://proceedings.neurips.cc/paper_files/paper/2024/file/38ace89a980ead30c6be2b46afc1a5bd-Paper-Datasets_and_Benchmarks_Track.pdf).

## Task fit

The official task is to generate a natural dialog response from dialog history,
question, audio, and video. It is not a supplied-option Decision task. Turning
its free-form answers into fixed option probabilities would require a new
answer-to-choice adaptation and distractor policy; there is no pre-existing
official option-label protocol to validate. The dataset paper argues that
multiple input streams help response generation, but that does not establish
that a small Path A classifier needs both audio and video on a frozen
candidate-choice task. See the [authors' task description](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge/blob/6a5ee8542132ad6634ee02896d7c935b8c447d78/README.md)
and [Meta Research summary](https://ai.meta.com/research/publications/audio-visual-scene-aware-dialog/).

| Gate | Finding | Disposition |
|---|---|---|
| Product-compatible rights | Access registration is described; Charades source terms are reported non-commercial and current primary license URL redirects | Fail closed |
| Group feasibility | README reports 1,787 validation dialogs, but the exact frozen file list and unique-video grouping were not acquired or checked | Unknown |
| Supplied-option task | Native task is free-form response generation | Not established |
| Both-input necessity | Upstream generation claim is not a matched supplied-option AV ablation | Not established |
| Prior exposure | Repository search found no AVSD-specific project artifact before this screen; no corpus payload was opened | Metadata-only finding, not a source-wide clean-data certificate |

## Decision and next action

Do not register, download, or adapt AVSD under the current product-directed
rights gate. Reconsider only if the source owner provides a clear applicable
commercial-use grant and an option-based task with source-grouped IDs and
predeclared unimodal/missing/shuffled controls. This screen provides no quality
evidence and does not authorize R2 fitting.

