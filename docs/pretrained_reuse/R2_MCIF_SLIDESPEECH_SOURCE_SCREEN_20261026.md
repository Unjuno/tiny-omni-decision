# MCIF and SlideSpeech source/task screen — 2026-10-26

## Scope

Read-only screen of two newly considered public audio/video corpus candidates
for Path A Decision work. No dataset payload, media, references, or labels were
downloaded or opened; no model was trained or evaluated. The MCIF dataset card
was inspected at pinned Hugging Face revision
`FBK-MT/MCIF@e24065b919758263cfe5d157057278affe76ea7b` and the SlideSpeech
publisher/OpenSLR descriptions were inspected on 2026-10-26.

## MCIF

Official dataset card: [FBK-MT/MCIF](https://huggingface.co/datasets/FBK-MT/MCIF/tree/e24065b919758263cfe5d157057278affe76ea7b).

- The card declares CC-BY-4.0 and describes multilingual audio, video and text
  instruction-following tasks based on scientific talks.
- All four published configurations (`long_fixedprompt`,
  `long_mixedprompt`, `short_fixedprompt`, `short_mixedprompt`) expose only a
  `test` split. The card explicitly says these configurations share the same
  original audio and video files; they are prompt/duration variants, not
  independent data splits.
- QA metadata distinguishes audio-only, visual-only, audio-visual and
  not-answerable types, but the benchmark's QA answers are references rather
  than a supplied-option probability target. Other tasks include recognition,
  translation and summarization.
- The source may be suitable for a separately specified one-time external
  evaluation of a compatible frozen model. It does not supply a train split for
  a supervised Path A Decision readout, and generating distractors from its
  evaluation references would change the benchmark target.

Disposition: **not admitted for training or current Decision-quality evaluation**.
Do not inspect its media or references as an iterative development set. Revisit
only after a task-matched output/metric contract is defined independently of its
test answers; if used, treat the shared source media as one evaluation cohort.

## SlideSpeech

Official sources: [SlideSpeech publisher site](https://slidespeech.github.io/),
[OpenSLR corpus description and downloads](https://www.openslr.org/144/), and
the [published corpus paper](https://arxiv.org/abs/2309.05396).

- The publisher explicitly licenses the **provided metadata** under CC BY-SA
  4.0. OpenSLR describes the corpus under CC BY-SA 4.0, while the paper states
  that the corpus is not commercially available and is intended only for
  academic research. The raw media are collected from YouTube; the inspected
  pages do not establish item-level rights for reuse of each original recording.
- The task is contextual audio-visual speech recognition using slides/OCR and
  speech transcripts, rather than supplied-option Decision probability.
- OpenSLR lists 1,659 training videos, 21 development videos and 25 test videos.
  The listed training media alone are 232.43 GB of video and 88.56 GB of audio;
  development is 887.68 MB of video and 429.96 MB of audio. The publisher site
  says the metadata files include segments, transcripts and OCR results.
- The limited number of held-out source videos and noncommercial/underlying
  media-rights questions make it unsuitable for the current reusable product
  training gate. The large training archives also do not match the local-first
  bounded-data policy.

Disposition: **review-only; no media access or fitting**. Reconsider only if
source-level media rights and the project-use terms are resolved, a Decision
task with option-probability labels is justified, and a source-grouped split
passes preflight.

## R2 consequence

Neither candidate supplies a presently admissible natural AV training cohort
for the required Path A Decision objective. MCIF is test-only and target-
incompatible for supervised fitting; SlideSpeech is an ASR corpus with unresolved
underlying-media terms for this use and a high local acquisition cost. This
screen does not establish that suitable public AV data do not exist. Keep the
R2 natural-AV training/evaluation admission gate open and continue screening
official sources without reusing exposed validation cohorts.
