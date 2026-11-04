# MAVERIX and AVCaps source/task screen — 2026-10-31

## Decision

Neither source is admitted for product-directed training or evaluation in the
current workstream. MAVERIX is the strongest task-design lead: its paper reports
2,556 questions over 700 videos, eight-option questions, and explicit
video/audio interdependence checks. But its released media is sourced partly
from Ego4D, and the paper says corresponding videos require accepting the
Ego4D license agreement. No such agreement was accepted, and no data, hidden
labels, videos, or audio were downloaded or opened. AVCaps has modality-specific
and joint captions for 2,061 AV clips, but its Hub card states academic,
non-commercial use only; its captions also do not establish a Decision task
where both inputs are necessary.

## MAVERIX

- Task fit: promising for a future *evaluation* battery. The paper says
  questions are checked with text-only, video-only, and video-plus-subtitles
  ablations and that MCQ distractors target subtle audiovisual details.
- Scope: 2,556 questions from 700 videos, mostly long-form (10 seconds to 63
  minutes), with eight-choice MCQs and open-ended items. This is useful for
  evaluating joint AV reasoning but does not itself supply a training split
  suitable for this project.
- Rights/access: the paper says MAVERIX is MIT but separately states that Ego4D
  videos are under the Ego4D license and are retrieved from the Ego4D site after
  accepting its agreement. MIT on benchmark code/questions does not override
  the media terms.
- Label protocol: the paper says ground truth for the evaluation benchmark is
  not released publicly beyond selected samples. It is therefore not a
  selectable training corpus, and using its held-out test key would be
  inappropriate for model selection.
- Disposition: `REVIEW_NOT_ADMITTED`; do not accept the external license or
  retrieve media/labels. Reconsider only after the user explicitly authorizes
  the applicable terms and a legal/data-use review establishes which split can
  be used for training versus one-time final evaluation.

## AVCaps

- Task fit: 2,061 clips / 28.8 hours derived from VidOR, with separate audio,
  visual, and AV captions. It can support representation/retrieval diagnostics;
  a supplied-option task with demonstrated both-input necessity would still
  have to be separately designed and validated.
- Rights: the dataset's Hugging Face card lists `cc-by-nc-4.0` and explicitly
  says academic research and non-commercial use only. This excludes it from
  the product-directed training path. The underlying VidOR media also retains
  source rights.
- Splits: publisher material reports 1,661 train, 200 validation, and 200 test
  clips; this screen did not verify source-identity grouping, overlap, or
  licensing of every underlying clip.
- Disposition: `NOT_ADMITTED` for current fitting/evaluation. No files or labels
  were accessed.

## What changed for the next action

MAVERIX shows that a strong both-input-required AV *evaluation* design exists,
but its media access terms and hidden evaluation labels prevent using it as the
current training/evaluation source. AVCaps cannot be used under its stated
non-commercial terms. Therefore the R2-M gate remains open; the next source
must have rights compatible with the product-directed path, accessible
training labels, independent source-group splits, and a predeclared
both-input-required option task. No GPU fit is justified by these screens.

## Primary sources

- MAVERIX paper: <https://arxiv.org/html/2503.21699>
- Ego4D access and license instructions: <https://ego4d-data.org/docs/>
- AVCaps dataset card: <https://huggingface.co/datasets/TUT-ARG/AVCaps>
- AVCaps Zenodo record: <https://zenodo.org/records/14536325>
- AVCaps paper: <https://trepo.tuni.fi/bitstream/handle/10024/229331/AVCaps_An_Audio-Visual_Dataset_With_Modality-Specific_Captions.pdf?sequence=1>
