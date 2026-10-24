# R2 synchronized audio-video source screen — 2026-10-24

## Scope

Read-only screening of newly surfaced official audio-video reasoning sources
against R2-M's requirement for a task whose labels require more than one
modality. We inspected official project descriptions and published use terms
only. We did not download/open media, annotations, validation answers or test
answers, accept data-use terms, or run a model. No source is admitted by this
screen.

## Results

| Source | Task and split evidence | Rights/access evidence | Decision |
|---|---|---|---|
| KilometerAudio (4th Perception Test Challenge) | The official page describes human-verified five-choice questions specifically designed to require audio and vision. The release has no training split, only 8 validation questions across 6 videos (reused from earlier editions), and 500 new questions across 218 test videos. Those test labels are not opened or eligible for iterative selection. | Videos are public YouTube walking-tour recordings. The official page links a zip but gives no dataset/media license or grant covering underlying video-owner rights. | Strong task-design lead; not a training source. Keep all benchmark answers sealed. Reconsider only for a final audit after rights, contamination, device feasibility, and protocol are resolved. |
| EvaOmni | The official repository describes 1,027 evidence-grounded AV questions from 392 videos, including audio and visual evidence intervals and shortcut checks. The quickstart points at a `test_file.json`; no training split or group-isolated development split was established in this screen. | The code is MIT, but the repository directs dataset users to `docs/data_license.md`; that file returns 404 at the screened revision. Dataset-host and underlying media terms remain unresolved. | Relevant benchmark lead, not admitted for training or model selection. Re-screen only after the data-use document and source-media terms are published. |
| AVI-Bench | The official repository describes 5,864 samples across 14 tasks, including AVQA and cross-modal matching/grounding. | Its Data Use Policy is CC BY-ND 4.0 plus an Anti-Training Addendum: model training, fine-tuning, distillation, alignment and derived training-data construction are prohibited. Each bundled upstream source retains its own media license; access is gated. | Not a training source. Do not access for this development run. It could only be considered as a separately authorized, non-tuning evaluation source after the policy and every upstream media term are reviewed; do not open it as the sealed audit. |

## R2 decision

None of the newly screened sources clears R2-M-2 for development fitting. The
closest task match, KilometerAudio, has no training split and its only sizable
split is the challenge test; its public page does not establish media rights.
EvaOmni's dataset terms are missing, and AVI-Bench expressly bars training.
The primary blockers are rights/split availability, not a lack of model
architectures. No new data license was accepted and no external media was
accessed.

The CC0 CLEVRER/generated-tone path was subsequently exercised in an already
completed v2 coverage run (four matched arms, seed 17, 2,048 updates per arm;
joint temporal LSTM best validation Accuracy 83.51%, NLL 0.3534, Brier 0.2244,
ECE-15 0.0369). A TRAIN-only perturbation audit later found that its joint
readout responds to both synthetic streams and their order, while audio-only
remains a strong shortcut (75.62% TRAIN Accuracy) and video-only is at chance.
These are synthetic-task development and TRAIN-sensitivity results, not
natural AV quality, media-unseen evidence, or a product gate. The full v2
protocol/results and separate TRAIN audit are recorded in the existing
[coverage report and PR #21](https://github.com/Unjuno/tiny-omni-decision/pull/21)
and [TRAIN input-dependency report](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-lstm-train-input-dependency-20261023/docs/pretrained_reuse/R2_CLEVRER_LSTM_TRAIN_INPUT_DEPENDENCY_20261023.md).
Do not duplicate those fits or reopen their observed validation for selection.

Accordingly, none of the currently screened sources enables a new natural-AV
development run. The next quality experiment requires a newly admitted source
with explicit media rights, adequate train/dev groups, and a both-required task;
until then, continue only source/rights and task-definition work. Do not start a
GPU fit on the screened challenge test splits or reuse CLEVRER's exposed
development validation for another tuning cycle.

## Official sources

- [KilometerAudio task, splits and results](https://perception-test-challenge.github.io/kilometeraudio.html), screened 2026-10-24.
- [The 4th Perception Test Challenge](https://perception-test-challenge.github.io/), screened 2026-10-24.
- [EvaOmni official repository](https://github.com/VisionXLab/EvaOmni), screened 2026-10-24; the referenced `docs/data_license.md` path was unavailable (404).
- [AVI-Bench official repository and data-use terms](https://github.com/FudanCVL/AVI-Bench), screened 2026-10-24.
