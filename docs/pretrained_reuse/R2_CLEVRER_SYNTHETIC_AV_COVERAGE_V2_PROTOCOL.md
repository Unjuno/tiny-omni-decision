# CLEVRER synthetic AV temporal-coverage v2 protocol

Status: **frozen for annotation preflight; no model fit is authorized by this
protocol file alone.** This continues a narrow synthetic temporal-binding
diagnostic. It does not replace, overwrite, or upgrade the v1 run.

## Rationale and scope

The v1 joint gated LSTM outperformed its audio-only control by 6.91 percentage
points on the v1 development validation, but that validation selected the
checkpoint, the run overfit, and the training corpus had only 1,596 unique
rows. The next controlled change is training coverage. Keep the task, frozen
encoders, readout architecture, optimizer, update budget, seed, and selector
fixed. This measures task-specific training coverage; it is not a new blind
estimate or natural audiovisual result.

The user-initiated objective remains the four-modality local Omni Decision
model. This synthetic task is a bounded Video/LSTM research diagnostic only.
It cannot satisfy text, natural image, natural audio, natural video, or product
quality gates.

## Source and split

- Source: `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`,
  official `train` only, CC0-1.0.
- Candidate pool: the 614 unique scene IDs in the already acquired
  `clevrer-temporal-mix-media-v1/train.jsonl` (1,024 source rows). Before any
  annotations are inspected, all 614 media files were rehashed against the
  source manifest: 759,391,226 bytes; scene ID digest
  `af0d17b0d1a6ec9757838e3dcecbb699803a29f38fd4cf722df4e7845f0ed778`.
- The candidate pool has zero scene-ID and media-SHA overlap with the 401
  scenes used in the prior synthetic AV v1 task. The previous v1 AV train and
  validation examples (401 scenes) become v2 **training data**. This explicitly
  reclassifies the old 79-scene v1 validation as training; those old validation
  rows are never used to select v2.
- Predeclared seed-17 media-only split: 240 of the 614 candidate scenes are
  held out for v2 task-specific development validation
  (scene digest
  `15a9eb46b6d66a6a5416341c285913b5435f50cf157134f7887ca028bd55fc5d`); the
  remaining 374 candidate scenes join the 401 previous AV scenes for 775
  prospective v2 training groups (digest
  `63db88a77a63f79be5e559bfd78e8c011a05504a0a24045a27be657546c57c3b`).
- The 240 candidate validation scenes were drawn from media already used in
  other CLEVRER tasks. Therefore v2 validation is neither media-unseen nor
  blind. It is disjoint from every v1 synthetic-AV scene and carries new
  task-specific AV examples. This limitation must remain prominent in every
  report.
- Only official-train annotations may be read. Do not access official
  validation/test labels or any sealed audit. The official train question JSON
  must match SHA-256
  `11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4`.
- The annotation archive is read with HTTP Range requests pinned to size
  `156761889` and ETag `"957ff21-59db3309a63ea"`. Do not download the full
  archive. Verify every ZIP member CRC and scene/video identity, then save each
  selected annotation's SHA-256. Do not fetch additional videos; use only the
  already present, manifest-verified official-train media.
- The split is immutable. The deterministic event eligibility rule is the
  v1 rule: unique object IDs and unique color/material/shape signatures,
  collision frame 16–111 inclusive, one collision per object pair. Paired
  before/after examples for one collision remain in the same scene split. If
  fewer than 200 validation scenes have eligible events, stop without
  reassigning scenes after inspecting annotations.

## Task and frozen model conditions

For each eligible event, retain v1's same video window, uniquely described
collision pair, scene-level timing shift, and paired question/options. Render
the same 880 Hz, 80 ms Hann-windowed audio cue 16 source frames before and
after the collision. The label is the relative temporal order. Text-only and
video-only controls remain at chance by construction for a complete pair.
The audio cue is synthetic waveform energy, not a pretrained natural-audio
representation.

All non-data conditions remain v1:

- Seed 17; fresh readout initialization. Do not warm-start v1 LSTM weights.
- Frozen VideoPrism-B at
  `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`,
  checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`;
  preserve 16 temporal features of width 768.
- Frozen MiniLM-L3 at
  `sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`,
  checkpoint SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Audio: same 16-bin normalized waveform-energy features; no encoder change.
- Same four matched arms: joint gated LSTM, audio-only LSTM, video-only LSTM,
  and joint mean-pooled control.
- Batch size 64, 2,048 updates, evaluation every 128 updates, LR 0.001, weight
  decay 0.0001, gradient clip 1.0, no performance early stop, and minimum
  validation NLL selector. Record full curves and final checkpoints.

Do not start fitting until annotation/media integrity passes, at least 200
eligible validation scenes remain in the frozen split, enough unique train
events are verified, and all features are built and hash-checked. Use a new
artifact directory. Run locally only; any GPU child must use the cooperative
HotUGPU guard after current process state is confirmed safe. No cloud compute.

## Predeclared comparison and reporting

1. Evaluate the v1 selected joint/audio-only/video-only checkpoints on the new
   v2 task-specific validation once as a transfer reference. This is a new
   development diagnostic and not a blind baseline.
2. Fit all four v2 arms under the unchanged v1 training settings. Report
   train/validation examples and unique scenes, every 128-step curve, selected
   and final metrics, and group-paired uncertainty. Do not select seeds or
   alter the fixed task on these results.
3. Main comparison: v2 joint against matched audio-only on v2 validation.
   Also report Video-only and mean-pooled controls, paired-event success,
   audio replacement, stream missingness, and timing-shift controls.
4. Reusing old v1 validation examples as training and using cross-task-exposed
   media for v2 validation make this a development-only coverage experiment.
   Do not call it blind, source-independent, or product quality; do not open
   sealed audits.
5. Compare v1 and v2 only on the same v2 task-specific validation IDs and
   preprocessing. Treat the v1 result on these rows as transfer evidence;
   v1 was never selected on them for this task.

## Preflight record

The first media-only preflight was stored outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-coverage-v2-preflight-attempt-02\preflight.json`.
Its SHA-256 is recorded after completion in the run receipt. Preflight confirms
only media identity, task-specific split construction, and candidate coverage;
it is not an annotation pass, feature pass, model run, or quality result.
