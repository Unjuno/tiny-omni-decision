# R2 CLEVRER synthetic AV coverage v2 results — 2026-10-22

Status: **completed matched development diagnostic; no product promotion**.

## Result in brief

The four frozen-encoder readouts each completed the predeclared 2,048 optimizer
updates on the seed-17 training split. On the same 1,110 task-specific
development-validation examples, the selected joint gated LSTM reached 83.51%
Accuracy, 0.3534 NLL, 0.2244 Brier, and 0.0369 ECE-15 at step 1,792. The
matched audio-only readout reached 75.14%, 0.4577, 0.3193, and 0.0213 at step
1,792. Video-only and mean-pooled controls stayed at chance. A 5,000-replicate
scene-paired bootstrap estimated a joint-minus-audio-only Accuracy difference
of 8.35 points (95% percentile interval 5.89 to 10.83 points).

These measurements support temporal AV interaction **for this synthetic
collision/tone timing task only**. They do not measure natural audio or video
understanding, arbitrary multimodal decisions, image or general text tasks, or
the four-modality product goal. The validation split is task-specific and is
not blind or media-unseen.

## Scope and integrity

- Source: official CLEVRER train data only,
  `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef` (CC0-1.0).
  No official validation/test labels or sealed audit data were opened.
- The earlier synthetic AV v1 scenes were assigned to v2 training, together
  with the new eligible training scenes. The resulting train split has 3,696
  examples across 773 scenes/media assets. The frozen v2 development
  validation has 1,110 paired examples across 240 scenes and 555 collision
  pairs.
- Train/validation scene overlap and media-hash overlap are both zero. The
  sample-ID order hashes are recorded below and checked by the feature loader.
- The task-specific validation scenes were drawn from media already present
  in other CLEVRER tasks. There are 349 normalized question-template overlaps
  across the task rows. Thus the split is scene/media-disjoint from this task's
  training split, but not cross-task media-unseen; template overlap also limits
  how broadly the result should be interpreted.
- CLEVRER has no audio. The sound is a synthetic 880 Hz, 80 ms tone placed
  before or after a collision. The audio input is normalized waveform-energy
  in 16 bins; the video input is frozen VideoPrism features; the text input is
  frozen MiniLM. This is a controlled timing intervention, not natural AV.
- The v1 selected checkpoints were evaluated once on these same v2 validation
  IDs without selection. The v2 models were selected by the frozen minimum
  validation-NLL rule. Because v2 adds training scenes and examples, this is a
  coverage-extension comparison, not an isolated causal estimate of training
  steps alone.

## Execution and artifacts

- Branch: `codex/r2-clevrer-synthetic-av-coverage-20261022`.
- Source commit for the frozen protocol and main v2 run:
  `33256eaf5521b3965a9b2174870a6bfdaf226547`.
- Candidate config:
  `configs/pretrained_reuse/clevrer_synthetic_av_temporal_decision_v2.json`.
  Seed 17; batch size 64; four fixed arms; 2,048 optimizer updates per arm;
  evaluation every 128 updates; LR 0.001; weight decay 0.0001; clip norm 1.0;
  no performance early stopping; best checkpoint chosen by minimum validation
  NLL. Readout width 32; 48,868 trainable parameters per arm. Encoders were
  frozen. This is the trainable readout count, not a total packaged-model
  parameter count.
- Effective-config SHA-256:
  `75f690018eaf912113bbf14d3bcbf66de150033abba11af41a2892858414daaa`.
- Sample manifest SHA-256: train
  `3b323bab2ab52b54bd3d4aa8a14b7bc929c000f440394a8e724a2a88eaa8e9b3`;
  validation
  `7504eb1add69594d8d25df6b32b280e840b31018a5bd5f6e01baf0d01b3a4081`.
  Ordered sample IDs: train
  `977f1388e64c4d063e1496bed3747ee3e0c25639b1c069f4f889505b68686f9e`;
  validation
  `e21ec5bee231100f5949c38685890e452a140f6aade9f1bfd23efabaafd2c997`.
- Combined 16-frame VideoPrism feature cache SHA-256:
  `25d7a0f0cde2697a7a1644716e049575b3148500852c865e30007243d2c33f4f`
  (1,013 media assets; 401 reused v1 assets plus 612 newly extracted assets).
  New extraction was CPU-only, took 1,704.7 seconds, and reached a 4,199,313,408
  byte peak working set. VideoPrism revision is
  `be719a406d563b66f0ac969e7c94bab8e997c81a`, checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Decision feature cache SHA-256:
  `0b146fa637049d84ad88f0a71638cf30f6e967bd66add13ead0ab062bcbebdff`;
  feature report SHA-256
  `d73eb35fa67c031102c20fa9f90718a220b1a9374bcf087468f67a30e0420645`.
  Shapes: video `[4806,16,768]`, audio `[4806,16,1]`, text `[4806,384]`.
- v1 transfer report SHA-256:
  `f7bd5d0c4c13a0f34dd2dd574b9b579c7f805af340a076c5705482d2c4f2a2c4`.
- Main results SHA-256:
  `3d9545e9fc5b9abaccffbc463ba1f2d3f5a2790d0ed3cfa37b8a4341310cc8ff`;
  run-metadata SHA-256
  `fb41265749f7405e74b26ca108e533717201b728b362f8e994cc2d79938ae8fc`.
  Both files, all candidate histories, best/final checkpoints, and per-example
  predictions are in
  `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-temporal-decision-v2-coverage-attempt-01`.
- Start/end: 2026-10-21 17:49:57 to 17:51:04 UTC. The HotUGPU guard waited
  while GPU PID 7324 was active, then observed a free/idle device, reserved
  3,200 MiB, and launched the child with 13,015 MiB free at the gate. No process
  was manually stopped. GPU: NVIDIA GeForce RTX 3080 Laptop, 16 GiB; Python
  3.11.9; PyTorch 2.6.0+cu124; CUDA 12.4; cuDNN 9.1.0; Windows WDDM driver
  616.92. CUDA allocator emitted an unsupported `expandable_segments` warning;
  the run completed. Reported per-arm peak PyTorch allocated VRAM was 607.8 MiB
  (618.7 MiB for mean-pooled control). Total optimizer updates: 8,192; measured
  training time was about 60.4 seconds, plus about 1.8 seconds of evaluations;
  compute cost: $0. No cloud compute was used.
- Output attempt 01 is preserved as the failed first v1-transfer invocation;
  it created an incomplete directory after the runner exposed a missing
  `mean_pooled=False` argument. The evaluation code now specifies that flag and
  a regression test covers the call. The successful transfer output is attempt
  02. The training run itself completed without that issue.
- Initial smoke invocation resolved an unrelated installed package from a
  different worktree. Re-running with this checkout's `src` and root explicitly
  on `PYTHONPATH` passed; no data or model was changed by the failed smoke.

## Matched v1 transfer on v2 validation

All metrics below use the same 1,110 v2 task-specific validation examples and
identical cached features. v1 checkpoints were not selected on this validation.

| v1 checkpoint | selected v1 step | Accuracy | NLL | Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint LSTM | 512 | 78.65% | 0.4441 | 0.2934 | 0.0494 |
| audio-only LSTM | 256 | 75.32% | 0.4775 | 0.3279 | 0.0477 |
| video-only LSTM | 768 | 50.00% | 0.6931 | 0.5000 | 0.00005 |

## Candidate v2 results

All arms processed 130,512 examples per run and encountered every unique
training example (3,696; 773 scenes/assets). The 130,512 count reflects the
smaller final batch of each 3,696-row epoch: 35 complete passes plus 18 batches
of the next pass. It is not the number of distinct examples. Metrics in the
“best validation” columns belong to the checkpoint selected by minimum
validation NLL; final columns are the fixed 2,048-update endpoint.

| Candidate | Best step | Best Accuracy | Best NLL | Best Brier | Best ECE-15 | Final Accuracy | Final NLL | Final train Accuracy / NLL | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint LSTM | 1,792 | 83.51% | 0.3534 | 0.2244 | 0.0369 | 83.33% | 0.3711 | 89.10% / 0.2400 | 607.8 MiB |
| audio-only LSTM | 1,792 | 75.14% | 0.4577 | 0.3193 | 0.0213 | 73.87% | 0.4579 | 75.78% / 0.4438 | 607.8 MiB |
| video-only LSTM | 1,664 | 50.00% | 0.6931 | 0.5000 | 0.0007 | 50.00% | 0.6932 | 50.00% / 0.6932 | 607.8 MiB |
| joint mean-pooled control | 1,664 | 50.00% | 0.6931 | 0.5000 | 0.00006 | 50.00% | 0.6932 | 50.00% / 0.6932 | 618.7 MiB |

### Joint learning curve

| Updates | Consumed examples | Train Accuracy / NLL | Validation Accuracy | Validation NLL | Validation Brier | Validation ECE-15 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 32,640 | 79.55% / 0.4062 | 78.02% | 0.4399 | 0.3005 | 0.0570 |
| 1,024 | 65,264 | 80.25% / 0.3821 | 76.85% | 0.4261 | 0.2916 | 0.0351 |
| 1,536 | 97,888 | 86.07% / 0.2968 | 82.79% | 0.3823 | 0.2442 | 0.0610 |
| 1,920 | 122,352 | 86.85% / 0.2765 | 82.61% | 0.3659 | 0.2334 | 0.0480 |
| 2,048 | 130,512 | 89.10% / 0.2400 | 83.33% | 0.3711 | 0.2338 | 0.0517 |

Validation NLL selected step 1,792 (0.3534), rather than the endpoint. Train
NLL continued to fall while validation NLL rose from 0.3534 to 0.3711 and the
train/validation Accuracy gap reached 5.76 points. This is evidence of late
overfitting for the joint arm. The per-128-step history files preserve all
curves and are in the external run directory.

### Paired scene uncertainty and controls

The fixed comparison paired the joint and audio-only candidate predictions by
the 240 validation scenes. With 5,000 bootstrap replicates and seed 17, the
joint-minus-audio-only Accuracy difference was +8.35 points (95% percentile
interval +5.89 to +10.83); audio-only-minus-joint NLL difference was 0.1185
(95% interval 0.0841 to 0.1512), and Brier difference was 0.1035 (95% interval
0.0834 to 0.1242). These intervals describe this task-specific development
split and are subject to checkpoint selection on that same validation.

The video-only and mean-pooled controls remained at 50% with near-uniform
probabilities. The paired task intentionally uses identical text for each
opposite-label pair, so text-only has a constant chance baseline by
construction. The result indicates that preserving the temporal audio cue and
its relation to the collision features matters for this synthetic benchmark;
it does not show that the model understands natural AV content.

## Checks, issues, and next gate

- CPU smoke passed with 4,806 rows, expected feature shapes, no split overlap,
  and no sealed audit access.
- Full `ruff check .` passed. Full CPU suite: **2,535 passed, 11 skipped**;
  targeted temporal/transfer subset: **17 passed**. All 27 CI manifest,
  catalog, and audit commands passed.
- A transfer-evaluation runtime bug (`mean_pooled` was omitted) was fixed and
  covered by a regression test; the completed transfer evaluation used the
  fixed code. The source model and v1 artifacts were read-only.
- The v2 synthetic-AV diagnostic is complete as specified; it is not a
  product-model pass. Do not convert this result into a natural AV quality
  claim. Any next quality step should use a permitted, truth-bearing natural
  multimodal task with group-disjoint validation and an explicit claim about
  which streams are necessary.

All v1 checkpoints, original v1 data, prior v2 artifacts, and unrelated runs
were left unchanged. No model checkpoint, media, or feature cache is committed
to Git. The result report and code fix are committed separately from the
protocol; push and exact-commit CI status are recorded in the follow-up below.
