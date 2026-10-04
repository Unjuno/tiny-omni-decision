# Video Teacher v2 experiment log

## Status

Candidate A's configuration, frame-count plumbing, and full 512-step local
run are complete on `codex/video-teacher-v2`. Its validation improves through
step 512 without a detected overfitting signal, but it is materially worse than
the Teacher v1 reference on the same 472-row validation subset and the same
eight-frame evaluation path. Candidate A is therefore recorded as a rejected
experiment, not a replacement Teacher. Candidate B's separate 512-step run is
complete; its validation-selected checkpoint is below Teacher v1 on aggregate
and weakest-modality accuracy, so it is not a replacement Teacher. Candidate C
has started as a separate rank-32 + rsLoRA capacity experiment. Its first two
scheduled validations show small improvement through step 256, but remain below
Candidate B and Teacher v1; the fixed run continues to the remaining scheduled
checkpoints.

The first two Candidate A attempts are retained in their own output
directories. One failed before training with a Windows access violation
(`0xC0000005`) during Transformers 5.6.2 model loading, where the default
safetensors mmap path reached `torch.storage` tensor materialization. Disabling
asynchronous loading alone did not resolve it. A process with the Windows
`pread` backend loaded the full pinned model successfully with default async
loading: 5,104,297,504 parameters, BF16 on `cuda:0`, and 10,209,116,672 bytes
peak allocated VRAM. This matches the upstream Windows mmap issue and merged
`pread` fix ([issue #48285](https://github.com/huggingface/transformers/issues/48285),
[PR #48341](https://github.com/huggingface/transformers/pull/48341)). The
second attempt stopped on a full disk after recording its baseline only. The
successful run used the Windows-only `pread` loader and is stored separately.
Dataset expansion, ternary conversion, and sealed-audit evaluation have not
started.

Teacher v1 remains the fixed reference. Its selected run used
`configs/decision/teacher_v1_weak_modalities.yaml`; SHA-256
`a68fae06267078dbbbcc8afc0ce4680cb553ed28b968e64e2bde50955c5a6409` still
matches both the Teacher v1 manifest and
`artifacts/tiny-omni-decision-teacher-v1/selected/training-config.yaml`. The
original config, train/validation corpus, selected adapter, learning curves,
selection lock, and sealed-audit files were not edited or reloaded. A new
`teacher_v1_explicit_reference.yaml` spells out the effective v1 runtime values
for comparison without changing the historical run bytes.

## Candidate A

| Setting | Value |
|---|---|
| Artifact ID | `tiny-omni-decision-teacher-v2-candidate-a` |
| Config | `configs/decision/teacher_v2_candidate_a.yaml` |
| Config SHA-256 | `9eea882d00f10746a0a1190c1dfac67b8ed7f213127fb52f98d1dd5d0e97a81d` |
| Video frames | 8 |
| LoRA policy / rank | q/v / 16 |
| Learning rate | fixed `5e-5` (same as v1) |
| Training data | frozen Teacher v1 train JSONL |
| Validation data | frozen Teacher v1 validation JSONL; same selection code |
| Train SHA-256 | `5a9d8f3c1608b9cedef0c4c4b1513dcf5e1face4a67acc99c8002849cf07fb2b` |
| Validation SHA-256 | `ebfa46399f3139f4c97a6e126b3036e00afc274de9433fdb5755c089bb9995a6` |
| Base / processor revision | `google/gemma-4-E2B-it-qat-q4_0-unquantized` at `6befbaca7398925921802abd1f277b495b78b738` |

### Candidate A measured outcome

The successful run is
`artifacts/tiny-omni-decision-teacher-v2/candidate-a/seed17-512-pread-retry-20261004T1205Z`.
Its invocation overrides the config's experiment ceiling with 512 optimizer
steps; it uses seed 17, four microbatches per optimizer step, the v1 selected
adapter as the reference initialization, and no audit records. The directory
contains `run-metadata.json`, `experiment-manifest.json`, `learning-curves.jsonl`,
`training-history.jsonl`, all four scheduled validation reports, validation
predictions, the selected adapter, and checkpoints. Candidate A's best
validation checkpoint is step 512, and save/reload validation predictions
matched. The adapter SHA-256 is
`3c4168c03f173378a250732aca0683be87e5104f8b4930c1c0fc65811d113e40`.

| Step | Train CE | Rolling train Accuracy | Validation Accuracy | Validation NLL | Minimum modality Accuracy | Video Accuracy | Train-minus-validation Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.525 | 0.479 | 0.597 | 1.076 | 0.407 | 0.407 | -0.119 |
| 256 | 1.052 | 0.582 | 0.617 | 1.012 | 0.415 | 0.415 | -0.034 |
| 384 | 0.983 | 0.611 | 0.642 | 0.911 | 0.432 | 0.432 | -0.031 |
| 512 | 0.898 | 0.656 | 0.672 | 0.893 | 0.466 | 0.466 | -0.015 |

At every scheduled point, train CE fell while validation NLL also fell; none of
the configured overfitting signals fired. The train Accuracy is measured over
the most recent 512 consumed examples, so the displayed gap is descriptive,
not a fixed-example generalization estimate.

The selected candidate is compared with Teacher v1 re-evaluated on the same
validation subset and eight-frame processor path:

| Metric | Teacher v1 reference | Candidate A step 512 |
|---|---:|---:|
| Macro Accuracy | 0.737 | 0.672 |
| Minimum modality Accuracy | 0.593 | 0.466 |
| Macro NLL | 0.706 | 0.893 |
| Macro Brier | 0.356 | 0.434 |
| Macro ECE | 0.082 | 0.106 |

| Modality | v1 Accuracy / NLL / Brier / ECE | Candidate A Accuracy / NLL / Brier / ECE |
|---|---:|---:|
| Audio | 0.907 / 0.376 / 0.136 / 0.052 | 0.864 / 0.429 / 0.162 / 0.047 |
| Image | 0.712 / 0.765 / 0.381 / 0.102 | 0.669 / 1.067 / 0.466 / 0.103 |
| Text | 0.737 / 0.633 / 0.356 / 0.054 | 0.686 / 0.750 / 0.412 / 0.092 |
| Video | 0.593 / 1.052 / 0.550 / 0.120 | 0.466 / 1.328 / 0.696 / 0.182 |

The eight-frame training run improved over its bare-base initialization, and
Video rose from 0.407 at step 128 to 0.466 at step 512. However, the v1
reference was stronger in all four modalities on this same evaluation path;
Video fell by 0.127 Accuracy and its NLL, Brier, and ECE all worsened. This
candidate must not be selected as the product Teacher. Candidate B remains a
controlled follow-up to test whether decoder target coverage changes that
outcome; a different frame count or optimizer schedule is not being folded
into B.

The run consumed 2,048 unique examples with zero repeated examples and 1,381
unique underlying assets (audio 298, image 282, text 505, video 296). Source
counts were Speech Commands 341, CLEVR-4 512, Open-Jev 256, Typed Decisions
Synth 256, and CLEVRER 683. Candidate A has 2,678,784 trainable parameters and
a 10,729,912-byte adapter. On the local NVIDIA RTX 3080 Laptop GPU it used
11,662,924,288 bytes peak allocated VRAM, averaged 4.396 seconds per optimizer
step, spent 37.5 minutes in optimizer steps, and took 69.6 minutes wall time.
The frozen train SHA-256 is
`5a9d8f3c1608b9cedef0c4c4b1513dcf5e1face4a67acc99c8002849cf07fb2b`; validation
SHA-256 is
`ebfa46399f3139f4c97a6e126b3036e00afc274de9433fdb5755c089bb9995a6`. The
run also records config, corpus-manifest, base-weight, checkpoint, adapter,
and environment hashes/versions. The sealed audit remained unopened.

### Processor smoke

The smoke used one existing **training** video (`MIT-IBM/CLEVRER:638:0`, media
SHA-256 `7d4255a20ce41be2c775ff03e1d2a8d4741e19f44abf488a7ed5291af99b4c2f`) and
the local pinned processor. It did not load model weights, train, or touch
validation or audit records.

```text
video_num_frames from Candidate A config: 8
pixel_values_videos: [1, 8, 630, 768]
video_position_ids: [1, 8, 630, 2]
input_ids: [1, 610]
configured maximum sequence length: 1024
result: 610 <= 1024 (pass)
processor time after load: 1.313 s
```

The local processor emitted a warning that `torchcodec` is not installed and
fell back to the available `torchvision` decoder. The fallback produced all
eight frames successfully; its deprecation warning is retained as an
environment limitation. The machine is the local RTX 3080 Laptop 16 GiB, but
this processor-only smoke used CPU and consumed no GPU training time.

The detailed smoke record is stored locally at
`artifacts/tiny-omni-decision-teacher-v2/candidate-a-smoke/processor-smoke.json`
and is ignored by Git, like model weights and other generated artifacts.

## Candidate B target inventory

Candidate B is configured in `configs/decision/teacher_v2_candidate_b.yaml`
(SHA-256
`a8c34f3929a4d38d3910221ec56bffc5fd8cd55cdee0d161b96b79dba822bd92`). It
keeps A's 8 frames, rank 16, seed 17, fixed `5e-5` LR, warmup 0, modality
weights, train/validation corpus, and sequence limit; only the target policy
changes from qv to `decoder_all_linear`.

The loaded v1 adapter config confirmed actual decoder paths such as
`model.language_model.layers.4.self_attn.v_proj`. A first attempt to load all
model weights for a target-only scan exited with status 1 before emitting a
traceback and left GPU allocation at 11 MiB; no weights, audit data, or v1
artifact were changed. To avoid repeating a large load without diagnostics, the
inventory below instantiates the exact pinned `AutoModelForMultimodalLM`
architecture from its cached, revision-pinned config under
`accelerate.init_empty_weights`. The training path still resolves and checks
targets against the fully loaded model before PEFT injection.

| Policy | Actual module paths | Rank-16 trainable parameter estimate | BF16 adapter-weight estimate |
|---|---:|---:|---:|
| qv | 50 (35 q, 15 v) | 2,678,784 | 5,357,568 bytes |
| attention | 100 (35 q, 15 k, 15 v, 35 o) | 5,357,568 | 10,715,136 bytes |
| decoder_all_linear | 205 (35 each q/o/gate/up/down, 15 each k/v) | 24,158,208 | 48,316,416 bytes |

All 205 Candidate B target paths begin with
`model.language_model.layers.`. The resolver matches only actual linear
modules with the seven explicit decoder leaf names; no generic model-wide
selector is used. Counts and every resolved path are retained in the local
`artifacts/tiny-omni-decision-teacher-v2/candidate-b-target-smoke/target-resolution.json`.
This is a meta-architecture count, not a measured PEFT artifact size or VRAM
measurement; both will be recorded during training.

### Candidate B training run

The run started with seed 17, a 512-step budget, and output directory
`artifacts/tiny-omni-decision-teacher-v2/candidate-b/seed17-512-pread-20261004T222136`.
Its experiment ID is `20261004T132136Z-734d9b4f`; source commit at launch was
`096f521085fa31dca1f9487420bdef6c22b5683d`. The run has passed the fully
loaded-model target-path check and entered training. It holds Candidate A's
fixed LR `5e-5`, constant scheduler, zero warmup, frame count 8, rank 16, data,
sampling policy, and validation set constant; decoder target coverage is the
only experimental factor changed. Effective invocation and config snapshots
are stored beside the run outputs. The run completed 512 steps, selected step
384 by the frozen validation rule, and verified identical validation
predictions after adapter reload. It used 24,158,208 trainable parameters and
produced a 96,693,360-byte adapter (measured, not the BF16 estimate in the
target inventory). Peak allocated VRAM was 12,006,595,072 bytes; mean
optimizer-step time was 6.195 s, optimizer time was 52.9 minutes, and wall time
was 91.2 minutes.

| Step | Train CE | Rolling train Accuracy | Validation Accuracy | Macro NLL | Macro Brier | Macro ECE | Minimum modality Accuracy | Video Accuracy | Train-minus-validation Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.245 | 0.566 | 0.640 | 0.908 | 0.442 | 0.089 | 0.415 | 0.415 | -0.073 |
| 256 | 0.927 | 0.635 | 0.674 | 0.899 | 0.424 | 0.098 | 0.441 | 0.441 | -0.039 |
| 384 | 0.943 | 0.633 | 0.667 | 0.779 | 0.395 | 0.086 | 0.415 | 0.415 | -0.035 |
| 512 | 0.862 | 0.658 | 0.682 | 0.802 | 0.396 | 0.082 | 0.466 | 0.466 | -0.024 |

At step 512, training CE fell and Accuracy rose while validation NLL rose from
0.779 to 0.802; that scheduled point fired `validation_nll_rising`. Step 384
remains selected by the predefined score (macro NLL + 0.2 macro Brier + 0.1
macro ECE - 0.25 macro Accuracy - 0.25 minimum-modality Accuracy). The
`weak_modality_degraded` guard also fired at step 384 when Video fell from
0.441 to 0.415. Train Accuracy is measured over the most recent 512 consumed
examples, so the displayed gap is descriptive rather than a fixed-example
generalization estimate.

Selected Candidate B step 384 versus Teacher v1 re-evaluated on the same
validation subset and Candidate A's selected checkpoint:

| Metric | Teacher v1 reference | Candidate A step 512 | Candidate B step 384 |
|---|---:|---:|---:|
| Macro Accuracy | 0.737 | 0.672 | 0.667 |
| Minimum modality Accuracy | 0.593 | 0.466 | 0.415 |
| Macro NLL | 0.706 | 0.893 | 0.779 |
| Macro Brier | 0.356 | 0.434 | 0.395 |
| Macro ECE | 0.082 | 0.106 | 0.086 |

| Modality | v1 Accuracy / NLL / Brier / ECE | Candidate B Accuracy / NLL / Brier / ECE |
|---|---:|---:|
| Audio | 0.907 / 0.376 / 0.136 / 0.052 | 0.864 / 0.416 / 0.160 / 0.036 |
| Image | 0.712 / 0.765 / 0.381 / 0.102 | 0.653 / 0.879 / 0.413 / 0.118 |
| Text | 0.737 / 0.633 / 0.356 / 0.054 | 0.737 / 0.662 / 0.375 / 0.077 |
| Video | 0.593 / 1.052 / 0.550 / 0.120 | 0.415 / 1.161 / 0.631 / 0.113 |

Decoder-wide target coverage substantially improved loss and calibration over
A, but the selected Video Accuracy was 0.415, below both v1 (0.593) and A's
selected checkpoint (0.466). Candidate B's final scheduled point reached
Video 0.466 and macro Accuracy 0.682, but its validation NLL rose and it was
not selected. This shows that rank-16 decoder-wide capacity still does not
reliably improve Video; Candidate C's rank-32 + rsLoRA run is justified as the
next isolated capacity experiment. The remaining limitation may still be
temporal representation rather than adapter capacity.

Candidate B consumed 2,048 unique examples with zero repeated examples and
1,381 unique underlying assets (audio 298, image 282, text 505, video 296),
using the same per-source counts as A. The selected adapter SHA-256 is
`28d301401f7f51ec72a288e85e9395fa894fae52a196c9018e55c55afd76ecb9`. Config
SHA-256 is
`a8c34f3929a4d38d3910221ec56bffc5fd8cd55cdee0d161b96b79dba822bd92`; train
and validation corpus hashes match Candidate A. Sealed audit data remained
unopened.

### Candidate C training run

Candidate C uses `configs/decision/teacher_v2_candidate_c.yaml` (SHA-256
`792227ce687e4a2f569870bcc463484a7fa456441314ad4972aead319db1a084`) and a
new output directory:
`artifacts/tiny-omni-decision-teacher-v2/candidate-c/seed17-512-pread-20261004T235936`.
Its experiment ID is `20261004T145937Z-58ce76dd`; source commit at launch was
`611c750b2337f4af8b603fcd4fd9dbd43b2546f5`. It keeps Candidate B's rank-16
run settings, data, video frames, `decoder_all_linear` target policy, LR,
scheduler, warmup, sampling, seed, and 512-step budget fixed while raising rank
to 32 and enabling rsLoRA. The loaded-model path check passed for the 205
decoder-linear targets, the v1 reference was evaluated on the same validation
subset, and Candidate C has entered training. Its first scheduled validation
at step 128 completed after 323.6 seconds. The 512-example rolling training
window had CE 1.288 and Accuracy 0.551 (train-minus-validation Accuracy gap
-0.081). Validation had macro Accuracy 0.631, minimum-modality Accuracy 0.398,
macro NLL 0.995, macro Brier 0.456, and macro ECE 0.113. Per-modality
Accuracy / NLL / Brier / ECE was:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 0.890 | 0.453 | 0.154 | 0.041 |
| Image | 0.627 | 1.234 | 0.502 | 0.129 |
| Text | 0.610 | 0.858 | 0.476 | 0.151 |
| Video | 0.398 | 1.436 | 0.692 | 0.129 |

At step 256, the next 512-example training window had CE 1.158 and Accuracy
0.572 (train-minus-validation Accuracy gap -0.070). Validation had macro
Accuracy 0.642, minimum-modality Accuracy 0.407, macro NLL 0.992, macro Brier
0.459, and macro ECE 0.102. Per-modality Accuracy / NLL / Brier / ECE was:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 0.907 | 0.413 | 0.145 | 0.051 |
| Image | 0.619 | 1.296 | 0.497 | 0.126 |
| Text | 0.636 | 0.844 | 0.490 | 0.090 |
| Video | 0.407 | 1.412 | 0.702 | 0.141 |

The second evaluation took 333.1 seconds. From step 128 to 256, macro Accuracy
rose 0.631 to 0.642, minimum-modality Accuracy 0.398 to 0.407, and macro NLL
improved 0.995 to 0.992; the selected validation score improved from 0.8404
to 0.8313. Training CE fell while validation NLL also fell. Of the explicit
signals, only `train_loss_falling` was true; validation NLL was not rising,
training Accuracy did not rise while validation stalled, and the weakest
modality did not degrade. Candidate C step 256 remains below Candidate B step
256 (macro Accuracy 0.674, minimum-modality Accuracy 0.441, macro NLL 0.899)
and the re-evaluated v1 reference. This is modest progress, not grounds for
selecting C early. Continue the already-running 512-step configuration
unchanged through its scheduled checkpoints. Its run remains active after step
256; resource totals and final selection/reload verification are pending.
Invocation and config snapshots are stored beside the run outputs; the sealed
audit remains unopened.

## Verification so far

- `ruff check src tests`: passed.
- Full CPU test suite after the Windows loader regression tests: **102 passed**.
- Candidate A config regression test: changing only `video_num_frames` from 4
  to 8 leaves the effective training configuration unchanged otherwise.
- Recording processor test: receives `videos_kwargs.num_frames == 8`.
- Real processor smoke: produced an eight-frame tensor and stayed below the
  sequence-length guard.
- Candidate B target-policy tests: qv, attention, and decoder-all-linear select
  only the intended decoder layer names; candidate B differs from A only in
  target policy. The test suite passed at 100 tests before the Windows loader
  regressions brought the total to 102.
- Pinned-config meta-architecture scan: all seven requested linear leaves
  exist; all selected paths stay under the language-model decoder.

## Next gates

Candidates A and B are complete and Candidate C is running. Keep C's settings
fixed through 512 steps, scheduled validation, checkpoint selection, and
save/reload verification. The step-512 NLL reversal and Video
accuracy plateau/variance in B support the rank-32 + rsLoRA capacity test; A/B
show that broader target coverage alone has not closed the Video gap. Compare
C with A/B and the re-evaluated v1 reference on validation. Cosine with 3%
warmup and any frame count change remain separate later experiments. Do not
read the sealed audit.

The training loader uses the `pread` backend only on Windows and restores
Transformers' loader after the base weights are loaded. This setting is
recorded in run metadata and does not change optimization or data settings.
The earlier target-only full-weight scan did not complete, while its
pinned-config meta-architecture inventory did; Candidate B training will
resolve and verify all actual target paths against the loaded pinned model.
