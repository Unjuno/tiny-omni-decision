# Video Teacher v2 experiment log

## Status

Candidate A's configuration, frame-count plumbing, and full 512-step local
run are complete on `codex/video-teacher-v2`. Its validation improves through
step 512 without a detected overfitting signal, but it is materially worse than
the Teacher v1 reference on the same 472-row validation subset and the same
eight-frame evaluation path. Candidate A is therefore recorded as a rejected
experiment, not a replacement Teacher. Candidate B's decoder-only target
resolver/config are implemented, and its separate 512-step run is now in
progress.

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
are stored beside the run outputs. Candidate B has not yet completed a
scheduled validation point, so no quality conclusion is available.

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

Candidate A is complete and Candidate B is currently training. Keep the B run
at its 512-step budget with the frozen corpus and validation files. B changes
only `lora_target_policy` from `qv` to `decoder_all_linear`; 8 frames, rank 16,
seed 17, fixed `5e-5` LR, warmup 0, constant scheduler, modality/source mix,
and sequence limit remain fixed. The previous user clarification explicitly
fixes the A/B schedule; cosine with 3% warmup is a separate schedule experiment
and is not part of B. After B's final validation and reload check, compare it
with both Candidate A and the re-evaluated v1 reference before deciding whether
the evidence justifies Candidate C. Do not read the sealed audit.

The training loader uses the `pread` backend only on Windows and restores
Transformers' loader after the base weights are loaded. This setting is
recorded in run metadata and does not change optimization or data settings.
The earlier target-only full-weight scan did not complete, while its
pinned-config meta-architecture inventory did; Candidate B training will
resolve and verify all actual target paths against the loaded pinned model.
