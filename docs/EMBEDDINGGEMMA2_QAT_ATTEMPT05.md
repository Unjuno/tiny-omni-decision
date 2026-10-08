# EmbeddingGemma 2 ternary QAT — attempt 05 interim record

**Status at 2026-10-08 15:28 JST: running.** This is an interim record, not a
completed training result or a model-selection decision. The process, output
directory, source inputs, existing Teacher artifacts, and prior results remain
untouched. The run has not loaded sealed-audit data.

## Run identity and fixed inputs

- Experiment: `embeddinggemma2-ternary-qat-v0-attempt05-adafactor`.
- Source commit in the run metadata: `762db1263c09977f3905263ce91463a8d7297d3d`.
- Invocation recorded by the live process:
  `python.exe scripts/train_student_ternary_qat.py --config configs/student/embeddinggemma2_ternary_qat_v0_attempt05_adafactor.yaml --repo-root . --teacher-repo-root C:\Users\junny\OneDrive\ドキュメント\ChatGPT\tiny-omni-decision`.
- The exact process working directory was not separately captured; do not infer
  it from the relative CLI paths.
- Output: `C:\CodexArtifacts\embeddinggemma2-ternary-qat-v0-attempt-05-adafactor\`.
- Source and saved effective config SHA-256:
  `c5dbf379fb1ac4b75638ccfbf6669e3974d4c205b550aaa732aef2ddec53084f`.
- Trainer script SHA-256:
  `3e31db2ada47bf78889b3d0492cdf506b776b0c2cb6080a1fa6151f8d0d88490`.
- Run code snapshot hashes are in `run-metadata.json`; source commit and all
  files have not been changed by this record.
- Student initialization: pinned pretrained base plus new ternary-QAT shadow
  weights; no previous attempt adapter or optimizer resume was loaded.
- Student: `google/embeddinggemma-2`, revision
  `914f7f89142e33e77833254d9c9b90c3cef7303b`; model manifest SHA-256
  `f332611f15edf4b637fb37e327d6b13fe3e800ad027e6fc805da71ab6fec0ab6`.
- Training corpus: 4,859 examples; SHA-256
  `c3e6796a2df47471d42e671272b3050f9db150ac6b7ef7289e3902253f36d84d`;
  ordered ID SHA-256
  `d12e2225776fe895efd9c9686742326e494170ee4291506649f75dd70b7afbac`.
- Sampling uses no repeats and equal configured modality weights. The run plans
  1,215 optimizer updates with accumulation 4; the last partial accumulation
  contains three examples, so the full plan consumes 4,859 unique examples.
- Frozen Teacher cache: `tiny-omni-decision-teacher-v0`, revision
  `6befbaca7398925921802abd1f277b495b78b738`, cache SHA-256
  `f2d3cc1a9d579c7fe75f0dfb1e5441b81964a6d41e8efa76ed1cc43bc0d3e0a4`.
  Teacher temperature is 1.0. Student option-score temperature is 0.1.
  Loss weights are option KL 1.0, labeled CE 0.2, and Brier 0.2.
- Validation is the fixed 256-example snapshot (64 examples per modality),
  SHA-256
  `43358f0e4444dabfaba30eba0d9e4af7851f1aff56d71b2edbf38982b1bb6faf`; its
  manifest SHA-256 is
  `7aa1168ce3baa6402e0ed05b21038015c8dcc3607570e942f3a92ad667a9bc08`.
- The frozen initial packed ternary overlay manifest SHA-256 is
  `01e4d9e6b265c149f3aeefc25775f6a1e5d4ed1fb224f345339872edfe5e0997`.
- Optimizer: Adafactor with FP32 CPU master parameters (2,976,518,656 bytes),
  BF16 shadow weights, cosine schedule, 36 warmup updates, learning rate
  `1e-5`, and gradient norm cap 1.0. Early stopping is disabled. The fixed
  selector minimizes macro-modality NLL, then maximizes minimum-modality
  accuracy, then minimizes macro Brier.
- QAT target inventory: 483 tensors / 744,129,664 elements; 241,824
  non-target parameters remain frozen at higher precision.

## Current execution evidence

The live process was PID 78748, launched at 2026-10-08 06:39 JST. At the latest
recorded checkpoint it had completed 128 updates and consumed 512 unique
examples. Validation had run twice (step 0 and step 128); step 128 was the
current best. The latest saved checkpoint, metadata, history, and prediction
files were timestamped 2026-10-08 11:55 JST. The run has not yet produced step
256. No optimizer resume state is saved; if the process exits, exact resume is
not established and a restart must not be described as a continuation.

| Validation point | Macro Accuracy | Minimum modality Accuracy | Macro NLL | Macro Brier | Macro ECE |
|---:|---:|---:|---:|---:|---:|
| Step 0 | 0.1836 | 0.0938 | 1.8786 | 0.8219 | 0.1698 |
| Step 128 | 0.2227 | 0.1250 | 1.8300 | 0.8146 | 0.1357 |

At step 128, modality Accuracy / NLL / Brier / ECE was:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 0.1250 | 2.4110 | 0.9155 | 0.0917 |
| Image | 0.1563 | 2.3190 | 0.9020 | 0.0824 |
| Text | 0.4063 | 1.1382 | 0.6630 | 0.1484 |
| Video | 0.2031 | 1.4519 | 0.7778 | 0.2203 |

The early validation changes are mixed: Text accuracy increased, but its NLL
rose; Video accuracy was unchanged while its NLL worsened. These 256 validation
examples are used for development/checkpoint selection, so they are not a
blind final evaluation. Rolling train loss is not a fixed-example
generalization-gap estimate.

## Hardware and memory interpretation

- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB dedicated memory.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 5.19.0.
- At 2026-10-08 15:28 JST, `nvidia-smi` reported 16,149 MiB used and 27 MiB
  free. Windows GPU Process Memory counters sampled at 15:27 JST reported
  15.78 GiB dedicated and 5.91 GiB shared/non-local usage for PID 78748.
- Run metadata records `torch.cuda.max_memory_allocated()` as
  21,935,046,144 bytes. This allocator statistic is not dedicated VRAM; the
  Windows counters show the process also committed shared system memory.
  Keep dedicated and shared memory separate in the final report.
- Adafactor's FP32 CPU master parameters use about 2.77 GiB of host memory.
  No cloud compute was used; recorded cloud cost is `$0`.

## Not yet established

- Completion of 1,215 updates, final-step checkpoint, or final-step validation.
- Final selected checkpoint, reload equivalence, or export from recovered QAT
  weights.
- Whether the QAT candidate is sufficient to avoid the conditional Recovery
  LoRA stage.
- Final paired evaluation, latency, package size, and final hardware-memory
  profile.
- Any sealed-audit result. The sealed audit remains unopened and must not be
  used for tuning.
