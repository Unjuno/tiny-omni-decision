# QAT GPU workspace optimization — 2026-10-09

This change is based on PR #14 commit
`c3391a0c689d69df3300686d9c287546b741e7c8`, in branch
`codex/qat-groupwise-performance`. It reduces quantization workspace without
changing the training objective, data, optimizer, scheduler, cache lifecycle,
checkpoint policy, or warm-start protocol. No training run was restarted.

## Follow-up: real-model migration smoke

Before migration, a separate harness loaded the hash-verified step-384
`best-shadow.safetensors` and the pinned local model/processor with the same
eager BF16/deterministic settings. The harness's first attempt omitted the
trainer's `CUBLAS_WORKSPACE_CONFIG=:4096:8` setting and failed before inference;
the harness was corrected to match the existing trainer. No production code or
environment package was changed for that correction.

Four fixed validation examples (first encountered for each modality, retaining
snapshot order) produced exactly the same probability records before/after.
All 483 BF16 shadows matched the saved values after cached validation.
The initial paired runs took 4.538 s / 3.941 s for evaluation plus restoration,
with 4,775 / 3,622 MiB peak PyTorch GPU allocations respectively.

A second paired process run performed one **discarded diagnostic optimizer
update** at the step-385 position: train examples 1,537–1,540 in the original
hash-verified sampler order, accumulation 4, original losses, fresh CPU Adafactor
state and the original cosine LR at step 384. No validation examples were used
for gradients. This is optimizer-reset warm-start behavior, not an exact resume.

| Phase (seconds) | PR #14 reference | Optimized |
| --- | ---: | ---: |
| Four train examples + shadow restoration | 18.522 | 13.701 |
| Gradient clip + gradient transfer to CPU | 1.058 | 1.031 |
| CPU optimizer | 1.656 | 1.753 |
| Master transfer back to GPU | 0.620 | 0.589 |
| Total diagnostic update | 21.856 | 17.074 |
| Peak allocated GPU memory (MiB) | 18,337.66 | 18,342.28 |

Train logits/losses, validation predictions and the hash of **all updated BF16
shadow bytes** matched exactly. Non-target frozen parameters and the source
checkpoint remained unchanged. The updated diagnostic weights were discarded.
Raw records are `benchmarks/qat-update-{reference,candidate}-2026-10-09.json`.
The local harness/logs are retained in `C:/CodexArtifacts/qat-groupwise-performance/`.

This is one update per variant, not a throughput learning curve. While time
improved about 22% in this pair, training peak allocations did not improve;
activation/gradient memory is still material. PyTorch allocated memory exceeding
16 GiB on Windows must not be described as physical dedicated VRAM usage.
Forward/backward and shadow restoration dominate this measurement; it does not
establish that the CPU optimizer is the main bottleneck.

The migration config `embeddinggemma2_ternary_qat_v0_attempt08_workspace_warmstart.yaml`
changes only experiment ID/output directory from attempt05. It retains the
1,215-step full-corpus plan and takes step 384 from the immutable source run via
the explicit warm-start CLI argument. Attempt08 has its own output directory;
its actual execution status must be checked in process/progress evidence.

## Measured hardware and run state

- NVIDIA RTX 3080 Laptop: NVIDIA reports 16,384 MiB dedicated memory.
- Windows DxDiag: RTX dedicated 16,175 MB, shared 16,254 MB; Intel Iris Xe
  dedicated 128 MB, shared 16,254 MB (16,382 MB total display memory).
- Physical system RAM: 34,087,497,728 bytes. The shared figures describe access
  to system RAM; they are not two additional independent VRAM pools.
- Local PyTorch 2.6.0+cu124 / CUDA 12.4 enumerates one CUDA device, the RTX.
  Intel compute offload is not implemented or benchmarked here.
- At inspection, no QAT Python process was present and NVIDIA reported zero
  allocated device memory. Attempt07's last progress remained validation start
  at step 384 / 1,536 examples; no newer completed validation or training step
  was found. Historical `state=running` metadata alone does not prove liveness.
- Attempt05's saved step-384 checkpoint and all existing run artifacts were
  left unchanged. Benchmark inputs are synthetic; no Teacher, corpus, audit
  set, or model checkpoint is loaded.

## Code assessment and implementation

The original quantizer builds an element-sized int64 index tensor and validity
mask, even when all groups are full. Dequantization builds another element-sized
int64 group index and gathers a full-sized scale array. The code also retains
unneeded FP32 buffers while forming ternary codes.

The patch uses one integer count per group, with an explicit final-group count.
Padded values are zero and cannot pass the existing nonzero active predicate.
Dequantization broadcasts one scale per row of grouped codes. Codes are formed
directly in int8 and masked in place; unused FP32 storage is released earlier.
The FP32 reductions, thresholds, finite-value checks and identity STE remain.

PR #14 already caches quantization across four accumulated examples and through
validation. This patch does not alter its cache restoration or exception paths.
CPU FP32 optimizer masters, gradient transfers and the two shadow copies around
an update remain possible costs. Removing those copies needs a separate
correctness and timing study; they currently implement restoration guarantees.
Media decode and model forward/backward costs are also unmeasured here.

## Controlled GPU microbenchmark

`scripts/benchmark_ternary_workspace.py` compares saved reference code with the
candidate in a separate process on an idle GPU: seed 17, BF16 inputs, group size
256, threshold 0.7, two warmups, five alternating trials per method and size.
Times are synchronized wall-clock medians. Extra peak allocated memory excludes
the input tensor and is not total model VRAM. Allocator cache is cleared before
each timed trial. Raw results are in `benchmarks/qat-workspace-2026-10-09.json`.

| Elements | Reference ms | Candidate ms | Reference extra MiB | Candidate extra MiB |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | 0.77 | 0.66 | 1.63 | 1.07 |
| 1,048,576 | 2.56 | 1.38 | 26.11 | 17.09 |
| 16,777,216 | 22.52 | 14.49 | 417.75 | 273.50 |
| 16,777,219 | 22.90 | 16.00 | 417.76 | 273.51 |

At 16,777,216 elements, this is about 36% less time and 35% less temporary GPU
memory. It is **not** a measured improvement in complete QAT update time.
Codes, FP32 scales, FP32 reconstructed values and BF16 fake-quantized outputs
matched reference exactly for every benchmark size, including the partial tail.

The initial baseline intentionally failed the 20% workspace reduction gate.
Removing index arrays alone improved speed but only reduced peak memory about
4%. Releasing scratch buffers and avoiding FP32 zero/where arrays was necessary
to pass that gate for both full and partial groups. Intermediate measurements
remain under `C:/CodexArtifacts/qat-groupwise-performance/`.

## Verification and next migration gate

- Ruff: pass.
- Full CPU suite: **130 passed**, including existing four-microbatch cache/STE,
  validation restoration and warm-start tests.
- Added scalar-oracle tests for noncontiguous inputs, partial groups, group
  size 1, a group larger than the tensor, FP32/BF16/FP16 and STE gradients;
  also verifies gradients to dequantization scales exclude tail padding.
- CUDA workspace gate: pass; more than 20% reduction at both large sizes.
- Model/dataset manifests, four catalogs and example policy audit: pass.
- New CPU tests are included in the QAT regression workflow. Remote CI has not
  been run for this local branch; GitHub CLI authentication is unavailable.

Before a long run on this branch, measure complete update and validation time
with the fixed real model configuration in a new output directory. In particular,
measure decoding, forward/backward, CPU optimizer and CPU/GPU transfer time.
Those measurements determine whether compute or offload is the next bottleneck.
Do not infer a training throughput improvement from the kernel benchmark alone.
Use the verified source checkpoint and optimizer-reset warm-start policy at
migration; these changes do not add optimizer state or exact-resume capability.

To reproduce the benchmark, preserve an unmodified `ternary.py` from the reference
commit outside the working tree, then run on an idle local CUDA GPU:

```powershell
python scripts/benchmark_ternary_workspace.py --reference C:/path/reference-ternary.py --output C:/path/new-result.json --require-improvement
```

The result file records both source SHA-256 hashes, individual trial times and
peak allocations. The benchmark refuses to overwrite an existing result file.
