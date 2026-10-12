# TAPIR local GPU runtime and numeric audit — 2026-10-13

GPU admission checked RTX3080Laptop16GiB,0MiB used/0% utilization; no model training process. Existing unrelated Django processes preserved. Same fixed train video_10103 first32frames, uniform256 queries, chunk16, FP32 checkpoint and preprocessing as prior CPU probe. No labels, fitting or quality evaluation.

| Execution | Forward seconds | Peak allocated bytes | Peak reserved bytes | CPU max track difference (pixels) |
|---|---:|---:|---:|---:|
| CPU reference | 15.245247 | N/A | N/A | 0 |
| GPU backend defaults | 1.364820 | 587769856 | 786432000 | 21.051758 |
| GPU TF32 disabled | 1.041892 | 576538624 | 715128832 | 0.076515 |

GPU timing synchronized around forward and input transfer; model construction/load excluded. These are individual short-clip measurements, not a statistical throughput comparison; do not conclude disablingTF32 is generally faster. No warmup/steady-state latency claim. CPU/GPU outputs are not bit-identical.

Default GPU numeric discrepancy is material. Explicitly disabling both CUDA matmul and cuDNN TF32 and requesting highest float32 matmul precision reduces maximum coordinate discrepancy to0.076515pixels on this clip. Occlusion/expected-distance maximum differences drop from0.393356/0.406553 to0.002048/0.004272. This supports numerical backend settings as a factor, without proving parity on other clips or thresholded downstream decisions. Preserve both attempts; use explicit precision settings for any prospective GPU comparison and validate its task-level behavior.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`: `perception-occlusion-grid256-gpu-runtime-01/` and `perception-occlusion-grid256-gpu-ieee-runtime-01/`, all track/uncertainty arrays and result metadata. Scripts under `C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01/`: `grid256_gpu_runtime.py`, `grid256_gpu_ieee_runtime.py`. Settings are process-local; no environment upgrade or existing artifact modification. No paid compute. Dense-grid Decision quality, semantic target selection and full-length performance remain unmeasured.


## Full pre-reveal clip follow-up

The same fixed uniform 16×16 grid was run on train video_10103 frames 0–458,
ending before the exclusive reveal cut 459. This is the complete pre-reveal
interval, not the complete original video. No annotation-selected queries,
training, validation selection or checkpoint changes were used. GPU admission
was 0 MiB / 0% utilization. The existing model, FP32 precision with TF32 disabled,
256×256 inputs and query chunk size 16 were retained.

- Synchronized forward including input transfer: **11.322870 seconds**.
- Peak allocated: **4,790,118,400 bytes (4.46 GiB)**.
- Peak reserved: **6,106,906,624 bytes (5.69 GiB)**.
- Track shape: `[1,256,459,2]`; uncertainty shapes: `[1,256,459]`.
- All outputs finite; strict checkpoint loading succeeded.
- Actual loaded parameters: **31,072,327**; no paid compute.

This establishes feasibility for this one interval on the local RTX 3080 Laptop.
It does not measure Decision quality, long-clip CPU/GPU parity, steady-state
throughput, or the peak for longer videos. Do not extrapolate it to the whole
corpus without accounting for sequence length. The next unresolved gate is
whether annotation-independent trajectories supply usable condition-dependent
Decision evidence; oracle initialization results do not establish that gate.

New outputs: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-grid256-gpu-fullclip-runtime-01/`.
The directory retains trajectories, uncertainty arrays, result metadata and a
SHA-256 ledger covering those files and the exact script. Execution command:
`python C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01/grid256_gpu_fullclip_runtime.py`.
Source repository commit at execution: `579bcda`. Existing outputs were not
modified. This is an inference measurement, not another comparison training run.


## Ordered trajectory input conversion

`trajectory_features.py` now retains all fixed queries and uniformly samples 32
ordered times, concatenating normalized xy, sigmoid occlusion and sigmoid
expected-distance logits per time. It produces 256×128 features on the saved
459-frame train clip. It never takes annotation boxes, target labels or an
object-selected query mask. Coordinates outside the image remain unclipped;
uncertainty channels are not calibrated Decision probabilities. Short clips
repeat temporal indices explicitly rather than inventing new observations.

Five CPU tests cover temporal order, query permutation, all-query retention,
shape errors and nonfinite rejection. Ruff passes. Real saved inputs were
verified against their SHA-256 ledger before conversion. Output and exact frame
indices: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-trajectory-features-smoke-01/`.
Feature SHA-256: `3a501543f8b895cde2ac1e00189e275b115e737346871185d9953a13796bc0b0`.

This is a reusable input primitive, not a trained Decision head or evidence of
quality improvement. Geometry alone omits appearance needed to associate a
question with its object. A subsequent readout experiment must account for that
limitation and retain mismatch/condition controls; the previous failed pooled
feature probes remain unchanged.
