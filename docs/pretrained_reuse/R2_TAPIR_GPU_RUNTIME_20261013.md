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
