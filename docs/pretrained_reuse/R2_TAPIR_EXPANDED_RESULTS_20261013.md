# TAPIR eight-video train tracking results — 2026-10-13

All eight fixed videos completed; six new runs and two exactly reused prior records. Identity order and reuse equality verified. No selection, fitting, validation or sealed audit. [Protocol](R2_TAPIR_EXPANDED_PROTOCOL_20261013.md).

| Video | TAPIR error | Static error | Inside /7 | Forward seconds |
|---|---:|---:|---:|---:|
| video_10103 | 0.024746 | 0.036967 | 6 | 26.947 |
| video_10186 | 0.094238 | 0.179315 | 5 | 37.113 |
| video_10212 | 0.059455 | 0.397922 | 6 | 66.876 |
| video_10387 | 0.113944 | 0.146670 | 4 | 28.544 |
| video_1087 | 0.098502 | 0.106564 | 5 | 33.657 |
| video_10886 | 0.229135 | 0.153313 | 4 | 37.000 |
| video_10894 | 0.176213 | 0.246454 | 4 | 31.819 |
| video_10903 | 0.024986 | 0.095015 | 6 | 33.916 |

Mean normalized center error0.102652 versus static0.170278;7/8 videos improve, video_10886 worsens. Inside40/56 scored boxes. Summed forward time295.872 seconds including the prior two runs; additional six total231.812 seconds. CPU four threads, FP32; no GPU/cloud cost. This is not end-to-end pipeline latency.

Paired mean error difference TAPIR minus static percentile95 interval [-0.155775,-0.001549],10,000 video bootstrap draws,seed17. Only eight ordered train examples, including two previously examined ones: the interval is descriptive and not a population/generalization claim. No frame-wise independence assumption.

The tracker is promising relative to static coordinates on these clips but still misses16/56 boxes. It needs an oracle initial point and tracks a physical point while annotations score box centers. Neither hidden-object Decision answers nor automatic target discovery, option probabilities or held-out quality are established. Retain as a research representation candidate only. Do not count40/56 as Decision Accuracy or claim95% quality.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-tapir-expanded-tracks-01/`. Full trajectories, per-video uncertainty logits, results and `summary.json` with hashes of plan/results/executed script/trajectories. Prior outputs remain unchanged. Next gate: fix a held-out tracking evaluation and automatic initialization contract before treating this as a deployable Decision input.
