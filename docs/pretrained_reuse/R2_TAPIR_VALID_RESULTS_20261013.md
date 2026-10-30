# TAPIR development validation tracking results — 2026-10-13

All16 predeclared videos completed with the frozen model/configuration. No tuning, fitting, exclusions or checkpoint selection during this evaluation. [Protocol](R2_TAPIR_VALID_PROTOCOL_20261013.md).

| Video | TAPIR error | Static error | Inside /7 | Static inside /7 | Forward seconds |
|---|---:|---:|---:|---:|---:|
| video_10016 | 0.032819 | 0.016141 | 6 | 7 | 31.695 |
| video_10029 | 0.018567 | 0.095944 | 7 | 4 | 20.386 |
| video_1003 | 0.068227 | 0.095970 | 5 | 3 | 27.904 |
| video_10039 | 0.055957 | 0.089917 | 4 | 2 | 26.355 |
| video_10110 | 0.062707 | 0.258563 | 6 | 0 | 35.804 |
| video_10120 | 0.034251 | 0.099629 | 6 | 4 | 20.565 |
| video_10305 | 0.184522 | 0.130234 | 2 | 2 | 35.039 |
| video_10319 | 0.149113 | 0.141411 | 2 | 1 | 30.327 |
| video_10326 | 0.023730 | 0.022514 | 6 | 6 | 22.337 |
| video_10336 | 0.075730 | 0.149246 | 5 | 2 | 35.011 |
| video_10338 | 0.004163 | 0.036881 | 7 | 7 | 20.688 |
| video_10344 | 0.009845 | 0.065613 | 7 | 6 | 23.824 |
| video_10347 | 0.140669 | 0.135453 | 3 | 3 | 29.474 |
| video_1049 | 0.233081 | 0.218296 | 3 | 2 | 27.684 |
| video_10543 | 0.044681 | 0.030948 | 5 | 5 | 15.693 |
| video_10572 | 0.135160 | 0.200762 | 5 | 3 | 30.517 |

Mean normalized center error0.079576 versus static0.111720;9/16 videos improve. Predicted point inside79/112 boxes; static inside57/112. Summed forward time433.301 seconds on CPU, four threads, FP32. End-to-end latency and peak memory were not measured for this run.

Video-cluster paired difference(TAPIR minus static) percentile95 interval [-0.062553,-0.006591],10,000 resamples,seed17. Development validation already used in Decision probes; neither blind nor sealed. Small ordered subset and oracle initialization restrict the conclusion to this fixed tracking setup. Frames within each video are not independent.

The mean benefit transfers beyond the eight train clips, but misses33/112 annotated boxes and is not consistently better on every video. This does not meet95–98% Decision quality. Query initialization still uses an annotated center; object semantics, automatic discovery, option probabilities and multimodal fusion are absent. Retain a research-only motion representation candidate; do not ship or begin quantization from these results. Future work must connect a non-oracle input contract to actual fixed Decision tasks without reusing these results as product validation.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-tapir-valid-tracks-01/`: trajectories, uncertainty logits, all results and summary hash ledger. Original train and sparse artifacts unchanged. No GPU/cloud compute, training or sealed-audit access.
