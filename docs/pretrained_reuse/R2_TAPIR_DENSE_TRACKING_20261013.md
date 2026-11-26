# TAPIR dense temporal diagnostic — 2026-10-13

Same two predeclared train videos, pretrained checkpoint, initial oracle point, spatial256 resize and seven scored annotation times as the sparse diagnostic. Change only temporal inputs: all original frames0..450 and0..630 (451/631 frames), within the existing reveal cuts. No new training, checkpoint selection or validation. Offline bidirectional processing, not causal online tracking.

| Video | Sparse error | Dense error | Static error | Sparse inside | Dense inside | Dense forward seconds |
|---|---:|---:|---:|---:|---:|---:|
| video_10103 | 0.028853 | 0.024746 | 0.036967 | 5/7 | 6/7 | 26.946797 |
| video_10186 | 0.155358 | 0.094238 | 0.179315 | 3/7 | 5/7 | 37.112947 |

Errors are normalized image-coordinate Euclidean distances from predicted points to annotated box centers. Physical-point tracking differs from box-center tracking; point-in-box is reported separately. No frames were filtered by confidence. Sparse inference took0.723/0.512 seconds. CPU four threads, FP32; sampled process working set3.89GB during execution, not a measured peak. No paid compute or GPU used.

Both fixed examples improve with intervening frames, supporting temporal sampling as a factor in this particular tracker diagnostic. Two train clips do not establish generalization, task accuracy or the only bottleneck. The model still misses2/7 boxes on the moving clip and requires an oracle initial target. Its output is not yet a conditional Decision model. Do not substitute this diagnostic for the product task or count frame hits as95% Decision accuracy.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-tapir-dense-tracks-01/` contains full predicted trajectories and results with uncertainty logits. Exact executed script: `C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01/track_dense01.py`. Media, annotation-plan and checkpoint hashes checked; strict checkpoint load. Previous sparse results unchanged. Next quality gate is a broader train-only tracking diagnostic with fixed points and temporal policy, followed by independent validation only after the policy is fixed; automatic query initialization and total resident model budget remain separate unresolved gates.
