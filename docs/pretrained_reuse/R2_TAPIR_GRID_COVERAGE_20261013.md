# Annotation-independent grid initialization coverage — 2026-10-13

Train-only geometry preflight for replacing oracle TAPIR queries. Predeclared uniform8x8 cell-center grid over the whole normalized image,64 points. Grid construction does not read labels, boxes, target identity or future frames. Annotations are scoring-only; no point is selected or removed based on them. No model inference or training.

Frozen99-train-video manifest and official train archive hashes verified.98 videos have at least one unmasked frame0 occluder box;78 have a grid point in the box;20 annotated videos have no point. One video has no eligible initial box and remains unscored, not counted as a hit. All annotated boxes are covered in the same78 cases. This is78/98 annotated-video coverage, not Decision or tracking Accuracy.

Conclusion:64 fixed grid points do not remove the initialization bottleneck on this corpus. Do not launch a full64-point tracker comparison and pretend it starts from a known target. Denser coverage or an independently licensed automatic localization mechanism needs cost/coverage measurement before Decision integration. Even box coverage does not establish semantic target identification, occlusion handling or track quality. The missing20 videos must not be removed to inflate later quality.

Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-grid64-coverage-01/result.json`, includes all99 rows, fixed coordinates, counts and source hashes. Executed script: `C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01/grid64_coverage.py`. No validation or sealed audit was loaded; existing models/checkpoints remain unchanged.

## Fixed density follow-up

Train-only geometric comparison at16x16(256 points) and32x32(1024 points) covers all98 annotated initial occluder regions in both cases. The same one video lacks eligible initial annotation. Both densities were specified before this comparison; no validation or box-specific grid offsets were used. This proves coverage on these train boxes only, not semantic selection or tracking quality. Full artifacts preserve all99 rows at `perception-occlusion-grid256-coverage-01/` and `perception-occlusion-grid1024-coverage-01/` under the same R2 artifact root.

Next bounded cost probe is running: first32 consecutive frames of fixed train video_10103,256 uniform points,256px RGB, CPU four threads, query chunks16, same frozen checkpoint. It loads no annotations and scores no quality. New output `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-grid256-runtime-01/`; script `C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01/grid256_runtime.py`. No full-corpus tracking launch is authorized by this runtime probe itself; use measured memory/time and keep automatic target identification as a separate unsolved gate.

Cost probe completed successfully:256 points x32 frames, finite tracks[1,256,32,2] and uncertainty[1,256,32],15.245247 seconds CPU forward. No accuracy evaluation. This is a single short clip, not a full-length latency projection or peak-memory benchmark. All output arrays and result.json are saved separately.
