# Annotation-independent grid initialization coverage — 2026-10-13

Train-only geometry preflight for replacing oracle TAPIR queries. Predeclared uniform8x8 cell-center grid over the whole normalized image,64 points. Grid construction does not read labels, boxes, target identity or future frames. Annotations are scoring-only; no point is selected or removed based on them. No model inference or training.

Frozen99-train-video manifest and official train archive hashes verified.98 videos have at least one unmasked frame0 occluder box;78 have a grid point in the box;20 annotated videos have no point. One video has no eligible initial box and remains unscored, not counted as a hit. All annotated boxes are covered in the same78 cases. This is78/98 annotated-video coverage, not Decision or tracking Accuracy.

Conclusion:64 fixed grid points do not remove the initialization bottleneck on this corpus. Do not launch a full64-point tracker comparison and pretend it starts from a known target. Denser coverage or an independently licensed automatic localization mechanism needs cost/coverage measurement before Decision integration. Even box coverage does not establish semantic target identification, occlusion handling or track quality. The missing20 videos must not be removed to inflate later quality.

Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-grid64-coverage-01/result.json`, includes all99 rows, fixed coordinates, counts and source hashes. Executed script: `C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01/grid64_coverage.py`. No validation or sealed audit was loaded; existing models/checkpoints remain unchanged.
