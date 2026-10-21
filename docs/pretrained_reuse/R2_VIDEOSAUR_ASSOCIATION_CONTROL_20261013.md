# Sparse mask-overlap association control — 2026-10-13

Completed on unchanged VideoSAUR caches: eight train videos,164 later points.
Only frame0 box initializes the object. Thereafter choose the mask with highest
IoU against the previously selected hard mask; ties retain previous index, then
lowest index. No thresholds, optimization, seed search, training or GPU inference.
The association function cannot access future boxes: it receives masks, frame IDs
and the initial box only. Other boxes are supplied afterward solely for scoring.

| Pooled later-point IoU | Value |
| --- | ---: |
| Fixed original slot |.138612|
| Greedy overlap association |.086843|
| Per-frame oracle reassignment |.341503|

**Reject this sparse overlap association policy.** It worsens this diagnostic.
Sparse cached frames (often30frames apart), motion, disappearance, mask splitting
and merging may contribute; this result does not isolate one cause or establish
that dense-frame association, appearance features or learned tracking cannot work.
Do not call these values Decision Accuracy. No validation or sealed audit used.

Code: scripts/associate_slot_masks.py. Tests cover slot permutation, chronological
frame checks and stable tie handling. Full CPU suite676passed,1skipped;Ruff passed.
Replay after formatting/explicit zip strictness reproduced the same metrics.
Source parent20232e4. Script hash3b3df8f15cb67cd6a65b59021e6d19b890288b547c6e105af87344073eb62dce.
Input result hash4e6b245ab64fb803e16c8cc828926764badb094bdc183a75a20d0a1496b73531.
Annotation and individual mask hashes checked against existing records.

Artifacts: C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-masks-01/
association.json and association-replay.json. Both retain every frame's selected
index, associated/fixed/oracle IoU. Existing outputs were not overwritten.

```powershell
python -m scripts.associate_slot_masks --run C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-masks-01 --annotations C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01/train_annotations.zip --output <fresh-output.json>
```

This closes the cheap cached-mask association hypothesis. Do not sweep matching
heuristics on these same eight videos. Further video progress needs an explicit
appearance/trajectory identity mechanism or a separately justified train-only
Decision representation probe; higher oracle localization alone does not justify
claiming stable temporal objects or deploying this visual path. Full Omni quality,
calibration and product-size gates remain unfinished.
