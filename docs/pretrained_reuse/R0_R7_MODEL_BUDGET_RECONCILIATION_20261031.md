# Path A plus sensor parameter-budget reconciliation — 2026-10-31

## Read-only artifact evidence

The v4 task-scoped Path A package manifest was read from
`C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4\package-manifest.json`.
Its SHA-256 is
`79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
It reports 170,499,716 unique model parameters and 687,254,772 package-file
bytes excluding the package manifest. Its flags explicitly report a standalone
text task and temporal-video task, but `supports_joint_audio_video=false`.
The package is an assembly of separate task components and runtime profiles,
not one tested end-to-end model.

The preserved subject-disjoint HARTH receipt at
`C:\CodexArtifacts\pretrained-reuse-r2\sensor-subject-training-01\result.json`
reports 6,700 parameters for its temporal Conv1d readout. Its best checkpoint
SHA-256 is
`74c887f665d5637b985b6b839cbcddc53acedb9a75be1bf35c1232092afe2ace`.
The selected validation Accuracy is 0.9147, but the held-out validation has
only three participants, was used for selection, and has substantial class
imbalance; this is not a product sensor-quality result.

## Arithmetic and interpretation

| Inventory | Parameters | Model-file bytes | Scope |
|---|---:|---:|---|
| Path A v4 task-component union | 170,499,716 | 687,254,772 | Audio, image, synthetic text, video-feature tasks; no joint AV |
| Separate temporal HARTH sensor readout | 6,700 | 29,162 (`best.pt`) | Subject-disjoint narrow sensor task; validation is selection-affected |
| Simple unique-parameter/file union | **170,506,416** | **687,283,934** | Arithmetic only; no integrated package or simultaneous residency measured |

On this provisional accounting, a small sensor readout fits within a 200M
pre-quantization union, leaving 29,493,584 parameters of numerical headroom.
This weakens the case for adding a 35.3M MOMENT embedding tower solely to
obtain a sensor pathway: doing so on top of the Path A union would exceed 200M.
It does not prove the small sensor readout is suitable for the product task.

## Gates that remain open

- The v4 component union does not establish a single common model process,
  unified calibrated output space, or compatible runtime dependencies.
- It does not implement the missing screen-only task path.
- It explicitly lacks joint AV; the current R2-M source/task gate remains open.
- Audio, image, text, video, and sensor metrics are narrow task-specific
  results, far below or not comparable to the 95–98% per-task target. HARTH
  uses exposed, selection-affected development participants.
- The 687 MB file union misses the provisional 200 MB package target. No
  quantization was run because the R2 quality gate has not passed.
- No model, dataset, checkpoint, or run was modified; no training, GPU, or
  sealed-audit access occurred in this reconciliation.

## Next engineering decision

Keep the model-capacity envelope below 200M by preferring compact task readouts
for a sensor route; do not add a large sensor encoder without evidence that the
native temporal route fails on a newly admitted, independent task. The next
quality milestone is still a rights-cleared, grouped, both-input-required
multimodal task with controls, followed by a unified runtime and package audit.
