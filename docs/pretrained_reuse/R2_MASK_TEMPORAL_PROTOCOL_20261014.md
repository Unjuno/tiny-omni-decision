# Fixed train-only model-mask temporal initialization protocol

## Question and inherited evidence

Can all existing nonempty SAM2 NMS masks initialize their own frozen TAPIR
trajectory, avoiding unsupported fixed-grid pooling? This is a component
feasibility experiment, not a Decision/Omni quality run. The prior CPU pooling
smoke retained696 candidates but209 lacked initial grid support; separate source
scoring found23/24 initial similar-object opportunities with joint point support.
The unchanged original masks offer24/24 bbox opportunity. Masks and prior grid
results are preserved; no GT index may choose or initialize a track.

## Frozen inputs and execution

- Same eight known train IDs:10103,10186,10212,10387,1087,10886,10894,10903.
  This previously selected/inspected subset is not blind or representative.
- Preserve all696 nonempty saved NMS masks and their original proposal IDs;
  no TopK, prompt, confidence filter, seed search or mask/NMS change.
- One initial point per mask: foreground pixel nearest binary-mask COM, exact
  integer-distance comparison and row-major ties. Use pixel centers mapped to
  the256 tracker raster, t=0. Verify FP32 query remains inside its own mask.
- Corrected plan `C:/CodexArtifacts/pretrained-reuse-r2/perception-mask-query-plan-02/`;
  plan SHA `d5de3ccd938b4a6354719a5ffd504c38413015a66dd4e55fa2ed99e5114545b3`.
  Plan01 remains preserved and will not be used: review found a floating-point
  COM tie issue before any GPU launch. Snapshot/module hashes distinguish them.
- Existing pinned TAPIR constructor `pyramid_level=0,extra_convs=False`, frozen
 31,072,327 parameters; checkpoint SHA
  `628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa`.
  Existing source-file hashes must match the old extraction ledger. No upgrade.
- Process-local seed17, FP32, TF32 disabled, deterministic,4 CPU threads,
  query chunk16. Same RGB256 reformat and [-1,1] scaling as the original tracker.
- Decode the full official input, excluding every frame at/after a nonnegative
  reveal cut; cut=-1 retains full source. Maximum1051 frames. No temporal thinning.
- Local GPU only; preflight >=14GiB free GPU and >=10GiB free disk. Hard600s
  body cap and12GiB peak Torch allocation cap; preserve failures without replacing
  clips. No old process interruption, model training or paid compute.
- Fresh output `perception-mask-query-trajectories-01/` under the external R2
  artifact root. Do not overwrite any legacy bank, trajectory or teacher artifact.

## Saved evidence and limits

Retain raw tracks, uncertainty outputs, initial appearance, all query/proposal
IDs,32 ordered temporal indices,512-wide combined tokens, hashes, timings and
memory. Verify finite values, output order/count, input hashes, bitwise first-clip
repeat and save/readback. Check model-state/weight hashes unchanged.

The original `mask_trajectory_features` pooling primitive retains unsupported
zero tokens with explicit false flags. These flags describe availability of
initial support, not track validity; a consumer must fail closed if all are
unavailable. Direct mask queries are a separate initialization condition, with
new inference and outputs rather than invented/reused point trajectories.

Only after outputs are saved may a separate CPU scorer open training annotations
for initial geometry and joint mask/point support. Retain all cases and the0.5
bbox-IoU gate. Do not choose GT-best trajectories. Source annotations are already
known to the investigator; scorer isolation does not establish blindness.
No validation feature arrays, sealed audit, checkpoint selection or QA Accuracy/
NLL/Brier/ECE is part of this protocol. A later quality fit requires an independent
fixed protocol and evidence for target association, temporal retention and actor
coordinates. Initial support alone does not prove any of those or95–98% quality.
