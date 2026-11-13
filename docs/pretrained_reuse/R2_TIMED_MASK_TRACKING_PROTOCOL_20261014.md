# R2 timed model-mask tracking primitive — fixed 2026-10-14

## Question and scope

Can every model-produced proposal at every retained time become a time-specific
TAPIR query, without selecting the known small-item mask using annotations?
This extends the existing mask-query path, not the failed readout fit. It does
not measure QA, semantic target selection, cover association or Omni quality.

Use the already admitted public Perception Test train regression scene
`video_10903`, original official558-frame cut and unchanged32 uniformly retained
times. Prior dynamic discovery is pinned to result SHA-256
`191796c3e85ffea628c55d45f15cd1a09ecdba546551476e57dd55ce09a7ae0e`.
Its all32 banks contain32,768 raw proposals and1,070 unfiltered NMS survivors.
Keep every survivor, its original proposal ID, original frame and duplicate
coordinates. No annotations, option answers, source role IDs, TopK or score gate
enter query construction. The previously inspected proposal592 is not singled out.

## Input contract and tests

For each mask use the existing exact COM-nearest foreground pixel-center seed
(row-major tie), converted to float32 `[original_frame,y,x]` in256×256 raster.
Do not substitute retained-time ordinal or frame0. Empty masks retain invalid
flags; the fixed GPU experiment fails admission if any seed is invalid rather
than generating substitute queries or dropping candidates. Pair identity is
`(original_frame, proposal_id)`, not proposal_id alone. Candidate time is the time
of its observation, not proof of physical birth/first visibility.

CPU tests cover xy/order/time conversion, last frame557, retained duplicates,
empty flags, proposal permutations/no mutation, invalid identities/times and
on-or-after-observation flags. Run the full CPU suite and Ruff before execution.
An independent code reviewer checks the bounded change before GPU admission.

## Fixed actual execution

Frozen pinned original TAPIR31,072,327 parameters; checkpoint SHA-256
`628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa`.
Strict `pyramid_level=0, extra_convs=False`, seed17, FP32, TF32off, deterministic,
CPU4, original chunk16. Native RGB256 conversion and scaling `/255*2-1` match
the admitted tracker. Full558 decoded frames, every1,070 query; all three raw
predictions across all558 frames retained. Query appearance is sampled from its
own original time in the full video, not `video[:,:1]`. The512-column token keeps
the existing128 motion/uncertainty columns plus384 query-time appearance.

Store the unchanged32 original sample IDs and an explicit query×sample
on-or-after-observation flag. Backwards predictions remain in raw arrays; this
flag is neither visibility nor confidence. TAPIR sees the complete video, so
even flagged future samples do not establish causal/streaming inference.

Use fresh independent output directories. Save source/config hashes, revision,
model-state hashes, frozen flags/None gradients, CPU/CUDA RNG before/after
tracking, every array with shape/dtype/byte readback, wall/forward time and peak
Torch/reserved VRAM. A fresh process/model replays the entire same query inventory
and must match every saved array including RNG states byte-for-byte. No source
or scientific tolerance changes after seeing outputs.

Resource admission: GPU free≥14GiB, disk free≥8GiB, peak Torch allocation≤12GiB,
body≤600s. The last two are checked after forward, not watchdogs. Preserve any
failed attempt and report incomplete, never silently lower queries/frames or
raise gates. No additional downloads, training, cloud compute, dev/sealed access,
environment upgrades, quantization, product promotion or main merge.

## Next decision

Passing this primitive permits separate posthoc analysis of visibility-aware
target memory and covering-object relation; it does not qualify adoption. All
candidate outputs must exist before such labels are opened. Do not infer a
detector ceiling from interpolated hidden boxes/occluded times, and do not infer
95–98% general Decision quality from this known one-scene numerical experiment.
