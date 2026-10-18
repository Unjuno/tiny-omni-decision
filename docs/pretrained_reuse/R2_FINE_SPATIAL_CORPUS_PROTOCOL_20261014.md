# Actual32×32 spatial retention — fixed before execution

## Bounded purpose

The all99-train16×16 grid supervision check has only19 same-hidden pairs/5scenes.
Retain actual finer frozen feature grids before changing labels or fitting again.
This extends the existing extraction path in a separate script/output, authorized
by the user's autonomous strategy. Full native text/image/audio/video, conditional
reuse, useful joint/OOD/calibration95–98% and <=200M product gates remain unchanged.

No training, GT crops/runtime seeds, annotations, validation arrays/annotations,
sealed audit, precision/encoder changes, additional model, cloud or product
promotion. All99 existing known train clips remain with identical media/cuts and
32 original sampled times. Existing coarse bank/config and Teacher artifacts are
read-only. This is input retention, not a new quality benchmark.

## Input and implementation contract

Pinned complete coarse resultSHA
`a306930eb673a00c246b7b2e9caad3044315cb9f19b0f5ee2c43987d2f04f0da`,
configSHA`d17c1f1f9e55f17d58983603c1c8bf27ac171bed9d2ee04108535c84b15e35ff`,
train media manifestSHA`469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3`.
Select all99 train IDs before array reads; check cross-split identities from the
already-observed metadata without opening development arrays. Bind every selected
media and old appearance/position/frame file and verify hashes before/after run.
Exact original-frame and row-major coarse positions, media cuts and preparedRGB
hash must match the existing corpus. No source-role metadata or label selection.

Released31,072,327-parameter frozen TAPIR,
`google-deepmind/tapnet@730cda1c730877cfedbe01bf87fb1cadb78a565d` and owner Hub
`google/tapnet@5d3fb48e76c5422841e38501514121e251beabb7`, recorded inR1. Full
implementation pins are checked from the prior completed extractor report;
weightSHA`628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa`.
Existing local environment; pyramid_level0,extra_convsFalse, strict state load.
FP32/raster256, TF32off/highest/deterministic, seed17, fixed32-frame batch.
Decode/native resize256×256 exactly as the previous extraction. No upgrade or
new weights. StateSHA`3c92ad70854f549bc46fde282f4248e949d733fadffcfa2b060afada2ea6dd4a`
before/after, requires_gradFalse and no gradients, TorchCPU/CUDA RNG unchanged
across forwards. Historical coarse runner/pooling hashes are required, not replaced
by the current source bytes; smoke/fresh bind the contract inside the actual config.

Actual lowres[32,32,32,256] and hires[32,64,64,128] from raw frames produce
32×1,024×384 tokens: each native low cell is retained, each aligned2×2 high block
is mean-pooled withFP64 accumulation then storedFP32. Preserve every cell/time;
add normalized row-major x/y/original-time and original frame IDs. There is no
upsampling of coarse features. Stored dimensions are4× the coarse cell count,
not4× independent videos or4× source data diversity.

Every new forward must also produce direct16×16 pooling **bitwise equal** to
the old saved appearance. Do not instead average rounded32×32 tokens and claim
bitwise equivalence to the original single-stage pooling. For the eight existing
raw-grid reference clips, full low/high/time arrays must match archived reference
bytes. First selected clip repeat-forward exact. Actual saved fine arrays reload
with identical shape/dtype/bytes. Hash config/code/protocol/weights/media at exit.

## Execution, smoke and admission

8GiB disk reserve plus64MiB safety; exact .npy estimate50,725,504bytes/clip,
99clips5,021,824,896bytes. Require12GiB free GPU at preflight, peak allocation<=4GiB,
process peak<=4GiB, totalbody<=1,800s checked between clips and at exit (not watchdog).
Existing GPU idle confirmed before starting. Only one GPU process from this task.
Failures retain partial outputs/logs; never delete old data to make room or silently
relax a numerical/resource guard.

Separate-process1-clip smoke uses firstfixed train video_10103/full32times and the
same hash-bound all99-plan/runtime contract. Frozen31.07M/actual coarse and raw-grid
parity/repeat/save-reload/config/finite/resource gates must pass before main.
Measure clip time/VRAM/storage and projected full cost; do not launch main if those
measurements make the fixed limits impractical. Main requires smoke's actual bound
config/result/three array hashes and must reproduce that smoke clip exactly. Main
retains all99 scenes; no result-based subset or automatic retry/resume.

After terminal complete extraction: fresh separate-process smoke of the firstclip
compares all three arrays to main. No full5GB second copy is required under current
disk limit. Independent CPU audit checks every99 file/schema/native geometry/time/
feature finite/hash/source binding; derives all fine cells for the eight preexisting
raw-grid reference clips using independent explicit four-cell arithmetic and native
low channels. First-clip fresh arrays exact. This numerical reference scope is8/99,
not every99 video reconstructed independently; report the distinction.

Tests RED→GREEN for native fine shapes/arithmetic/position alignment, exact storage
reserve and actual smoke artifact/config binding. Read-only review before execution,
Ruff/full CPU/manifests and publication-head CI. No quality score or new fit in this
unit; only after complete admission define a separate train-only supervision check
on actual fine cells, with its own unchanged old-data controls and scope.
