# Track-aligned temporal appearance: executed input primitive

The full Omni goal remains incomplete. This admits a numerical feature-routing
primitive, not a Decision model or an improvement in Accuracy. No fitting,
checkpoint selection, annotations, development feature arrays or sealed/audit
data were used. All eight clips are previously used training references.

## Fixed operation and result

All256 original model-produced tracks and the same32 source frame IDs were
retained on each clip: 65,536 point-times, output256×32×384 FP32 per arm.
Fine low32×32×256 and high64×64×128 maps were sampled at estimated x/y using the
official half-cell/border convention, FP64 accumulation, then FP32. No extra
descriptor normalization. The comparison samples each point's first estimated
position across the same time-varying maps; it is not a first-frame feature repeat.

All point/permutation/repeat/readback gates passed. Official pinned CPU interpolation
maximum absolute error 6.85453415e-07 is below the predeclared1e-5 gate. A fresh
process recomputed all48 arrays with identical shape, dtype and bytes. A separate
audit checked every output and cached-input hash, exact motion/frame binding, support and
summary statistics, plus384 independently implemented scalar-vector interpolations.
No hash, tolerance or admission gate was weakened.

|Training clip|Inside closed raster|Official maximum absolute error|Mean actual/static cosine|
|---|---:|---:|---:|
|video_10103|8192/8192|5.73694706e-07|0.591056|
|video_10186|8136/8192|5.96046448e-07|0.228585|
|video_10212|8182/8192|6.85453415e-07|0.212864|
|video_10387|8190/8192|6.1839819e-07|0.308858|
|video_1087|8189/8192|6.2584877e-07|0.452441|
|video_10886|8149/8192|6.2584877e-07|0.257785|
|video_10894|8182/8192|5.66244125e-07|0.345042|
|video_10903|8177/8192|6.29108399e-07|0.404183|

139 out-of-raster point-times remain in the output with border features
and explicit false support flags. Support is only a geometric raster check; it
does not mean an object is visible, correctly tracked or semantically recognized.
Actual/static descriptor variation likewise establishes neither object identity
nor useful temporal reasoning. Occlusion/uncertainty and all old point tokens remain
unaltered; no QA example or point was removed.

## Reproduction, costs and artifacts

- Source commit `e41471000a16059a570ef94a4961056d302d08c2`; the protocol was committed before execution.
- CPU only, official sampler4threads, NumPy2.4.6, Torch2.6.0+cu124 on the
  existing Windows/i7-12700H environment. No model was constructed/loaded and no
  encoder forward, optimizer, backward, GPU training, download or paid compute ran.
- Primary body12.813895s; fresh body12.714092s; independent audit
  5.341576s. These include input hashing/I/O; they do not measure original
  TAPIR tracking/encoder extraction or complete raw-video inference latency.
- Windows peak working set 658,939,904 bytes primary,
  659,427,328 replay. The300s admission gate checks
  between clips and at completion; it is not an external watchdog. Both passed.
- New arrays 201,990,144bytes in48files; replay references these arrays without
  duplicating them.8GiB disk reserve passed. Added parameters0; local/cloud fees0.
- Effective config SHA `fb787e288bee0a74f7aa8190495bea8ccc10c07e3ceef4d3e08c5955acf8992c`.
- Primary result SHA `b395e18d6b57262197c8ac04133a0a10b12dd94698f0b1680bf1ef892088e7fb`.
- Fresh result SHA `7ba8be0837db28fb3eb5589c42bc18771f51194fe58d6813395170b0d00c6c8f`; its separate config binds the
  primary result hash and the same pinned source files.
- Independent audit SHA `0043388b2dab6b11fd989364b9cbff64081338e6e8beb52ae562e46e8193e5ea`; script SHA `43d06bae3d89c29bada1bfcebb7d1ba9c32d8edb1239fa4e1992bf6e4d15565d`.
- Artifacts `C:\CodexArtifacts\pretrained-reuse-r2\track-appearance-primitive-01/`, `C:\CodexArtifacts\pretrained-reuse-r2\track-appearance-fresh-replay-01/`,
  `C:\CodexArtifacts\pretrained-reuse-r2\track-appearance-independent-audit-01.json` and `C:\CodexArtifacts\pretrained-reuse-r2\audit-track-appearance-01.py`.
- Original fine grids/point arrays/media and pinned official source hashes were
  checked before/after. No legacy worktree/config/checkpoint/weight was overwritten.

```powershell
python -m scripts.probe_track_appearance --output C:/CodexArtifacts/pretrained-reuse-r2/track-appearance-primitive-01
python -m scripts.probe_track_appearance --output C:/CodexArtifacts/pretrained-reuse-r2/track-appearance-fresh-replay-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/track-appearance-primitive-01
python C:/CodexArtifacts/pretrained-reuse-r2/audit-track-appearance-01.py
```

Use new output directories for a rerun; the runner refuses an existing output.
Only code, tests, protocol and small result metadata are versioned; arrays and
media stay outside Git. CPU tests1074passed/1skipped16.64s; Ruff, all manifests,
four dataset catalogs and policy audit pass. Read-only code review found no
Critical/Important issue; resource-limit wording was clarified before execution.
Exact publication-head CI is checked after push.

## Next evidence needed

Close numerical extraction/routing admission for these eight train clips. Do not
adopt a model or launch another fit solely because descriptors change. The next
controlled test must establish label-free target/cover association and a mapping
from relationships to supplied conditions, followed by scene-generalization and
genuine complementary multimodal benefit. A corpus-wide cache/readout is not yet
run for this primitive. The earlier readout overfit remains a failure to generalize;
this feature check does not reverse it. The<=200M pre-quantization/full-resident
constraint and four-modality95–98% target remain unchanged and unmet. No compression,
Recovery, main merge or product promotion follows this input-only result.
