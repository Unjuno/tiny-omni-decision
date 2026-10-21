# TAPIR checkpoint admission — 2026-10-13

Motivation: pooled video decisions overfit, and direct frozen-token nearest-neighbor correspondence failed the two fixed train diagnostics. A released correspondence model is a candidate source of transferable motion evidence. It does not replace semantic understanding or answer selection, and requires an initial query point.

Official source pinned at `google-deepmind/tapnet@730cda1c730877cfedbe01bf87fb1cadb78a565d`; owner Hub revision `google/tapnet@5d3fb48e76c5422841e38501514121e251beabb7`. The pinned official README explicitly licenses released checkpoints linked from it under Apache-2.0, separately from dataset materials. [Source and license](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/README.md), [Hub checkpoint](https://huggingface.co/google/tapnet/blob/5d3fb48e76c5422841e38501514121e251beabb7/tapir_checkpoint_panning.pt).

Downloaded `tapir_checkpoint_panning.pt`:124,348,474 bytes; SHA-256 `628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa`, matching immutable Hub LFS metadata. `torch.load(weights_only=True,map_location='cpu')` succeeded:188 state tensors,31,072,327 FP32 tensor elements. This is state-dict inspection, not yet a loaded-model parameter count or runtime pass. Full Omni resident accounting remains required before adoption.

Official PyTorch files were fetched at the pinned source revision and inspected. Dependencies `tree` and `einshape` are absent from the normal environment; no environment update was made. Model construction, strict load, forward, memory/latency and task quality remain UNMEASURED. No training, inference, validation or sealed audit was performed in this admission step.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01/`: source/Hub metadata, official README, four PyTorch source files, checkpoint and inspection JSON. Next safe step is isolated dependency resolution and strict model load, then the same predeclared train-only correspondence points if runtime is viable. No tracker-derived score should be reported as general Decision accuracy; oracle query initialization and any eventual automatic point selection must be evaluated separately.

## Executed CPU audit and fixed train diagnostic

Isolated `deps/` now contains einshape1.0 and dm-tree0.1.9 (`pip --target --no-deps`); normal environment unchanged. Default current constructor failed strict load because it includes extra convolutions and a wider mixer than this legacy checkpoint. Explicit `TAPIR(pyramid_level=0, extra_convs=False)` matches all keys strictly. This constructor choice follows exact checkpoint shapes; do not claim the current BootsTAPIR demo default matches these weights.

Loaded parameters:31,072,327. Two synthetic frames/one query produced finite tracks, occlusion and expected-distance tensors in0.222264 seconds, CPU four threads, FP32. This is plumbing evidence only.

Same fixed train plans as previous two-token diagnostics, initial annotated center as oracle, eight sparse frame IDs, RGB resize256, input[-1,1], query[t,y,x], outputs[x,y]. No training or validation. No confidence-based filtering of scored frames. Mean center errors compare seven subsequent annotated centers; a physical point need not coincide with a changing bounding-box center.

| Video | TAPIR error | Static exact-center error | Predicted point in box | Forward seconds |
|---|---:|---:|---:|---:|
| video_10103 | 0.028853 | 0.036967 | 5/7 | 0.723376 |
| video_10186 | 0.155358 | 0.179315 | 3/7 | 0.512228 |

Both errors improve over static on these two examples, but moving-video tracking remains weak. This is not general Decision quality or population evidence. Eight sparsely sampled frames create large motion gaps; dense temporal input is a distinct unresolved factor. Oracle initial points, offline bidirectional access and no automatic object selection prevent direct product adoption.

Artifacts: `tapir-audit-01/runtime01.py`, `runtime-synthetic.json`, `track01.py`; real-video results at `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-tapir-tracks-01/result.json`. Failed default-load command is recorded above; no strictness bypass or dropped checkpoint keys. Original artifacts preserved.
