# VideoSAUR reconstructed runtime admission — 2026-10-13

Status: **PASS_RECONSTRUCTED_RUNTIME_NOT_ORIGINAL_TRAINING_PROVENANCE**.
This completes a real checkpoint/runtime check, not a Decision quality gate.

## Exact load

Used source7a9f85d9388fdeb44eb729ce7ada31569a2e77a2 and the fixed checkpoint/hash
in [the checkpoint audit](R1_VIDEOSAUR_CHECKPOINT_AUDIT_20261013.md).
Only the checkpoint-required slot width was changed from64 to128 relative to the
published config. Seven slots, four attention heads, two grouping iterations
(three initially), predictor/decoder depths and other non-shape settings come
from the published config/source. Exact original training settings remain UNKNOWN.

Built the upstream initializer, FrameEncoder/TimmExtractor, SlotAttention,
Transformer predictor, temporal wrappers and SlotMixerDecoder. No network weights
were omitted: all256 state tensors loaded with strict=True, zero missing and
unexpected keys. The backbone is instantiated pretrained=False then populated
from this checkpoint; no second unpinned backbone download occurs. No strict=False.
Actual resident parameters93,530,201, buffers0, trainable0. Includes reconstruction
and temporal-target renderer, which remain resident even if a future decision
path does not need them. This is a visual component, not complete Omni accounting.

Used standalone nn.Module wiring of upstream inference components. Three utility
functions were AST-extracted unchanged to avoid Lightning/training imports; all
loaded source hashes are recorded. Training losses/loggers were not instantiated.
This is not proof that the complete historical training environment works.

Dependencies timm0.9.7 and einops0.6.1 installed with --no-deps into isolated
`C:/CodexArtifacts/pretrained-reuse-r1/videosaur-runtime-deps-01/`.
Global working environment was unchanged: Python3.11.9,Torch2.6.0+cu124,
Torchvision0.21.0+cu124,CPU four threads; RTX3080Laptop16GB for GPU smoke.

## Real-video inference and reload

Train clip video_10103, frames0/1, original media SHA verified. Direct tensor
bilinear resize518x518 with antialias=True, ImageNet mean/std. This retains the
full frame but distorts aspect ratio; it is explicit runtime preprocessing,
**not asserted equal to the original dataset's crop/resize behavior**.
No annotation, question answer, validation or sealed audit loaded.

| Measurement | CPU FP32 | GPU FP32 |
| --- | ---: | ---: |
| First/repeated/reloaded forward seconds |1.5995 /1.6373 /1.5955|.3214 /.1597 /.1178|
| Same-device same-seed replay |exact|exact|
| Strict save/reload output |exact|exact|
| Frozen state unchanged |yes|yes|
| All outputs finite |yes|yes|

Output slots[1,2,7,128], decoder masks[1,2,7,1369]. Seed17.
Forward timings exclude loading/decode/preprocessing; GPU timing synchronizes.
Three calls are a smoke, not a throughput/latency distribution benchmark.
GPU max allocated1,140,736,512bytes, max reserved1,350,565,888bytes during the
whole harness, **including two resident model instances for reload validation**.
Not a measured production single-model peak. GPU was0MiB/0% before smoke with
only Codex listed; no other process stopped or modified. No cloud cost.

## Device randomness check

Same numeric seed does not imply identical CPU/CUDA random initialization.
Naively comparing independently initialized CPU/GPU outputs produced max slot
and mask differences2.137268 and.999526. Do not call this numerical parity failure
without controlling initial slots. With the **same CPU-generated initial slots**
passed to both devices, max differences fell to5.1189e-5 (slots) and3.6895e-5
(masks), both finite. This verifies close numerical agreement for this input,
not exact cross-device reproducibility or general quality. Later experiments
must freeze device and initialization policy; do not select favorable seeds.

## Preserved artifacts

Root `C:/CodexArtifacts/pretrained-reuse-r1/`:
- `videosaur-runtime-01/`: CPU result, state, features.
- `videosaur-runtime-gpu-01/`: GPU result, state, features.
- `videosaur-device-parity-01/`: shared-initial-state result and paired features.
- `videosaur-source-audit-01/`: runtime_audit.py, runtime_audit_gpu.py,
  device_parity.py. All outputs use exclusive-create directories.

Harness hashes:
- CPU60df674c3a733580952042bdcfecac9ebc9dc252d979fb5a73816a18bc6ac6d4
- GPU4ba4c877375133305852fcba2f83cee7710ca2824f66006eb1aebddf13108fab
- parity24bfb76d6db142951d4237c0999f46f4b16e4a7a9401add98224a5f0e4a92276

Each harness ran as `python <absolute script path>` from the project worktree at
f06f296. Results include full imported-source hashes and package versions.
CPU saved state374,228,482bytes/SHA8512541e916372104de58f12934557b4ca552ad3d630ca2d6ba9f743d1ee0823.
GPU serialization differs in device metadata; exact tensor/reload checks passed.
Large weights/features stay outside Git and legacy artifacts remain unchanged.

## Next gate

Runtime is feasible locally. Next perform the same fixed eight-train-video
oracle mask diagnostic with an explicit518x518 input policy and seeded
initialization; no full validation fit yet. Compare engineering candidates with
all preprocessing/model differences disclosed, not as a one-factor pretraining
causal experiment. Checkpoint rights remain REVIEW; no product adoption,
compression or R2 quality claim is justified yet.

