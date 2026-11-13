# Temporal discovery pilot — actual execution and visible-cue scope

## Result and interpretation

The fixed one-scene training regression pilot completed32/32 uniform source times,
all32,768 raw model proposals and1,070 NMS survivors. No label chose a frame,
prompt, threshold or candidate; all output banks were saved before a separate
posthoc scorer opened the training ZIP. Inference exit0, frozen state/RNG/source
checks passed. The full prequantization<=200M Omni objective remains unmet.

The diagnostic threshold is bbox IoU>=0.5, not pixel IoU, recognition Accuracy,
correct-answer counts or a deployable target selector. Sparse boxes were linearly
interpolated only within source ranges; unavailable boxes were not true negatives.

|Posthoc source role|All time/ID cases|Box available|Box unavailable|Dynamic raw opportunity|Dynamic NMS opportunity|Frame0 static NMS opportunity|
|---|---:|---:|---:|---:|---:|---:|
|group|96|96|0|81/96|80/96|27/96|
|hidden|32|19|13|1/19|1/19|1/19|

Game-object groups are the exact grounded IDs[1,2,3]; hidden source ID7. Group
boxes are exact annotations at12/96 cases; the hidden box is exact at1/19. Nearest
annotation distance is at most13frames, not proof of interpolation correctness.
All128 source/time rows remain, including13 missing hidden-box times. Source
masked bracket flags are saved without filtering or equating them with visibility.

### Observed cue versus missing memory/semantics

The dynamic hidden opportunity occurs at234; the static opportunity occurs at395.
Both aggregate counts are1/19, but they are **different observations**. Raw frame
review shows a small blue item exposed at234 when a cup is raised; it cannot be
directly confirmed in the reviewed395 image with closed cups. Posthoc maximum-IoU
NMS proposal592 at234 isolates pixels in that small-item region (bboxIoU.6868).
That proposal was chosen by GT geometry for inspection only, not inference.

This establishes an available later visible cue in this known scene, **not** an
input-only way to choose the intended item or maintain its covering-object relation.
The first annotation at224 must not be interpreted as object birth or first
visibility. The item may already be present but hidden. Likewise,1/19 does not
establish a detector/model capacity ceiling or18 visible detection failures.
Static bbox coincidence at395 is not recognition of that item in frame0.

Do not discard the temporal cue or promote per-frame masks as a Decision solution.
Next bind all model-produced per-time candidate queries to their source times and
measure memory/association across occlusion, with source roles used only for
supervision/reporting. Preserve variable groups, input-only initialization and
uncertainty; do not seed only proposal592 from its GT match. The game's semantic
begin/end anchors and actor-relative left/depth relations remain unverified for
deployment. This pilot does not justify another fit or budget extension on the
failed99-scene readout. Scene generalization and genuine joint evidence remain open.

## Cache admission and cost

The installed reset_tracking_data API clears object/prompt/output/history state
while retaining one source frame's vision features. A new session is made at each
source time. CPU contract tests verify same-frame retention and cross-frame
isolation. A separate GPU smoke proved raw masks/scores for128 proposals are
bitwise equal to fresh sessions, and all1024 cached frame0 packed masks/scores/
boxes/points/NMS indices match the original pinned archive. Encoder hooks count
one image encoding for16 cached chunks, versus one per fresh chunk.

This removes15 redundant image-encoder calls per frame, but **does not demonstrate
end-to-end speedup**. Smoke fresh timing covers only128 proposals, while its cached
window covers1024, with different warmup/cache contexts; no speed ratio is valid.
Primary total541.532991s for32 times (~16.923s/time) remains
too expensive for a claimed low-latency product. It includes decoding, preprocessing,
hashing and save work. No comparison against a fresh-session full32 run was made.

|Run stage|Body seconds|Peak Torch allocation bytes|Peak process working set bytes|
|---|---:|---:|---:|
|Independent cache smoke|24.893610|1090208768|1686380544|
|Fixed32 inference|541.532991|682148352|1614536704|
|Fresh last-time model reload|20.777543|682148352|1605390336|
|Posthoc CPU score|13.070476|not used|not measured|
|Independent CPU audit/raw review|5.695721|not used|not measured|

Primary synchronized cached forward windows sum520.172815s; body excludes
initial Python/library imports and is not isolated GPU busy time or Decision
latency. Small CPU tests/preparation partly overlapped inference. Peak allocation
excludes driver/other-process memory.6GiB Torch/8GiB disk/120s smoke and1200s main
admission gates passed; no external watchdog is claimed.

Only the existing frozen SAM2.1 Tiny was loaded:38,962,498 registered parameters,
including unused memory components; extra learned parameters0, optimizer updates0.
This is a component count, not a full Omni inventory. The existing checkpoint is
155,908,064bytes. Main proposal banks total10,709,457bytes; raw masks/media/weights
remain outside Git. RTX3080Laptop16GB, i7-12700H, existing Torch2.6.0+cu124,
Transformers5.6.2, FP32/SDPA, TF32off, seed17/CPU4; no install/upgrade/download,
cloud charge, quantization, Recovery, main merge or product promotion.

## Verification and preserved corrections

- Complete32 proposal archives passed hash/key/shape/dtype/finite/box/NMS checks
  before annotations. The scorer independently regenerated all boxes and NMS.
- Independent audit checked160 saved arrays, all32 raw source RGB hashes,
  all128 posthoc cases/denominators, raw128 cache/fresh outputs and first/last
  full proposal arrays. Fresh last557 model reload matched all5 arrays by bytes;
  full32 fresh GPU replay was not run or claimed.
- Model state before/after `658e3aaff3d08b6cc8337a1f125e5ee55451ed3bfec38f9def601a14bb6912d8`; all parameters frozen with
  gradientsNone; CPU/CUDA RNG and checkpoint/processor/code/media hashes unchanged.
- CPU session tests initially used the library's string dtype default, which is
  rejected by Tensor.to; explicit torch.float32 matches the existing processor path.
  No library patch. Missing pure APIs supplied TDD REDs before implementation.
- Before inference, review caught unpinned legacy-reference and missing video
  processor source guards; both fixed. Before scoring, review caught annotation
  access before complete-bank validation; added6 regression cases and fixed it.
  No result-based setting/tolerance/source change or retried seed occurred.
- Ruff passes;1108CPU tests pass/1skip15.81s. All manifests, four catalogs and policy
  audit pass. Publication-head CI is verified after push.

## Commands and artifact provenance

Inference source `ee91735382105b11669bd49fac1b4dd3c2c5112b`; scorer source `a3e38e58e737bfd50701666aea76b7bace9ab449`.
Primary started `2026-10-14T03:51:37.912428+00:00`, ended `2026-10-14T04:00:39.445910+00:00`. All processes exited0.
Protocol and inference code were committed before smoke/main execution.

```powershell
python -m scripts.probe_dynamic_discovery --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/dynamic-discovery-cache-smoke-01
python -m scripts.probe_dynamic_discovery --output C:/CodexArtifacts/pretrained-reuse-r2/dynamic-discovery-fixed32-01 --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/dynamic-discovery-cache-smoke-01
python -m scripts.probe_dynamic_discovery --output C:/CodexArtifacts/pretrained-reuse-r2/dynamic-discovery-last-reload-01 --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/dynamic-discovery-cache-smoke-01 --replay-last C:/CodexArtifacts/pretrained-reuse-r2/dynamic-discovery-fixed32-01
python -m scripts.score_dynamic_discovery --run C:/CodexArtifacts/pretrained-reuse-r2/dynamic-discovery-fixed32-01 --output C:/CodexArtifacts/pretrained-reuse-r2/dynamic-discovery-posthoc-01.json
python C:/CodexArtifacts/pretrained-reuse-r2/audit-dynamic-discovery-01.py
```

Use new output paths; existing outputs are refused. Local independent raw-review
PNGs are posthoc diagnosis only, never prompts/initialization. Full source/license
admissions remain pinned in the earlier SAM2/Perception Test reports. Known train
scene selection and observed failure do not establish blindness or representativeness.
No Accuracy/NLL/Brier/ECE for supplied-option Decisions was measured. The original
full four-modality95–98% goal and <=200M resident constraint remain unchanged.

- Main config SHA `303f0582ae0c7d42c181e9070d66f72e73ac4da62701e837f9299d1d451d1c7c`.
- Main result SHA `191796c3e85ffea628c55d45f15cd1a09ecdba546551476e57dd55ce09a7ae0e`.
- Smoke result SHA `bc05e706e702013b36a123166077e092442966f2e29ab01d785792bc55c83684`.
- Last reload SHA `34a40ab1c0dece860e699375908b6f5d0683d7e98994a8f77111f2b5837efa7c`.
- Posthoc score SHA `de2c495522f978edcefeb39a6c81084ef32d8add90223a53b882fd2bccab57cc`.
- Independent audit SHA `097f08b712267839922236e381e701acd8fdd3806f980b84a616474f5266df5a`; script SHA `8bec9154f580a8a321a9814551a0532e35612f0232402be1b4262854389678ba`.
- Artifact root `C:/CodexArtifacts/pretrained-reuse-r2/`, prefixes
  `dynamic-discovery-*`. All older artifacts/configs/checkpoints remain protected.
