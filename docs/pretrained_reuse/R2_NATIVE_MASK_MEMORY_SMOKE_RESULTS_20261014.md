# Native video mask memory: executed smoke, scale not admitted

## Execution and scope

The full <=200M prequantization native text/image/audio/video Omni Decision goal
with calibrated supplied-option JSON, joint/OOD and95–98% quality remains unmet.
This is a frozen representation feasibility experiment, not a product candidate.

Source/protocol commit `8d29426`; corrected execution commit
`7e0cf77` (full SHA in [machine report](R2_NATIVE_MASK_MEMORY_SMOKE_RESULTS_20261014.json)).
[Predeclared protocol](R2_NATIVE_MASK_MEMORY_PROTOCOL_20261014.md).
Use existing released SAM2.1tiny38,962,498parameters, no training, GT seeds or
semantic prompt. One already-observed TRAIN clip10103, first2 original automatic
NMS masks and all32 original observation times. Eight videos/all696NMS survivors
were planned; they were **not inferred** in this experiment.

Commands executed in `C:/CodexWorktrees/pretrained-reuse-omni`:

```powershell
python -m scripts.probe_native_mask_memory --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-smoke-01
python -m scripts.probe_native_mask_memory --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-smoke-02
python -m scripts.probe_native_mask_memory --smoke-from C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-smoke-02 --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-scale-admission-01
python -m scripts.audit_native_mask_memory_smoke --smoke C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-smoke-02 --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-independent-smoke-audit-01.json
```

Smoke01 stopped before propagation because its reader compared native-resolution
RGB with the inherited256-square resized RGB hash. Fixed reader reproduces the
original verification view while returning raw-resolution RGB to SAM2. Regression
test RED→GREEN; hash gate unchanged; failure/config preserved. No model/data/time
or precision adjustment. Read-only execution review cleared inherited settings,
full replay cohort admission and final resource checks before execution.

## Measurements

| Variant | Vision calls | First seed(s) | Second seed(s) | Saved native object-frames |
|---|---:|---:|---:|---:|
| Actual32 RGB/cache1 |64|8.8560|8.3668|64|
| Actual32 RGB/cache32 |32|8.4984|5.6170|64|
| Frame0 repeated32/cache32 |32|8.4524|5.6576|64|

All actual/reference versus actual/cache32 arrays have identical shapes, dtypes
and bytes, including raw FP32 mask/presence outputs and packed masks/bboxes. The
two NPZ files also have the same SHA256. Cached repeat is a changing-visual-input
control, not a surgical memory-weight ablation. Empty predictions are retained.
Independent NumPy audit checks every192 saved raw frame product, packed mask,
bbox, empty flag, score and streamed raw SHA; plus seed/frame identities, exact
cache parity, source/config hashes and runtime projection. It does not establish
model correctness or independently recreate source annotations.

Smoke02 body55.4603s; summed propagation bodies45.4482s (not pure GPU active time).
RTX3080Laptop16GiB; peak allocated3,980,698,624bytes (3.71GiB), peak process working
set3,312,787,456bytes (3.09GiB). Three NPZ files43,148,996bytes. No cloud cost.
All parameters frozen, no gradients, CPU/CUDA/NumPy RNG unchanged; model state
before/after SHA `4d2293307db6c980b45d44a0f38ebde77c1d29e7928d3272be0db54d7b83e025`.
FP32 weights/outputs, native internal memory BF16; no library patch or batched-head
replacement. Full model/processor/library/protocol/source pins in machine report.

The full two-arm696seed projection uses each cache32 arm's second-seed time×696
×1.25 plus first-seed time×8: **9,944.5008s (2h45m44.5s)**. It exceeds the fixed
7,200s admission budget. Actual main CLI rejects it before model creation or any
propagation, preserving `FAIL_INCOMPLETE` with the admission error. A smoke PASS
does not imply a scale PASS. No seed/frame reduction or tolerance change.

## Conclusions and unfinished work

Native public cache reuse is numerically admitted **only on this fixed2seed prefix**.
It halves encoder calls here, but repeated per-object memory/decoder work still
dominates scale cost. This is not a product latency or quality improvement.

Full8video/696seed execution, fresh separate-model replay, posthoc group-identity
comparison and semantic Decision Accuracy/NLL/Brier/ECE are **unperformed**. No
layer/memory capability ceiling, identity gain or blind evaluation claim follows.
The original scale gate remains failed; increasing a wall-time budget would need
an explicit new recorded resource protocol while preserving numerical/data/cohort
conditions. An engineering alternative must independently prove native parity;
do not reuse the previously rejected batched decoder or shrink the cohort.

All legacy Teacher/student/config/corpus/artifacts retained; official evaluation
and sealed data unopened by the inference/auditor processes. CPU-only publication
adds small code/tests/metadata; model weights/media/NPZ are not in Git. Larger
multimodal task quality, useful synchronized public-data fusion and deployment
still require separate evidence. No new head, learning, quantization or promotion.

## Publication checks

Ruff passes. Full CPU suite: **1,414 passed /1skipped** in22.64s. Forced no-Torch
suite: **1,007 passed /95skipped** in7.02s. Model/dataset manifests, all four
catalogs and ALLOW policy validation pass. Read-only code review finds no remaining
Critical/Important issue. Publication-head GitHub Actions are checked after push;
the exact-head receipt is kept separately to avoid changing the evaluated commit.
