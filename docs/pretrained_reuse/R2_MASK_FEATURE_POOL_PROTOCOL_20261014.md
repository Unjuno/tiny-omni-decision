# R2 masked frozen-feature statistics — fixed 2026-10-14

## Purpose and declared representation

Current point-only appearance/nearest geometry has not established semantic
cue-to-cover binding. Prepare object-region inputs for a later supervised
readout, while native wider extraction continues unchanged. This is **input
representation admission only**, not a new semantic ranking, training, QA or
held-out experiment. No annotations, source roles, dev/sealed data or GPU/model.

Use completed known train10903's **all32 native time banks/all1,070 NMS survivors**
and corresponding cached TAPIR low32×32×256/high64×64×128 FP32 feature maps.
Hash-bind parent results/configs/media/time/proposal archives and dense grids.
No GT regions, target object/score selection, changed NMS/mask threshold or TopK.
Preserve each mask/time/proposal ID; duplicates/empty masks are not removed.
One existing media asset, not new independent data or1,070 physical objects.

Treat each feature cell as piecewise constant over its integer raster support.
For each model mask, its foreground pixel count weights each covered cell.
Compute per-channel weighted mean and two-pass population standard deviation in
FP64; save FP32. This is a deliberately derived area representation, not the
former interpolated point descriptor or a claim about exact pixelwise CNN output.
All grid cells must partition the256×256 mask raster exactly; no interpolation,
unannounced resize/crop, bbox/background pooling, L2 normalization or clipping.
Concat low256 then high128: **384 means and384 standard deviations per mask**.
Empty mask descriptors remain zero with explicit count; a frame with no NMS
survivors keeps empty arrays and its frame offset. Zero registered parameters
added; future readout capacity/quality must be measured separately.

## Correctness and execution gates

New `mask_feature_pool.py`, `probe_mask_feature_pool.py`, CPU tests. TDD covers
hand-derived area weighting/partial cells/population std, exact zero variance
on large constants, input/order/overlap/RNG preservation, empty retention, binary
mask/finite FP32/aligned cell rejection. Existing extractor/config/corpus unchanged.

Before larger inputs, CPU-run this completed clip only. All1,070 descriptors and
32 time offsets must survive/save/reload exactly. Independent numerical reference
for the **first eight input-order frame0 NMS survivors** expands cached cells
onto every256×256 pixel, then directly computes masked-pixel FP64 mean/std.
Fixed absolute tolerance≤1e-5, relative0; all eight required, not chosen from GT.
Reference scope is eight objects only; all1,070 statistics still saved/finite.
Do not change tolerance or prefix after results. No quality/seed selection.

Fresh separate CPU process must reproduce all saved array shapes/dtypes/bytes
under identical source/config hashes. Primary reference config/result/archive
must be hash-bound before compute and after replay. Seed17; NumPy RNG unchanged.
Record both bodies/bytes/config/result/producer hashes/golden errors. Body cap
300s after imports, checked at end (not watchdog). Failed attempts retain config
and `FAIL_INCOMPLETE` traceback; never report an execution fault as quality failure.
No cloud/download/upgrades/weights/training/quantization/main merge/intervention
in the live native extraction. CPU work overlaps that job; no isolated throughput
or product latency claim. No GPU tests or new GPU workload while it is occupied.

```powershell
python -m scripts.probe_mask_feature_pool --output C:/CodexArtifacts/pretrained-reuse-r2/mask-feature-pool-pilot-01
python -m scripts.probe_mask_feature_pool --reference C:/CodexArtifacts/pretrained-reuse-r2/mask-feature-pool-pilot-01 --output C:/CodexArtifacts/pretrained-reuse-r2/mask-feature-pool-fresh-replay-01
```

Fresh output paths mandatory; arrays/media/model weights stay outside Git.
This probe does not pass wider native corpus admission or permit opening its
post-hoc labels before full extraction/fresh replay/audit. No further closed
nearest-distance/cosine ranking, model fitting, model promotion or95–98% claim.
Next: after full native input admission, preserve these raw-derived stats in a
separate wider token bank, then define semantic role/memory supervision and fixed
validation controls before any readout training. Overall Omni quality remains open.
