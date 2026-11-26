# R2 mask-region frozen-feature statistics — executed 2026-10-14

## Result and scope

**PASS for derived input integrity on one known training clip only.** No decision
quality, semantic cue-to-cover binding, checkpoint selection or product adoption
has been established. The wider native extraction remains unchanged and requires
its own terminal256-frame coverage, fresh-frame replay and complete input audit.

Use all32 observation times/all1,070 native NMS survivors from train10903;
retain every proposal ID/box/score/time/count. These are correlated proposals
from **one media asset**, not1,070 physical objects or independent examples.
No source roles, annotations, dev/sealed data, model or GPU loaded by these CPU
processes. Existing sources/configs/archives were hash-checked before and after.

## Representation and actual measurements

Treat frozen TAPIR low32×32×256/high64×64×128 cells as piecewise constant over
their256×256 raster support. Model-mask foreground pixel counts weight each
cell. FP64 two-pass weighted mean/population standard deviation are saved FP32:
384 means +384 standard deviations per proposal. No bbox/background pooling,
normalization, clipping, GT-selected proposal, TopK or ranking. Empty masks keep
zero descriptors and counts; the API preserves empty frame offsets in the runner.
This is a declared derived area representation, different from the old interpolated
point descriptor. **0 additional registered parameters**; future readout capacity,
storage/feature cost and quality must be measured before adoption.

| Actual process | CPU body seconds | Checked scope |
|---|---:|---|
| Primary | 9.742226 |1,070 rows,32 times,9 arrays saved/read back |
| Fresh separate process | 9.729345 |All9 shapes/dtypes/array bytes exact |
| Independent audit | 1.582023 |All IDs/boxes/scores/counts/times/offsets;8 numerical references |

Total measured bodies21.053594s, excluding process imports; CPU work overlapped
native GPU extraction, so these are not isolated performance/product-latency results.
Each NPZ is2,976,880bytes (~2.98MB decimal), mean/std shapes[1070,384]. Both archives:
`717c1c02b0ef7c4a7a89104b89cdcc8926149967457e9747295f8112619c476d`.

The producer expands feature cells to pixels for the fixed first8 input-order
frame0 survivors. A separate audit independently gathers pixel-to-cell coordinates
and computes masked-pixel FP64 statistics. Both mean/std maximum absolute errors
are0.0 at unchanged1e-5 absolute/0 relative tolerance. **Only8 numerical rows were
independently recomputed**; do not extend that reference claim to all1,070 rows.
All9 saved arrays were compared over their full inventory and all source masks'
foreground counts checked. Seed17 NumPy state remained unchanged in both probes.

## Reproduction and hashes

Producer source commit: `b83012e8b208d170c6402941e9139175f810a0e2`.
Effective config hash:
`75ebe886edadcb2431f6a0e7bfee460ac24234202e90a8000126555665a67e0c`.
The companion [JSON](R2_MASK_FEATURE_POOL_RESULTS_20261014.json) includes exact
reports, config, all parent/producer/media/archive hashes and artifact paths.
Arrays remain outside Git under `C:/CodexArtifacts/pretrained-reuse-r2/`.

```powershell
python -m scripts.probe_mask_feature_pool --output C:/CodexArtifacts/pretrained-reuse-r2/mask-feature-pool-pilot-01
python -m scripts.probe_mask_feature_pool --reference C:/CodexArtifacts/pretrained-reuse-r2/mask-feature-pool-pilot-01 --output C:/CodexArtifacts/pretrained-reuse-r2/mask-feature-pool-fresh-replay-01
python C:/CodexArtifacts/pretrained-reuse-r2/audit-mask-feature-pool-01.py
```

Those paths record executed outputs; the probe requires fresh output directories
for any future run. Audit script SHA:
`e9b9237bab56b6a264ae999d2b97bc0f39df1ce5e184119d8ec2ea70fb70b68c`.
The [fixed protocol](R2_MASK_FEATURE_POOL_PROTOCOL_20261014.md) precedes execution.
CPU tests cover area weighting/partial cells, population std, large constant zero
variance, empty/order/overlap/RNG/input preservation and invalid grids/masks.
Local Ruff, full CPU tests (1,292 passed/1 skipped,20.25s), manifest/catalog/policy
checks passed, and read-only publication review found no issues before publication;
exact publication-head CI is checked after push and retained in the local status file.

## Next gate

Complete and admit the already-running native eight-clip corpus first. Then preserve
this derived representation in a separate wider token bank and define semantic
cue/cover/identity-memory supervision with fixed validation controls before fitting.
No repeat of closed nearest-distance/cosine policies. Native audio/image/video/text,
joint/OOD95–98% quality and the full <=200M resident product remain unqualified.
No cloud cost, training, quantization, main merge, legacy artifact changes or live
process intervention in this unit.
