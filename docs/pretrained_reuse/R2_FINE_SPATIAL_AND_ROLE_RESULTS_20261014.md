# Actual fine spatial corpus and fixed weak role-support comparison

## Executed result

All99 known train clips now retain **actual32×32×384 frozen TAPIR features at32
unchanged source times**. Main finished naturally in302.73s; every direct16×16
regression is bitwise equal to the existing corpus. Fresh first-clip replay and
independent CPU admission pass. No old features/model/config/Teacher overwritten.

After admission, a separate fixed train-only support census uses the **same source
bbox proxies, roles, times and containment/exclusion rule** at16 versus32 geometry.
Same-hidden pairs increase19→1,957, supporting scenes5→34. Same-hidden long-gap pairs
are only30,first→last0. This supports finer **proxy supervision coverage**, not better Decision
Accuracy, visible physical identity, long-term memory or a model capability ceiling.
No training, checkpoint selection, evaluation/validation/audit tuning or promotion.

[Extraction protocol](R2_FINE_SPATIAL_CORPUS_PROTOCOL_20261014.md) ·
[Separate support protocol](R2_FINE_ROLE_SUPPORT_PROTOCOL_20261014.md) ·
[All records/hashes](R2_FINE_SPATIAL_AND_ROLE_RESULTS_20261014.json) ·
[All99 scene counts/times](R2_FINE_SPATIAL_AND_ROLE_RESULTS_20261014.csv).

## Actual feature retention and integrity

Released frozen31,072,327-parameter TAPIR checkpoint remains unchanged, SHA
`628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa`.
No new pretrained model or trainable parameter. All output cells come from native
low32 cells plus means of aligned2×2 high64 cells; FP64 accumulation/storedFP32.
There is no upsampling of coarse features or GT-directed region choice.
All32times,1024cells/time, original frame IDs and normalized positions are retained.

Source/protocol commit`474cef4cbdd0bb0c6bf15ad8add463546ae98518`.
ContractSHA`6762e033c9302c0cdf2d0a4133dcca1d2b4ce95ac48b093e3bead429883e96eb`;
main configSHA`602fa6e0b21d352b791b5aa1dbbce0a0b3262bb868dbd13ce96491b86b03dd02`;
main resultSHA`06fb6205e8baec945780878e0c1bb4999ad326760ffd8b37b3132b65a8d0c147`.
Primary2026-10-14T09:56:42.669801UTC→10:01:45.402663UTC,99/99terminal success.

| Verification | Actual scope/result |
|---|---|
| Same sampledRGB/media/cuts | Every99 train asset, original hashes exact |
| Old16×16 features | Direct raw-grid pooling exact on every99 forward |
| Full old low/high grid references | Eight preexisting train raw-grid archives exact |
| Repeated first forward | All low/high/fine values exact |
| Saved fine arrays | 297files, exact producer readback/schema/finite/hash |
| Fresh separate-process replay | Firstclip, all3array bytes exact |
| Independent numeric reference | All262,144 fine cells in8raw-reference clips, maxabs0 |
| Independent complete schema/geometry/hash | Every99clip/297files |
| Frozen state/gradient | StateSHA unchanged,requires_gradFalse/no gradients |
| Random states | TorchCPU/CUDA unchanged across forwards |

**Independent numeric reconstruction covers8/99clips**, not all99 raw forwards.
All99 do have old-coarse producer parity and full independent file/position/time
admission. Do not conflate these different verification scopes.

Fine appearance/position/frame arrays total5,021,824,896bytes (~4.68GiB) outsideGit.
The fourfold token count is spatial retention, not fourfold independent data.
99media/3,168sampled images/3,244,032cell observations remain correlated. Package
weights/200MB/runtime and quality gates have not been satisfied by this cache.

## Hardware, cost and commands

Existing RTX3080Laptop16GB, Python3.11.9, Torch2.6.0+cu124, NumPy2.4.6, PyAV18.1.0.
FP32,TF32off/highest,deterministic,seed17,pyramid_level0/extra_convsFalse,strictload.
Main peak allocated677,198,848bytes (~646MiB),reserved834,666,496bytes;
Windows working set1,459,249,152bytes (~1.36GiB). No environment upgrade or cloud.
8GiB disk reserve/4GiB process+GPU/1,800sbody caps pass. Existing Django processes
were untouched; no legacy process intervention or overlapping task GPU run.

| Successful process | Measured body seconds |
|---|---:|
| Separate GPU smoke | 15.908061 |
| All99 main | 302.729790 |
| Separate GPU fresh firstclip | 15.826677 |
| Independent full CPU audit | 14.157970 |
| Independent smoke numeric check | 0.893330 |
| Fixed CPU role-support census | 16.921040 |

Three CUDA process bodies total334.46s including load/hash/decode/save, not pure
kernel time. Main timed primary forwards sum25.628370s, excluding its repeat.
All successful measured bodies total366.44s, excluding imports/failed launcher;
not product latency, energy or total system utilization. Cloud charges zero.

```powershell
python -m scripts.extract_fine_train_context --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/fine-train-context-smoke-01
python -m scripts.extract_fine_train_context --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/fine-train-context-smoke-01 --output C:/CodexArtifacts/pretrained-reuse-r2/fine-train-context-all-01
python -m scripts.extract_fine_train_context --smoke --reference C:/CodexArtifacts/pretrained-reuse-r2/fine-train-context-all-01 --output C:/CodexArtifacts/pretrained-reuse-r2/fine-train-context-fresh-01
python C:/CodexArtifacts/pretrained-reuse-r2/audit-fine-train-context-01.py
python -c "import runpy; runpy.run_path('C:/CodexArtifacts/pretrained-reuse-r2/measure-fine-role-support-01.py', run_name='__main__')"
```

These commands describe executed fresh directories, not instructions to overwrite
or rerun completed corpora. Initial direct external support-script launch failed
before any output directory/data operation: `ModuleNotFoundError: scripts`.
`fine-role-support-launch-attempt-01.json` retains that failure. The same unchanged
code was then launched throughrunpy from the repository cwd; no model, labels,
settings or guard were changed. The successful census is the sole data execution.

## Fixed coarse/fine weak-support comparison

Support code/protocol commit`44eb6dc93821df611ed31e345f1b0a2da47e5fe4`;
configSHA`55aa6b76e8b1548a1b09948581291d3a486ba1b6251401ad3870c612bcd4a31c`;
resultSHA`db710803cab8a2dfb2e8ebc90eced6c6fbc3075b78291033bd9100ea3abe44c0`.
Every old16-grid label and all99explicit-pair role/stratum/exclusion counts replay
exactly before the fine result. Native fine positions/source times match the admitted
inputs. The finer grid is the sole rule/input geometry change in this census.

| Quantity | Coarse16 | Fine32 |
|---|---:|---:|
| Native cell observations retained | 811,008 | 3,244,032 |
| Uniquely role-labelled | 37,410 | 231,830 |
| Unknown | 764,018 | 2,971,954 |
| Ambiguous | 9,580 | 40,248 |
| Role-instance IDs with any cell support | 258/381 | 337/381 |
| Hidden-supported sampled times | 17 | 168 |
| Scenes with any hidden cell | 8/99 | 55/99 |
| Scenes with same-hidden pair | 5/99 | 34/99 |
| Cartesian opportunities | 363,331,584 | 5,813,305,344 |
| Eligible pairs | 2,111,587 | 63,944,919 |
| Same-hidden pairs | 19 | 1,957 |
| Same-group pairs | 773,325 | 22,789,829 |
| Different hidden/group | 2,114 | 219,989 |
| Different group/group | 1,336,129 | 40,933,144 |
| Same-cover subset | 183,457 | 6,168,480 |

Fine same-hidden support: adjacent1,927;gap8samples30;first→last0. One scene accounts
for456/1,957 (~23.3%). Sixty-five scenes still have zero same-hidden pairs, although
all99 now have some eligible group/hidden-role pair. Keep every scene and source
instance; no filtering, oversampling or balancing to inflate a model score.

Counts use exact per-instance histograms for full Cartesian windows, verified against
the old explicit-pair corpus. **Billions of pairs were counted, not materialized or
consumed by training.** Save all99 fine label arrays32×1024int64,25,964,928bytes total;
each reload matches shape/dtype/bytes. Every native unknown/ambiguous cell remains.
Per-scene/allstratum ledgers and overlapping exclusion counts are inJSON/local
reports; counts from correlated cells are not independent objects or test samples.

The same saved source-box proxies are reused; source interpolation is not rebuilt
independently. Bbox containment does not prove foreground purity or visibility,
and unavailable competing role boxes can hide ambiguity. Other objects are outside
this task-role namespace. Source masked flags/annotation starts are not physical
visibility/birth evidence. Source IDs/boxes/labels are supervision only, not runtime
seeds, input channels or semantic answer selectors. No annotations ZIP, validation
arrays/labels or sealed data is opened in this comparison.

## Engineering evidence and next gate

15new tests RED→GREEN; fullCPU1,385passed/1skipped16.98s, Ruff/manifests/fourcatalogs/
policy pass. Preexecution review's historical-source/actual-contract/ordered-plan
guards and label-readback shape/dtype guard were fixed before relevant execution.
All protected source/weight/config/media hashes rechecked at exit. Publication-head
CI is checked after push and saved locally. Large arrays/media/weights stay outGit.

Finer retention increases geometric training-label opportunity on more known scenes.
It does **not** establish task semantics, accuracy improvement or long-term hidden
memory. Same-hidden gap8 support is sparse and endpoint support absent under this fixed
rule. Next predefine a modest grouped, source-free correspondence/semantic-cue
probe that separates input use from geometry/prior shortcuts and retains zero-support
scenes; do not simply repeat the closed8-scene fit or interpret pair volume as
training budget/quality. Original QA, useful native multimodal fusion, unseen
condition/OOD calibration95–98% and lightweight release remain unachieved.
