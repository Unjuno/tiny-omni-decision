# R2 native mask-region feature corpus — executed 2026-10-14

## Result

**PASS derived input admission on eight already-known train clips.** All18,441
native NMS survivors over256 times retained, with384 mean+384std per model mask.
Fresh separate CPU process matches all72 array shapes/dtypes/bytes. Independent
audit verifies full IDs/boxes/scores/counts/time/offset inventories and64 numerical
reference rows. This is not semantic recognition, QA, calibration or Omni quality.

| Process | Actual CPU body seconds | Scope |
|---|---:|---|
| Primary | 140.107228 |8 clips/18,441 retained regions |
| Fresh process | 142.902425 |All72 arrays recomputed/exact |
| Independent audit | 13.933642 |Full inventory +64 numerical references |

Total bodies296.943296s; all8 primary NPZ51,090,427bytes (~51.09MB decimal).
No model/GPU/new training/download/cloud. Added registered/trainable parameters0,
but all resident pretrained towers must still count in eventual <=200M accounting.
CPU memory peak/isolated throughput/product latency not measured. One18,441-row
bank is eight media assets/256 time inventories, not18,441 independent examples
or physical objects. Feature/storage increases and future readout cost remain open.

## Declared numerical representation and integrity

Reuse already-frozen TAPIR low32×32×256/high64×64×128 cells, piecewise constant over
256×256 raster. Foreground-pixel-mass-weighted FP64 mean/two-pass populationstd;
saveFP32. Native boxes remainFP64; no cast, normalization/bbox/GT/background pooling,
threshold change/TopK/deduplication. Empty masks/times keep counts/zero vectors/offsets.
Each row's identity is(video,time,native proposalID). Sources/cuts/uniformtimes match
exactly. All9 arrays for10903 also match the former1,070-row pilot byte-for-byte.

Producer's independent pixel-gather reference and separate audit's pixel-expansion
reference check **first8 input-order frame0 candidates per clip**,64 total. Mean/std
maxerror0.0 under fixed1e-5 absolute/0relative guard. Full72 saved arrays compare
exact, but do not claim independent numerical recomputation of all18,441 values.
Primary/fresh/source/config/hash checks pass before and after; NumPy seed17 unchanged.

Code/protocol committed before execution: `7f187a4635c15958839877be46d2f6d893552190`.
Effective configSHA `55fd81403f546167b919f49aaf2dd43b80eb7ef0cfdc8bca4b2ef8a8d5381eb6`.
[Companion JSON](R2_NATIVE_MASK_FEATURE_CORPUS_RESULTS_20261014.json) retains the
actual reports/file paths/hashes. Large NPZ/media remain outsideGit.

```powershell
python -m scripts.extract_mask_feature_corpus --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-feature-corpus-eight-01
python -m scripts.extract_mask_feature_corpus --reference C:/CodexArtifacts/pretrained-reuse-r2/native-mask-feature-corpus-eight-01 --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-feature-corpus-fresh-01
python C:/CodexArtifacts/pretrained-reuse-r2/audit-mask-feature-corpus-01.py
```

Existing outputs are immutable; future invocations need fresh directories. Independent
audit producer SHA `9e22f6451fbcdf9a520cc77d486788071896420aeb877c89f1fe8a23b1b1b75b`. See the
[fixed protocol](R2_MASK_FEATURE_CORPUS_PROTOCOL_20261014.md).

## Corrections and verification

Preexecution review found three issues, fixed before the primary: preserve actual
FP64 native boxes; verify pilot config against its original report instead of
repinning it; enforce600s/8GiB after final hashing. New regression tests failed
first and passed after fixes. No failed actual run was erased/retried. Existing
native/feature/helper configs/weights/artifacts untouched; no live intervention.

29 new CPU tests cover asset/cut/time/schema/dtype/empty/order/RNG retention,
exact native box bytes, protected config mutation and terminal resource limits.
Full suite1,321passed/1skipped17.30s, Ruff and manifests/fourcatalogs/policy passed.
Read-only preexecution review resolved allImportant findings. Exact publication-head
Actions is checked after push in a local status file. This unit opens no annotations;
the separately scoped posthoc training-only role census follows this admission.

Full four-modality/native joint/OOD95–98% and deployment/packaging goals remain unmet.
No new fitting, quantization, product promotion, main merge or paid compute.
