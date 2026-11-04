# R2 all-native mask-region feature corpus — fixed 2026-10-14

## Purpose and contract

Move from the completed one-clip area-statistics probe to the already-admitted
eight known training clips×32times. This prepares foreground appearance for
semantic cue/cover/identity-memory supervision; **not a QA or quality experiment**.
Do not repeat closed nearest-distance/cosine ranking or failed readout training.
No annotations, dev/sealed data, new model/GPU work, downloads or cloud in extraction.

Use every18,441 NMS survivor across the unchanged256 native time banks, no TopK,
deduplication, score filtering or GT regions. Reuse the existing frozen32-time
TAPIR low32×32×256/high64×64×128 FP32 grids. Verify exact training asset/cut/media
identity, source hashes, identical time sampling and every source NPZ before compute.
New output directory; native, dense, pilot and all legacy artifacts read-only.

Keep the admitted foreground-area representation: piecewise-constant feature cells,
pixel-mass-weighted FP64 mean and two-pass population std, save FP32; concat384
mean+384std. No interpolation/bbox/background/normalization/clipping/learned projection.
The feature dimension/storage increases; future readout cost remains unmeasured.
Added registered/trainable parameters0; this does not remove resident pretrained
encoder parameters from eventual product accounting.

Save9 arrays per clip: mean,std,foreground_counts,frame_indices,proposal_ids,boxes,
scores,frame_offsets[33],observation_times[32]. Preserve native NMS order and all
IDs, including duplicates across times, zero-count empty masks and empty frames.
Keep native boxes FP64 byte-for-byte; only mean/std/scores are FP32. Verify pilot
config bytes against its original report hash, without repinning changed settings.
One candidate is indexed by(video,time,native proposalID), not a unique physical
object/independent example. Total unique media8; all are previously known train.

## Admission and reproduction

Require completed native result SHA
`ebe85d5a33cea86a4b045e2244e52b4c8dbc03a3e7ab80b037d77762a238bcda`,
complete audit SHA
`e3bed6dc93aa299f2904aaad00fa8b00eb771b1d4e1fe1668c4799963f6750cc`,
dense feature result SHA
`a000672b4500a8b4ae9bfde3f15e06d8103c56e986c2d7d3c17a6a77addf98f4`.
The former10903 pilot must reproduce all9 arrays exactly as part of the new bank.
Hash-bind complete configs/media/archives/new producer/helper code before and after.
Do not weaken parent checks or substitute a running snapshot for terminal admission.

At each clip's frame0, **first8 input-order NMS survivors**,64 total, compare mean/std
against independently gathered foreground-pixel feature values in FP64. Fixed
absolute1e-5/relative0 guard. This numerical reference covers64 only; all18,441 are
computed/retained/finite. No source-role/score choice of reference samples.
Primary save/readback exact. Separate fresh CPU process recomputes all8 clips and
must match all72 array shapes/dtypes/bytes, with same config/source hashes. Verify
primary result/config/archive hashes before and after replay, not just self-report.
Independently audit complete IDs/time/offsets/boxes/scores/foreground counts and
fresh arrays before posthoc **training-only** role/opportunity analysis.

CPU-only local process, seed17/NumPy RNG unchanged, preserve8GiB disk reserve and
256MiB initial output allowance. Body cap600s after imports checked after eachclip,
and after final source hashing, not a watchdog; no silent retries or larger cap
on failure. Retain partial outputs
and FAIL_INCOMPLETE traceback. Arrays/large sources remain outside Git; commit
code/test/protocol before primary execution; publish measured failures too.

Tests first fail for absent APIs, then cover independent hand-valued frame offsets,
all ID order/counts, empty retention, duplicate IDs only within a frame, unchanged
inputs/RNG, and asset/cut/split/time/channel/dtype/nonfinite/mass/schema rejection.
Existing extractors/helpers untouched. No unrelated live-process intervention.

```powershell
python -m scripts.extract_mask_feature_corpus --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-feature-corpus-eight-01
python -m scripts.extract_mask_feature_corpus --reference C:/CodexArtifacts/pretrained-reuse-r2/native-mask-feature-corpus-eight-01 --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-feature-corpus-fresh-01
```

This is an execution/data prerequisite, not completion of the95–98% native
text/image/audio/video/joint/OOD product goal. After admission, fix the label
opportunity/supervision and validation controls before any new readout fit.
