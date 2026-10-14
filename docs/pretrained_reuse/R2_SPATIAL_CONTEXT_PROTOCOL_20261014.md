# Frozen spatial context corpus protocol — fixed before execution

## Objective and scope

Prepare the missing later-appearance input for a controlled Path A Video+Text
Decision comparison, following the negative/mixed
[role-supervised experiment](R2_ROLE_SUPERVISED_QA_RESULTS_20261014.md).
Quality, useful multi-input contribution and the full <=200M Omni model remain
the objective. This cache is not a new quality result or a smaller replacement goal.
Autonomous user strategy authorization covers this bounded implementation.

## Corpus and frozen encoder

Retain all existing99train/257observed-development media IDs, source cuts and
32 ordered uniform time samples from the completed trajectory cache. Never open
sealed/test/audit data. Extraction receives only pinned media and prior feature
metadata; no source boxes/grounded IDs/questions/answers/semantic crop or event cut.
Do not change old teacher configurations, corpora, features or weights.

Use the same strict legacy TAPIR(pyramid_level=0,extra_convs=False),31,072,327
frozen parameters, pinned source/checkpoint. FP32, seed17, deterministic, TF32off,
four CPU threads, existing RTX3080 Laptop16GB. No upgrade, cloud or training.
PyAV RGB256/scaling[-1,1] and same source indices as previous sparse appearance
probe. The original motion cache still summarizes full clips; this cache encodes
only32 sampled appearances per clip,11,392 frames across356 videos.

## Whole-grid pooling and recorded positions

Source lowres is32×32×256 and hires64×64×128 at each time. Mean every2×2 lowres
cell and every4×4 hires cell to16×16; concatenate channels. All source cells
contribute, with no annotation-dependent sampling. FP64 mean accumulation then
FP32 output, no feature renormalization. Output32×256×384 per clip,12,582,912bytes
before file headers. Pooling may dilute small-object cues; do not claim fine-detail
preservation, target identity or Decision quality. Higher spatial resolution is
not selected based on validation outcomes in this step.

Record normalized x/y centers and actual source frame/(source_frames-1), in the
same time/row-major cell order,32×256×3 FP32. Record all32 source frame IDs int64,
original cuts/media/feature/source hashes, selected RGB hash and output hashes.
Store uncompressed NPY for bounded memory-mapped readout; no large arrays enter Git.
Worst-case appearance arrays4,479,516,672bytes across356 clips, plus positions,
frame IDs/metadata. Preflight requires reserve8GiB after expected cache storage;
do not discard old artifacts to meet the gate. No existing GPU job intervention.

## Execution gates

Before the full corpus, execute a separate-process first-train-clip smoke:
strict load/frozen state, all finite, exact repeated extraction/pool/readback,
unchanged CPU/CUDA RNG, same full grids versus prior first-clip spatial cache.
Exact prior-grid equality is mandatory where reference exists; no tolerance
relaxation. All eight prior pilot clips get this check during full extraction.
Check source/model/media/prior-feature hashes before extraction and after completion.
Every output/time/position readback must be bitwise. Complete356 unique IDs with
zero media/prepared-window/selected-RGB identity overlap across splits.
No partial corpus may be used for model selection or a full training claim.

Safety cap:1800s main body,4GiB peak Torch allocation,12GiB free GPU at start.
Stop on integrity/NaN/OOM/disk failure; preserve partial artifacts and report
incomplete execution. Do not stop for model-quality guesses, since no quality
evaluation occurs here. New independent output directory, no overwrite/resume
bypass. Persist progress/PID; confirm live state before any restart after interruption.

## Subsequent comparison policy

The next readout comparison will use identical model architecture, initialization,
sampler, corpus, auxiliary-loss availability, options and validation selector.
One arm broadcasts initial appearance over the same32 times/positions; the other
uses actual later appearance. Keep motion features in both. All99/257 original QA
videos remain; auxiliary labels never enter inference. Freeze the readout design,
budget and controls before fitting. Only then report original-question Accuracy/
NLL/Brier/ECE, changed-condition pairs, artifact cost and grouped uncertainty.

## Engineering checks

CPU RED→GREEN regressions cover every-cell contribution, row-major order, temporal
order, large finite mean accumulation, invalid shapes/dtypes/time IDs and positions.
Fresh-context review, Ruff/full CPU tests and manifests before committed-source
extraction. After execution, independently verify all cache hashes/counts/positions,
fresh-process reload and resource scopes; record result, commit/push, exact-head CI.

## Pre-run evidence

Separate train-only smoke01 completed in7.716229s. First-clip low/hi grids and pooled
outputs agree bitwise with the original spatial cache and repeated inference;
readback/RNG/frozen-state checks passed. Peak Torch allocation677,198,848bytes,
reservation834,666,496bytes. Source state before/after SHA
`3c92ad70854f549bc46fde282f4248e949d733fadffcfa2b060afada2ea6dd4a`.
Config SHA `6909060dc3e5d2953f4caa8f8725b0906e238a36c78c03c348920f8f7df6fd94`.
Smoke artifact `C:/CodexArtifacts/pretrained-reuse-r2/spatial-context-smoke-01/`;
source files were uncommitted but fingerprinted in that smoke config. Full extraction
uses the committed implementation. Fresh-context read-only review reported no
findings. Ruff passed;1019 CPU tests passed,1 skipped in16.19s; all manifests/four
catalogs/policy audit passed. Full-corpus status remains pending until executed.
