# R2 learned mask-pair correspondence — fixed 2026-10-14

## Question and scope

Does admitted foreground appearance help a small shared scorer associate source
instances across times beyond model-box geometry alone? This is a **GT-eligibility-
conditioned, known-training-clip representation diagnostic**, not runtime semantic
cue selection, original QA, a sealed test or the Omni95–98% quality result.
All8 clips were previously inspected. Eight scene-held-out folds prevent within-
scene train/evaluation leakage but do not make those clips a new blind benchmark.

Frozen SAM2/TAPIR feature/parent corpora and all legacy models/configs unchanged.
No new pretrained model/GPU inference, cloud, encoder training, quantization or
product adoption. Existing raw18,441 regions remain immutable; labels are auxiliary
training supervision only and never input channels, runtime seeds or GT best masks.

## Pair corpus, eligibility and controls

Use every native survivor's admitted384mean+384std, model FP64 box and original
observation time. Source-case boxes from the completed training-only census assign
proxy IDs with the existing **IoU>=0.5 plus own foreground-point inside box** rule.
Own points come from model masks, not GT; unique match yields local source trackID,
unknown=-1/ambiguous=-2/empty retained in eligibility accounting. Sparse box range/
interpolation/visibility limits remain. IDs may supervise labels, never features.

For every clip enumerate all native NMS Cartesian pairs in56 fixed windows:
31 adjacent times,24 gap8 windows,first→last. No target/score-based best-mask choice.
Train/evaluate only pairs with both endpoints uniquely proxy-labelled; target1 iff
same local source instance, otherwise0. Save all/eligible/unknown/ambiguous counts
and their overlap, per clip/stratum. Excluded raw inputs are not erased or relabelled
negative. This annotation-derived eligibility limits interpretation of every score.
Pair rows/correlated masks are not unique media or independent physical objects.
Proxy uniqueness is among available source boxes; absent competing boxes cannot
certify complete source-instance membership. These are weak geometry labels.

Geometry5: absolute centerXY difference, absolute log width/height difference,
absolute original-time gap/(clipFrames−1). Appearance1536: absolute difference and
product of768mean/std channels. Symmetric1541-vector contains **no source box,
source ID/role, video ID, pair target, proposal ordinal, text or QA option ordinal**.
No absolute scene-specific coordinates. Normalize each arm's features with means/
populationstd from its7 training scenes only, scale floor1e-6 for constant features.

Compare fixed arms:
1. Geometry-only5→64GELU→1 (449 trainable parameters).
2. Geometry+appearance1541→64GELU→1 (98,753 trainable parameters).

Output logits[0,sameScore], options[different,same], softmax probabilities. Count
all encoder parameters separately for eventual Omni product; this head alone is
not a release. Save normalizer buffers/weights and actual file sizes.
The arms differ in parameter count as well as inputs; their difference alone
cannot attribute gains solely to appearance. The fixed-weight mismatch control
tests input use without changing head capacity, not semantic causality.

No sampling policy/hyperparameter search. For each held-out clip train on all seven
other scenes; seed17 reset per arm/fold, identical training pair order. AdamW constant
LR5e-4,weightDecay1e-4,batch128,clipGradNorm1,512 updates. Deterministic shuffled
without-replacement cycles; record unique/repeated pair exposure, underlying assets/
source-instance counts. Repeated proxy pairs are not new source data. No early
stopping/checkpoint selection from held-out labels; final512 endpoint is fixed.
All8 held-out folds mandatory. If a pool lacks classes/strata, fail or explicitly
report unavailable support rather than silently dropping a scene/stratum.

Record fixed learning curve0/128/256/384/512: train/evaluation Accuracy,NLL,Brier,
ECE15 bins, per stratum/counts, class recalls/balancedAccuracy, pair exposure and
gradient norm/time. Report train-vs-held-out gaps and aggregate scene macro metrics.
Also train-prevalence prior baseline; don't call majority correctness correspondence.
Final appearance-arm control permutes **only** region appearance independently
within each time (query offset1,destination offset7,wrap), leaving boxes/times/labels
fixed. No new fit/selection; report all control outcomes, not best arm.

## Execution and evidence

CPU-only existing Torch,2 compute threads,1interop,deterministic algorithms. Build
and save hash-bound pair corpus/config first; fresh directories. Require all native/
feature/fresh/full audits and census/scalar audit hashes before labels/features.
Validate7/1 scene separation and complete counts before fit. Body cap600s, process
peak working set<=4GiB, preserve8GiB disk reserve; checked between stages/updates,
not a watchdog. NaN/Inf/gradient/hash/resource faults preserve FAIL_INCOMPLETE.
No silent retries, metric-guided budget extensions, seed search or automatic resume.

Separate CPU smoke: first held-out clip,two arms,8 updates, same corpus/config,
finite gradients/weights, correct parameters, final save/reload exact predictions.
Smoke does not choose hyperparameters or count as main learning evidence. Only
after complete smoke run all16 fits at512 steps, retaining terminal weights/history/
predictions/per-pair IDs/targets/probabilities/control and all attempts. Primary
random state is reset in its own process. Main final reload must match all held-out
probabilities/log-probabilities exactly; inputs rehashed at exit.

Scene-paired bootstrap of final appearance-minus-geometry Accuracy/NLL/Brier,
8 scene means,2,000 replicates,seed17,percentile2.5/97.5. Report every scene and all
strata; eight groups give weak uncertainty and dataset selection remains known.
No repeated fitting or changing masks/eligibility/thresholds after outcomes.

New tests first fail, then cover hand-checked windows/eligibility/unknown ledger,
scene grouping, symmetric source-free pair arithmetic, train-only normalizer and
real binary scorer gradients/schema/parameter accounting. Review before execution;
CPU suite/Ruff/manifests and exact publication-head CI before completion claims.

This learns an auxiliary correspondence primitive. Even a high pair score would
not establish cue→cover memory, full runtime unlabelled proposal handling, native
Audio/Image/Text, joint/OOD95–98%, deployable package or product promotion.

```powershell
python -m scripts.train_mask_pair_association --prepare-only --output C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-corpus-01
python -m scripts.train_mask_pair_association --corpus C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-corpus-01 --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-smoke-01
python -m scripts.train_mask_pair_association --corpus C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-corpus-01 --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-smoke-01 --output C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-learning-01
```
