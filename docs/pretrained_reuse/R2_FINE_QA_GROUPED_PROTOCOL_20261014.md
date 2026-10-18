# Fixed grouped original QA resolution comparison

## Purpose and limits

Actual fine input admission and same-rule support19→1,957/5→34scenes are complete.
Test whether **input resolution alone** changes original decision quality in the
existing232,385-parameter readout. This new source-bound comparison does not extend
the old closed99train/257official-development optimization: a fresh readout excludes
20 whole train scenes, and the actual32×32 inputs are new. Old artifacts remain
read-only. No added capacity/loss/seed search or old official development scoring.

All99 previously known TRAIN videos/198 original questions are retained in the
combined pilot; no data/option simplification. They were observed in earlier work,
so the20scene selector is **not a new blind benchmark**. Actor/household separation,
encoder pretraining overlap and unseen question forms are not proven. Fixed question
templates/options repeat by design; video assets and both conditions stay grouped.
The full native Omni/95–98%/OOD/joint/<=200M release objective is unchanged/unmet.

## Data and prospective grouping

Keep immutable media/cuts,32times,256×512point-motion features, cached198×3×1536text
candidate features and original canonical answer mapping. Only train array paths
are opened from the existing mixed-split metadata; no official validation arrays,
labels, annotation archive or sealed audit. Reuse the original hash-bound99clip
cover auxiliary labels (98available); do not load old trained weights. Missing
auxiliary labels stay masked while all original QA remains scored.

Select20 whole scenes by lowest SHA256 of
`fine-qa:17:{video_id}:{media_sha256}`, with video-ID tie break; the other79fit.
Preserve original source order within each set. Save ordered IDs/source content
hashes/split hash BEFORE initialization/metrics. No target/hidden-support-based
stratification, retries or removing zero-support scenes. All two questions/options
from one source are in the same set; unique media IDs/hash/RGB checks and actual
question/text/point/spatial identity-time binding required. Do not call this actor-
disjoint or generalization to arbitrary new prompts.

## Two controlled arms

1. `coarse16`: actual32×256×387context tokens.
2. `fine32`: actual32×1024×387context tokens.

Both use unchanged `SpatialTrajectoryReadout(hidden64,latent_count16,heads4)` and
the same source-free point/text inputs, transient encode_state/score_candidates,
model parameter count232,385, fresh seed17 initialization, sampler seed17 order,
AdamW LR0.001/weightDecay0.01, batch8VIDEO PAIRS,50epochs (10updates/epoch,500/arm).
Original question CE +existing balanced cover-membership auxiliary weight1.
No encoder/projector load or training; no hidden-role loss, curriculum, new head,
token filtering, option shuffle/reduction, augmentation or architecture change.
No label/source-ID/bbox/event metadata enters forward. Logical frozen feature
producers: TAPIR31,072,327 and cached text17,389,824; plus readout48,694,536 total
logical components, not a verified all-modality resident runtime/package.

Primary comparison is matched **final50** with identical budget; also report
per-arm earliest strictly lowest selector original stable log-softmax NLL checkpoint.
Continue all50epochs regardless of quality dips, retaining best/final separately.
No selected control, accuracy, seed or final benchmark chooses hyperparameters.
Report both endpoints/whole curves; do not compare these40questions to old514-QA
metrics as if they were the same benchmark.

## Measurements and safeguards

At every epoch save full fixed fit158QA and selector40QA probabilities/log-probs/
cover outputs and original IDs/targets, Accuracy/NLL/Brier/ECE15/mean confidence,
initial/final and changed-target both-correct counts. These are fixed-pool train/
selector gaps, not rolling minibatches. Record CE/aux loss, gradient norm, consumed
video/question orders, unique/repeated exposure, update count,time,peak VRAM/process.
Best/final weights and optimizer/sampler/TorchCPU/CUDA states, exact config/base
revisions/data/code/protocol hashes saved; strict equivalent resume is NOT claimed.

Selected checkpoints only: normal, zero whole state, cyclic other-scene whole-state,
initial appearance broadcast and reversed context appearance (positions and point
motion unchanged for latter two). Fixed controls use no refit and never select
models. All candidates reversed within each3-option question must preserve scores
within1e-5/1e-6 and original answer identity. Report input effects and both-condition
performance, not only pooled Accuracy. Paired20-video bootstrap10,000/seed17/
percentile95 for final and selected fine-minus-coarse Accuracy/NLL/Brier; selector
and known-dataset effects remain explicit, no blind confidence claim.

Separate train-only GPU smoke: first16scenes from the full79fit seed17 sampler order,2updates/arm, evaluate those same
16only (no selector scores), exact final reload, same initialization/order/232,385params and finite loss/
gradients/three-option output. It must bind actual configs/input/split/code hashes
and saved weights/predictions before main. Measure VRAM/time, estimate main from
train+evaluation throughput (3,600s cap); do not start impractical main. Main repeats
each smoke's first2updates (parameter hash) before continuing; no setting change.
Fresh separate process verifies allfour selected/final weights against saved original
fullfit/selector log-probabilities/probabilities/cover outputs. Independent CPU audit recomputes
metrics/selector/exposure/bootstrap from saved outputs (not raw model semantics).

GPU preflight12GiBfree,peak allocation4GiB; hostpeak4GiB;8GiBdisk reserve and total
3,600sbody cap checked in batches/evaluation/end, not watchdog. Existing local
environment/FP32/TF32off/highest/deterministic unchanged. NaN/Inf/hash/gradient/
resource fault preserves incomplete run; no automatic retry or guard relaxation.
No legacy process intervention, cloud, main merge, ternary/recovery or promotion.

Read-only review before execution; tests first RED→GREEN for grouping, original
QA/candidate routing, zero-support retention, source-free context controls, dynamic
token readout/reuse and smoke binding. Ruff/full CPU/manifests/publication-head CI.
If finer inputs do not resolve conditional use or quality, close this fixed test;
do not extend the same budget or declare a pretrained-model ceiling.
