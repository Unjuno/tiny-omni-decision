# Common native supervision and query alignment

## Fixed intent before execution

Continue Path A and the optional temporal-state comparison toward one <=200M
conditional Omni candidate. Preserve the existing frozen train/development
validation partitions and complete features. This unit binds every admitted
question to the correct observation and original-language candidate vectors.
It does not introduce new media, load calibration/audit, or claim quality.

Inputs: all six existing conditional text files (18,432/10,732 questions),
all8,424/512 Clevr-4 images,4,080/889 Lombard utterances,1,955/486 STARSS windows,
and512/400 CLEVRER questions from256/200 scenes. Keep all rows. Do not call these
previously observed development cohorts blind evaluation or universal coverage.

Task rendering is fixed before a fit: text and CLEVRER retain exact questions,
options and targets; image retains the existing four semantic questions with
all40 typed attribute candidates; speech asks each of the six Grid slots with
its complete existing vocabulary (including25letters); STARSS retains its
original12 binary event questions. No transcript, label, target index, source ID,
or task name enters the model input except the supplied question/options and
observation itself. IDs serve only joins, accounting and grouped evaluation.

Image option order is deterministic seed17 per asset; all four image questions
use that same order. Original text/video/event order is preserved. Speech uses
its declared complete slot vocabulary in the original dictionary order. Later
training option permutation must be applied consistently to labels and query
vectors and logged; this builder does not tune ordering on model predictions.

Use the already verified native banks: original DeBERTa text states/queries,
17x768 image tokens,8x16x16x768 video tokens, and both speech and event tokens
for every audio example. No old pooled visual/speech profile substitutions.
The output observation index contains feature references, not label-derived
features. New question/candidate pairs use precisely the existing original
DeBERTa FP32/eager/batch1 mean-token profile with no truncation. Reuse existing
pair vectors by exact raw-string pair hash; save new vectors separately.

## Integrity and coverage gates

Reject duplicate question IDs, invalid target/candidate mapping, inconsistent
cache/source IDs, wrong split, missing/extra feature assets, bad offsets, and
cross-split group/media/normalized-text identity overlap. Preserve scene-level
CLEVRER grouping and source/speaker/room audio grouping. Feature/input receipts
and each consumed bank are pinned before and after execution. Hash the final
ordered question manifest and feature/query mappings. Shared question wording
is allowed; observation identity is the split boundary. Different media may
share attribute labels; that is recorded, not treated as unseen composition.

Retain text source/slice metadata for all previously defined retention slices.
Record per-task/per-class counts and positive/negative event counts, including
STARSS train bell-positive0 and validation water-tap-positive0. These cannot
support a claim of all-class success. Joint decisions, independent OOD,
Japanese, natural images and free-form speech remain separate unmet gates.

## Execution

Test pure joining and target preservation before implementation. Build the
complete manifest deterministically from explicit paths only. Run a small
TRAIN-only query encoding probe first, then encode only missing pair vectors;
recompute all new vectors in a fresh process and require exact equality. Match
the prior encoder state hash/runtime and reuse a fixed existing TRAIN query
for full profile parity. No optimizer updates. Fixed local budgets: <=1hour,
3GiB CUDA,6GiB host peak,2GiB disk reserve. Stop on resource/numerical/hash errors.

This closes input binding, not training. Next freeze the balanced common
sampler/budget/selector/retention policy from actual coverage and throughput,
then execute the shared-readout baseline and temporal alternative. Quality must
be measured on the same complete task matrix. Old retention/adoption failures
remain unchanged; new corpus assembly does not supersede them.
