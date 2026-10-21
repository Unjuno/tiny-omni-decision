# Teacher interface: display order × label assignment

Frozen before forwards on2026-10-15. Follow-up to the
[TRAIN teacher screen](R2_SEMANTIC_TEACHER_RESULTS_20261015.md), not a new quality
benchmark or a search for the highest-scoring prompt. No student training.

## Question and four arms

The prior exclusion reversal changed BOTH candidate display order and the mapping
from semantic candidate ID to output letter. Separate these factors on the exact
same128 binding-exclusion questions and256 numeric-original questions. Keep all
32 states/source, questions, targets, candidate descriptions and40/4/5 choices.
No examples or difficult task types are removed. Other prior conditions are not
part of this prespecified mechanism comparison.

| Arm | Candidate lines | Semantic candidate ID → output letter |
|---|---|---|
|00|Original order|Original mapping|
|10|Reversed order|Original mapping retained|
|01|Original order|Reversed mapping|
|11|Reversed order|Reversed mapping|

For example, if canonical candidates map to A,B,C, arm10 shows the lines C,B,A
with each letter still attached to its original candidate. Arm01 shows the original
candidate order but letters C,B,A. Arm11 shows reversed candidate order and letters
A,B,C, exactly reproducing the previous reversal. Preserve letter case.

Total384 base questions ×4 arms =1,536 prompts /64 source-qualified TRAIN states.
The00 and11 prompts, tokenizer IDs and continuation label IDs must reproduce the
previous frozen inputs exactly before GPU execution. During execution all768
historical endpoint logits/predictions are compared with their pinned saved values.
Fail on drift instead of interpreting it as a factor effect.

## Fixed runtime and safeguards

Same pinned pretrained Gemma4, BF16, batch1, SDPA math, Windows process-local pread,
seed17, thinking disabled, no adapter/training/generation search, temperature1,
max2,048tokens/no truncation, same40min/15GiB VRAM/24GiB host/8GiB disk reserve.
The config and parent manifests fix all runtime fields. Prior model weights,
datasets, output directories and source code remain read-only. No audit/validation
dataset is opened. New code is isolated from the previous runner.

Prepare all1,536 rows/prompts/token IDs first. Smoke uses the first previously
hash-selected state/source, every question and arm:48 prompts. This is resource/
mechanics validation only. Run all1,536 only if the measured projection fits.
Fresh-process replay uses those same48 predeclared rows, all semantic candidate
IDs, logit tolerance1e-5, identical predictions. It is not a full second-run replay.
All768 historical endpoints are checked during the full run in addition.
Save every completed prediction before an assertion or resource stop.

## Estimation and interpretation

Report Accuracy/NLL/Brier/ECE by source, question type and arm, macro metrics,
changed-target pair correctness, full-vocabulary mass on legal labels, and
answer/probability agreement aligned by semantic candidate ID.

Predefined contrasts, separately per source:

- Display-order effect at original mapping:10−00.
- Display-order effect at reversed mapping:11−01.
- Label-map effect at original order:01−00.
- Label-map effect at reversed order:11−10.
- Interaction:11−10−01+00.

Estimate differences in Accuracy/NLL/Brier with paired source-state bootstrap,
10,000 draws, NumPy seed17, percentile95% intervals. Keep all four arms and all
contrasts; no winning-arm selection or multiplicity-adjusted significance claim.
This is an already-observed TRAIN cohort with only32 states/source. Do not call
it new blind evaluation, a general semantic capability measure, or a benchmark
ceiling. Order and label-map interventions are identifiable here; letter case,
specific letters, arbitrary-map difficulty and general reasoning are NOT separately
identified. Non-alphabetic display order may itself be unfamiliar to the teacher.

If order has little effect while map reassignment causes failure, prioritize
mapping/readout supervision over interpreting the failure as missing world facts.
If display order alone changes answers, report sequence-position susceptibility.
If both or interactions matter, retain the mixed result. If no factor effect is
resolved, close this interface diagnosis without inventing a semantic cause.
No arm automatically qualifies a teacher; original semantic/calibration deficits
and the four-modality product requirements remain.

After this one comparison, do not sweep alternative letters, prompts or seeds on
these states. Use the diagnosed issue to define at most one distinct next readout
or supervision comparison. In parallel, the next student data contract must keep
TRAIN, unseen syntax/combinations, old-task retention and final audit roles separate;
use checked ground truth as the baseline, and only validated teacher capabilities
as optional extra supervision. No product promotion or quantization in this stage.
