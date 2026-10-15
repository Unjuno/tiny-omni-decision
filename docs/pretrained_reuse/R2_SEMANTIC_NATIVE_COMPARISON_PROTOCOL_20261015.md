# Fixed semantic adaptation connected to the existing native-image probe

Prospective protocol, frozen before new model inference. The primary fit, selected
fresh replay and independent verifier have completed. Preparation while the fit
was active was kept outside the checkout and used CPU-only tests/source checks.
This experiment performs no training, calibration or checkpoint selection.

## Fixed question and inputs

Does the same semantic adaptation improve decisions on the original observed
256-image diagnostic, and what is lost when correct annotation facts are replaced
by facts predicted by the frozen image head from actual cached image features?

Preserve exactly `semantic-state-bridge-01`:256 TRAIN images,1024 questions,
all40 typed options in their original order and description, and these questions:

- What is the color of the objects?
- What is the texture of the objects?
- How many objects are there?
- What is the shape of the objects?

Use the original canonical four-field JSON state serialization and public pair:
`Question: <question>\nCandidate answer: <option>` / `State: <state>`.
No source/task/field-routing ID, target or oracle metadata enters the model forward.
Annotations are a diagnostic-only arm. This is not a product that accepts oracle facts.

The existing image head was trained on these images. All256 image hashes,252
annotation-state groups,254 predicted-state groups and308 combined groups are
checked against the saved admission and historical-exclusion records. The source
records, features, head, media, old outputs and both oracle/predicted exclusion
proofs are pinned and reverified. No new transfer/audit payload is opened.
The full512-image native validation is not admitted here:110 states are outside
the historical exclusion list. Do not remove those rows and relabel a filtered set.

## Models and controls

Same pinned DeBERTa xsmall70830722 parameters, FP32/eager, entailment-logit0;
softmax over all40 supplied options. All model parameters frozen. Compare:

1. Pretrained step0.
2. Fixed1024 endpoint, even though it failed retention; an unqualified diagnosis.
3. Validation-selected weights only if different. The actual selected0 aliases
   step0, proved by named tensor/dtype/shape/byte fingerprints; avoid duplicate work.

For each distinct model, run oracle facts, predicted Top1 facts, empty state,
empty question, and next-image donor-state control. The donor uses previously
computed oracle logits, with original and donor-corrected targets reported apart.
912/1024 donor targets change;112 do not, and one image pair has identical four-field
states. An unchanged prediction on an unchanged target is not itself insensitivity.
No-question scores are reused across four questions and must satisfy the25%
maximum Accuracy / zero both-correct changed-question-pair control.

Top1 facts lose perception uncertainty; this is a diagnostic bridge. Reuse the
verified original image feature/head outputs; do not claim a fresh image-encoder
forward or whole-Omni resident runtime from this comparison.

## Output, uncertainty and numerical checks

Save all IDs, option IDs, targets, predictions, raw logits and probabilities for
each model/control. Report overall/per-field Accuracy/NLL/Brier/ECE/confidence,
field routing, all-four-correct, changed-question-pair-both-correct, and intersections
of native-fact/oracle-decision/predicted-decision correctness. Retain the separate
native ten-option head metrics. Within-field scores are attribution only.

Endpoint-minus-step0 oracle and predicted Accuracy:10000 paired bootstrap
replicates, seed17, whole annotation-state clusters,95% percentile/linear quantiles.
There are248 singleton clusters and4 two-image clusters. Keep all sampled images
and questions and recompute the question-weighted ratio with its sampled denominator;
do not replace it with equal-weight cluster means. This is observed TRAIN diagnosis,
not blind independent uncertainty or a95–98% product claim.

Batch32, no truncation, maximum512 tokens. On the first fixed image, permute all40
candidates using seed17 and compare actual predicted IDs in displayed order;
reordered logits are only for numeric comparison. Ties that change chosen IDs fail.
Compare all40 options of the first question with singleton inference. Predeclare
absolute logit tolerance1e-5 and exact predicted IDs. Full fresh-process replay
requires identical artifact-name coverage and byte-identical scores/predictions/
metrics/inputs/config. No numerical tolerance is used to relax fresh full replay.

## Execution and protection

[Configuration](../../configs/pretrained_reuse/semantic_native_comparison_v1.json)
pins all terminal prerequisites and native inputs. The runner fails before torch
import if they are missing/incomplete/unpinned. Freeze tracked source/config/protocol,
run CPU tests, then a separate-process smoke on the original first2 images with
both distinct checkpoints and controls. Smoke measures mechanics/resources and
does not select weights or change the full cohort. The full256-image run uses a
new output directory, followed by a separate full replay. No previous output is
overwritten. No active process is stopped and no second GPU process is launched.

Local RTX3080Laptop16GB only, zero cloud cost, no dependency changes. Limits:
2h wall time,14GiB CUDA,12GiB host,512MiB output,4GiB free disk. Track actual time,
peak memory, candidate-forward counts and max token length. Expected forwards per
distinct checkpoint: full92520 / smoke1080, including order/singleton probes.
Resource/numeric/integrity failures are INCOMPLETE, with phase, successful pair
count, available resources and partial artifact hashes retained.

Example invocation after the frozen source is tested:

```powershell
python -u -m scripts.evaluate_semantic_native_comparison --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-native-comparison-smoke-01
python -u -m scripts.evaluate_semantic_native_comparison --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-native-comparison-01
# Fresh replay additionally requires --replay-reference and the completed primary result SHA256.
```

## Decision after this diagnosis

### Pre-inference platform correction

The first smoke attempt at source `04aa3c13c12516b8ea50e99b50e3518cc0899e0e`
stopped before torch/model import with Windows error206: the941-file Git verification
command was35615 characters, above32767. Its separate `semantic-native-comparison-smoke-01`
directory/log remain intact, with0 candidate forwards. Source inventory now reads
NUL-delimited tracked paths from a58-character Git command; the full pinned file
set is unchanged. CPU regressions cover1000long paths and every required untracked
file rejection. No data, weights, questions, numeric conditions or cohort change.
Use `semantic-native-comparison-smoke-02` for the next smoke; do not overwrite01.

### Interpretation

If oracle facts fail, condition/interface transfer remains a bottleneck. If they
work but predicted facts fail, quantify perception/readout loss. If both work,
move to a separately admitted independent native task. Do not select another
checkpoint, sweep prompts/temperatures or add another fit on this observed probe.
Speech, temporal video, genuinely joint decisions and the complete <=200M/95–98%
product requirements remain in scope and unmet.
