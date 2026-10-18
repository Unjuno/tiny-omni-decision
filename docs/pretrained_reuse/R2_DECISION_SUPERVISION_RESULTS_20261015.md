# Independent decision supervision: executed corpus and selection contract

2026-10-15. **Execution PASS; model quality NOT EVALUATED.**
[Frozen protocol](R2_DECISION_SUPERVISION_PROTOCOL_20261015.md) ·
[Machine-readable evidence](R2_DECISION_SUPERVISION_RESULTS_20261015.json).

This completes the data/selection deliverable from the latest strategy.
It does not complete the Omni product or establish improved model accuracy.
No model was loaded, no optimizer update occurred, no previous checkpoint or
corpus changed, and no sealed audit was accessed.

## What actually ran

Frozen source commit: `f399216db6233f6a67d38f4149a410d13610c36c`.
Effective configuration and its SHA-256 are pinned in each corpus manifest:
`configs/pretrained_reuse/decision_supervision_v1.json`.

```powershell
python -m scripts.prepare_decision_supervision --output C:/CodexArtifacts/pretrained-reuse-r2/decision-supervision-corpus-01
python -m scripts.prepare_decision_supervision --output C:/CodexArtifacts/pretrained-reuse-r2/decision-supervision-corpus-replay-01
python C:/CodexArtifacts/pretrained-reuse-r2/verify-decision-supervision.py
```

Both corpus processes exited0. Primary UTC09:57:27.432–09:57:53.963 (26.531s);
fresh replay UTC09:58:03.961–09:58:30.266 (26.305s), totaling52.836s including
interpreter startup, generation, hashing and file I/O. Local Python3.11.9 on
Windows10.0.26300. No GPU/model runtime, cloud cost0; electricity and peak host
memory were not measured. Each output directory is58,176,526bytes; remaining
disk at publication was12,792,758,272bytes. No corpus payloads or weights enter Git.

## Actual coverage

| Partition | New binding states / questions | New numeric states / questions | Total states / questions |
| --- | ---: | ---: | ---: |
| train |512 /8,192 |1,024 /8,192 |1,536 /16,384 |
| selection validation |128 /1,024 |256 /2,048 |384 /3,072 |
| development transfer |256 /2,048 |256 /2,048 |512 /4,096 |

Total: **23,552 new questions,2,432 distinct states**, not23,552 independent
scenes. Training consumption remains0. Native assets remain0.

All11,264 binding questions retain40 choices. Each has exactly one true/relevant,
three true/irrelevant, nine false/relevant and27 false/irrelevant candidates.
All12,288 numeric questions retain their original task widths: four for the six
field/direction comparisons, five for count and constrained selection. The
independent verifier checked all502,784 candidate-label positions.

| Numeric risk-threshold result | train | validation | transfer |
| --- | ---: | ---: | ---: |
| count0 / no eligible item |211 |59 |66 |
| count1 |201 |49 |54 |
| count2 |191 |58 |44 |
| count3 |204 |38 |52 |
| count4 |217 |52 |40 |

No hard rows or empty eligible sets were removed to balance targets. Actual
target-position distributions are retained in the JSON. Candidate order is
state-dependent, fixed across condition changes, and not reduced by the target.

The accounting key `templates` is a historical naming imprecision in this frozen
producer: it counts distinct **rendered question strings**, including threshold
numbers (424/310/300). It does not count that many independent syntactic families.
There are four/two/two binding families in train/validation/transfer, respectively;
numeric wording uses one fixed split-specific form per operation. These authored
templates and known vocabularies remain a limited development distribution.

## Integrity and retained capabilities

- 23 exact local source files were hash-pinned. The builder opened only these
  declared TRAIN/development inputs, and no model/audit file.
- Exclude 7,612 previously used binding states and 3,328 numeric equivalence classes
  from17 historical numeric files. Numeric grouping removes item labels/order,
  cost/price aliases and positive global affine changes with integer arithmetic.
- Including legacy original-task states, the union of exclusions is 12,634 classes.
  New split state overlap and normalized full public-content overlap are0.
- Existing replay is byte-identical: 2,048 questions / 1,090 states, comprising
  1,024 original and 1,024 curriculum TRAIN questions. This is reused coverage.
- Existing validation is 5,612 questions / 1,244 states: 2,540 original typed,
  2,048 curriculum and 1,024 binding questions. Three individual copies are
  byte-identical; the combined file concatenates their unchanged JSONL bytes.
  Legacy train/validation state overlap is0.

The original sets are observed regression evidence. This preservation does not
turn them into blind evaluation. New development transfer is excluded from the
normal training/selection loader and has received no model evaluation.

## Selection safeguards implemented

The immutable selection schema has **31 required slices**:

- 8 new binding family-by-field slices and 8 new numeric task slices;
- 4 old binding fields, 8 old curriculum tasks and 3 original typed task types.

Baseline and every checkpoint must match all exact slice counts. Only validation
purpose is accepted. Missing slices/counts, impossible finite metrics or invalid
steps fail closed; nonfinite candidate metrics are ineligible. For each slice,
Accuracy loss<=1pp, NLL increase<=.02, Brier increase<=.01 and ECE increase<=.02
are required relative
to a prospectively fixed step0 initialization. Among eligible checkpoints,
minimum equal-slice macro NLL wins, including step0, with earliest-step tie-break.
These are research retention rules, not the95–98% product acceptance criterion.

The protocol also fixes advancement as improvement in the minimum Accuracy over
all16 new validation slices, with a paired whole-state uncertainty requirement.
That bootstrap and the actual metric/experiment runner still need implementation
and validation before any training. No model/optimizer/budget is silently chosen
by this corpus stage.

## Problems found and fixed before the freeze

Review exposed two selector holes: baseline counts could ignore expected corpus
counts, and impossible finite metric values could be admitted. Regression tests
now reject both. Historic numeric aliases and affine views are explicitly grouped.

Actual legacy files also contained389 prose TRAIN rows and959 prose validation
rows. Requiring every state to parse as JSON failed on these valid old records.
The fix preserves their bytes and computes whitespace-normalized prose identity;
focused red-to-green tests and the real pinned legacy gate pass.

## Reproduction and verification

All12 payloads plus the deterministic manifest (**13 files**) match the full
fresh-process replay byte-for-byte. Both manifest hashes:

`f2fac1bb77e762748e9ce49077009bb69a50d7c2869b93c27de6ffbeb7c55d17`

| Payload | SHA-256 |
| --- | --- |
| train |ecadf350f1003842703a5f41aeb2eb5ab760b236130c4d1f0543b9b1886a53b3 |
| validation |01cd6c8f55df5e68770408b8825711e9b7f2ea3ec7e9a3022725bbb4e5a0f536 |
| transfer |85a1f1c46d7bb956b6d2718787ba38adc55fd16f90106d7bb0300c5a5f11f145 |
| selection schema |a6d7aa74182b2f681819537d02d3befe44b10aeee6741370df736398360bbe63 |

The independent readback script uses no production-module imports or model
dependencies. It recomputed every answer and auxiliary label, required candidate
sets, group identity, cross-split/exclusion checks and old-byte preservation.
It is saved at `C:/CodexArtifacts/pretrained-reuse-r2/verify-decision-supervision.py`,
SHA256 `af358cf0519caea10939a523bb39a4481a0d36d6d404ed9d3b16138624117387`.
Its result and both execution records are included in the companion JSON.

Ruff passed; full CPU suite **1,671 passed /1 skipped** (23.96s);25 targeted tests
passed;25 manifest commands passed. The first manifest harness flattened a
PowerShell argument array and failed with `No such command 'v'`; log01 is retained,
the corrected command list in log02 passed. This was not a dataset/model failure.

Frozen-source [CI1765 succeeded](https://github.com/Unjuno/tiny-omni-decision/actions/runs/38716810272).
Final report-commit CI evidence will be saved after push as
`C:/CodexArtifacts/pretrained-reuse-r2/decision-supervision-results-ci-01.json`.

## What remains

Next freeze one bounded semantic training comparison: exact initialization and
whole-product parameter budget, rendering, trainable modules, loss, sampling,
optimizer schedule, steps and uncertainty implementation. Use this frozen
validation/retention contract, and evaluate transfer only after selection.
Then connect successful decisions to real input features and temporal/joint
tasks. No model score improved in this stage; all native four-modality95–98%,
whole<=200M and product runtime/quality requirements remain open.
