# Physion++ train-only data quality audit — 2026-11-28

## Scope

This audit inspected only the existing `fullclip-v1/train.jsonl` manifest and
its 652 train videos. It did not open validation or sealed/test rows, load a
model, train a model, or modify the corpus. The inputs are already observed
development data and cannot serve as a new blind audit.

The helper and its tests are `scripts/audit_physionpp_train_data_quality.py`
and `tests/test_audit_physionpp_train_data_quality.py`.

## Integrity and coverage

| Check | Result |
|---|---:|
| Train records | 652 |
| Train manifest SHA-256 | `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b` |
| Unique media paths / SHA-256 identities | 652 / 652 |
| Video files present and matching declared SHA-256 | 652 / 652 |
| Source asset groups | 492 |
| Source asset groups with both labels | 67 |
| Labels | 339 No / 313 Yes |
| Scenarios | 10 |
| Scenario × exact recorded source-factor strata containing both labels | 75 |
| Possible matched No/Yes pairs across those strata | 159 |
| Contact-sheet sample | 150 clips: one No and one Yes per matched stratum |
| Sampled frame count matching metadata | 150 / 150 |

The fixed exact-factor review selection is at
`C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-data-audit-20261128-attempt-04`.
Its selection JSONL SHA-256 is
`38766b31a0b0af9048479236d556724ca47c374f9cedbb905780cebfe2be1333`.
The earlier attempt-01 through attempt-03 outputs remain untouched. Attempt-04
records source-factor/label cross-tabs and exact-factor-matched clips from the
train IDs.

## Data-quality findings and limits

The class ratio is close to balanced overall, but this hides sparse and
single-class scenario/object strata. For example, some friction scenario ×
object combinations have only No examples. A model could therefore learn
scenario, object, or rendering correlations instead of the physical outcome.
The matched 150-clip review set matches every recorded scenario factor,
including target object, for a manual check; it does not prove that labels are
visually inferable or that unrecorded simulation parameters are balanced.

Recorded source factors are also associated with labels in this small train
corpus: `zdloc=3` has 28 No / 6 Yes; `use_blocker_with_hole=0` has 27 No / 7
Yes; `zld=0` has 7 No / 34 Yes while `zld=2` has 36 No / 12 Yes; and
`remove_middle=1` has 15 No / 0 Yes. These are descriptive counts, not proof
of leakage or causal effects, but they show that overall class balance and
target-object matching alone are inadequate data checks.

The source archive mapping audit reports 832 metadata rows, 800 mapped rows,
one duplicate, and 31 unmatched rows. It maps 634 rows through named
directories and 166 rows through 52 placeholder directories. The placeholder
check confirms contiguous RGB/segmentation/PKL index structure and expected
row counts, but does not independently verify semantic metadata-to-video label
alignment. Train `source_target` values agree with the corpus `target`; that
checks serialization consistency, not the correctness of the upstream video
association.

The review sheets show matched clips and sampled trajectories at 128×72 pixel
thumbnails. This is a triage artifact, not an expert annotation. Human review
of label observability, mapping correctness, and likely nuisance cues remains
open. The dataset's previously inspected labels and splits are already
observed; do not use these rows for a fresh sealed-audit claim.

## Decision

This audit supports a data-first next step: independently review the fixed
matched clips and upstream metadata-to-video mapping before any further model
capacity or BERT-family experiment. A BERT encoder can only be a text-path
candidate; it cannot resolve video label alignment, visual observability, or
simulation shortcuts. Existing BERT-family experiments are task-specific and
do not justify a broad sweep or multimodal-quality claim.

No quality gate, model-selection gate, or R2 completion gate is closed by this
audit.
