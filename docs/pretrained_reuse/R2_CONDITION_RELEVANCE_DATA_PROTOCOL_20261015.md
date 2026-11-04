# Independent condition-relevance supervision: data contract

This is corpus preparation for the next matched semantic comparison, not a fit,
calibration search or quality evaluation. Preserve every old run and selector.
The [strategy](NEXT_STRATEGY_20261015.md) keeps native four-modality quality,
unknown conditions, temporal/joint reasoning and <=200M total parameters open.

## Allocation and provenance

Seed17; the exact previous corpus manifest, exclusion identities, observed train
and validation, and retention payloads are SHA-pinned in
[config](../../configs/pretrained_reuse/condition_relevance_v1.json).
Reconstruct the previous hash-sorted binding reservation from its immutable
recipe, check its observed train/validation prefixes, and exclude its entire
512+128+256 allocation. Never open its transfer or raw state-partitions file.
The expected remaining space is1492 of the original10000 typed attribute states.
Sort this remainder using a new hash namespace, then allocate whole states:

| Partition | States | Questions | Purpose |
|---|---:|---:|---|
| train |512|8192|New request-relevance and correct-choice supervision|
| calibration |128|2048|Reserved binding-only calibration; no fit loader access|
| validation |128|2048|New wording development,16 field/query/form slices|
| transfer |256|16384|Four matched query/option wording conditions; no fit access|

No native asset is added. This is independent state-combination allocation,
not independent physical scenes or evidence of image/video quality. Remaining
468 states stay unallocated. New held-out generation validates labels/hashes;
it does not perform model inference. Old observed failures informed this design,
so later development gains are not a new blind product test.

## Public inputs and private labels

Each state has color, surface category, object count and geometric shape. Surface
categories include both materials and patterns: do not call every one a material.
All40 typed field/value candidates remain in every example, including the three
true facts irrelevant to the current question. Texture:star and shape:star are
distinct candidates. No target-field mask, answer leak or private task routing.

Six train questions per field are authored in the source bank; each train state
receives two, with deterministic rotation across the512 states. Both `field: value`
and `The field is value.` formats are used. Calibration and validation each use
two different authored questions per field. Transfer crosses seen/new questions
with seen/new option formats on the same256 states, same candidate order/IDs and
same correct candidate. Each factorial cell has4096 questions. Query style and
candidate style can thus be diagnosed separately without selecting on transfer.
The four literal old native-diagnostic questions are excluded from the new bank;
the old diagnostic remains a regression reference, not blind evaluation.

Each row privately stores40 booleans each for fact truth, request relevance and
correctness: exactly4,10,1 positives respectively; correctness is their
intersection. A future shared scorer may use these training labels; its public
pair renderer consumes only state, question and candidate description. Candidate
IDs, field, family, split, target and supervision never enter the model pair.

## Retained capabilities and selection scope

Copy all8192 previous numeric TRAIN rows byte-for-byte as individual JSONL lines,
plus all2048 legacy TRAIN rows. New binding replaces old binding for the proposed
training pool:8192+8192+2048=18432 questions. Only8192 are newly authored rows;
10240 are explicitly reused. No numeric task is removed or option count reduced.
Copy previous3072 new/5612 legacy validation rows byte-for-byte; their31 slices
remain alongside16 new binding slices. Verify state overlap across the complete
new+reused train/validation/calibration/transfer pools. The previous1024-step
experiment's selected0/failure is unchanged.

This does not freeze a future optimizer, loss coefficient or checkpoint selector.
Before fitting, specify matched arms, resources, retained-task gates and native
evaluation. The reserved calibration set covers binding only; it cannot calibrate
the full numeric/native/legacy product. Do not silently use selection/transfer for
that missing coverage. A later all-task calibration contract must be separate.

## Execution and validation

The builder refuses existing output directories, verifies source/code hashes
before and after generation, and records source commit, exact effective config,
state/row/source/field/view/target-position counts and payload SHA256 values.
It writes identity lists, not a raw cross-split state file. Development loading
admits only explicit train/validation payload names; calibration/transfer/audit
requests fail before file access. Output manifests are pinned externally.

CPU tests cover label semantics (including star ambiguity), private-label
poisoning, factorial equivalence, split rejection, old reservation, actual disk
generation, byte-preserved replay, corrupted hashes and forbidden payload access.
Generate into `condition-relevance-corpus-01` and replay into a different directory;
compare every artifact including manifest. No GPU model or optimizer is used.
Hash consistency and generation success are data-integrity evidence only.
