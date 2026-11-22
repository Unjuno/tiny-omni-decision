# R2 native numeric FIT field-order audit — 2026-11-22

## Question and scope

Does the fixed TRAIN-only numeric FIT corpus require a learned DeBERTa field
locator, or can its serialized state schema identify fields from structure
alone? This is a source/task-shape audit of the already fixed 6,144 FIT rows
and 768 FIT state groups. It is not a model result, independent evaluation,
or product-quality measurement. No official validation, prior holdout,
calibration, or sealed-audit data was loaded.

## Verified inputs and procedure

- Numeric TRAIN JSONL SHA-256:
  `c34b0d5ff04707b5af059f0886117583efd59808f5ec217fd995333aab644f9d`.
- Frozen split manifest SHA-256:
  `2a595eee6f442560c83a2f18a22011c68125169058bf70251538dc08d4fde6b2`.
- Exact selected scope: all 6,144 FIT row IDs and 768 FIT groups from
  `native-text-programmatic-group-holdout-v1`; every selected source row was
  checked to remain official `programmatic_decision_v1` TRAIN data.
- Counted the JSON key sequence in every one of the 24,576 candidate objects.
  Recomputed the six min/max labels directly from parsed structured values and
  checked them against the same FIT source labels.
- CPU-only metadata/rule audit. No model, tokenizer, GPU, optimizer, or new
  training data was used.

## Result

All 24,576 objects have exactly one key order:
`cost`, `id`, `risk`, `time`. Therefore the field position is constant in this
corpus: `cost` at object position 0, `risk` at position 2, and `time` at
position 3. Directly applying the declared min/max rule to each structured
state reproduces all 768 labels in each of the six extremum tasks (4,608/4,608
FIT labels).

The 100% figure is a deterministic source-rule consistency check on the same
FIT examples, **not** an ML accuracy estimate. It shows that this particular
structured benchmark cannot establish whether a BERT-family model learned
field localization: a parser can recover each field without any learned
representation. It also does not establish general instruction following,
OOD performance, or held-out quality.

## Consequence for BERT-family work

The existing MiniLM and DeBERTa reports remain valid for their declared
interfaces, but this audit weakens the case for another capacity sweep on this
fixed numeric corpus. The earlier frozen-DeBERTa numeric-span result used gold
field/object alignment, and its high OOF magnitude correlation does not show
that the model learned to locate fields or follow the requested operation.

Do not modify the frozen corpus or tune against its already exposed FIT OOF
results. A further BERT-family test must use a separately frozen task/source
whose surface form does not expose the answer through invariant serialization,
with independent source groups and full supplied-option probability metrics.
If a structured-input task is used, keep a deterministic parser/rule as a
baseline and explicitly test schema/key-order changes. No model was selected
or promoted from this audit.

## Reproduction receipt

- Source commit: `83bb474904a9aa77ffb512ca5f344d378271631e`.
- CPU command implementation: a read-only Python 3.11.9 audit using the two
  hash-pinned input files above; local artifacts were not changed.
- Machine-readable result:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\numeric-field-order-audit-20261122-attempt-01\audit.json`.
- Result SHA-256:
  `a69fab8e84cb8b93cc8c40c970aedca1020fd5dfeb0d37219a015a400ddd4f39`.
- GPU used: no. Training performed: no. Sealed audit loaded: no.
