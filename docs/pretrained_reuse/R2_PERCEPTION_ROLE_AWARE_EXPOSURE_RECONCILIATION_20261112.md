# Perception Test exposure-role reconciliation — 2026-11-12

## Decision

Do not fit or score another Perception Test cohort in this work item. The
content-deduplicated candidate set from the 2026-11-11 MCQA exposure audit is
identical to the 1,518 validation video groups already registered as observed
in the cohort ledger. The remaining path-screened groups contain only
question/option strings also present in official TRAIN. They may be considered
for a separately justified in-distribution video transfer task, but they are
not fresh question-content evidence and their media/history eligibility is
unproven. No group is admitted here.

The source history remains `INCOMPLETE`. No gate is relaxed, no product result
is claimed, and the official test split and any sealed audit remain unopened.

## Read-only evidence

- The 2026-11-11 MCQA exposure audit scanned 248 Perception-prefixed artifact
  directories on C: and 2 on D:, using paths and bounded text/JSON metadata.
  It did not open media/model/feature payloads or the official test split.
- The broad inventory contained all 2,184 TRAIN and 5,900 VALID video IDs,
  but this included aggregate source indexes and is not evidence that every
  listed video was scored.
- Direct-path classification found 1,939 TRAIN and 1,270 VALID IDs with a
  local media, feature, prediction, or named cohort path. The 4,630 remaining
  VALID groups are only candidates without a direct path match; the audit
  explicitly says absence of a path is not proof of non-exposure.
- Exact normalized question-plus-unordered-options deduplication against
  official TRAIN leaves 2,886 questions in 1,518 VALID groups. The cohort
  ledger's existing `validation_evaluation` entry contains 1,518 unique groups.
  An exact source/revision-scoped SHA-256 join produced 1,518 overlaps and
  zero unmatched groups.
- Therefore the remaining path-screened pool consists only of groups whose
  questions/options duplicate official TRAIN content under that normalization.
  It is not a fresh task-content holdout. Participant and script-variation
  identities are also absent from the publisher metadata.

## Exposure roles and limits

The fail-closed checker unions source/revision-scoped group hashes across all
ledger usage roles. This is appropriately conservative for a new-cohort claim,
but `BLOCKED_PRIOR_GROUP_USE` alone does not distinguish metadata inventory,
label inspection, fitting, checkpoint selection, or scored predictions. The
TRAIN metadata entry is not proof that all 2,149 groups were used for fitting;
other Perception reports independently document label inspection and specific
fit/evaluation subsets. The ledger stays `INCOMPLETE` until a complete,
role-labelled source history is reconciled across reports and artifacts.

This distinction does not authorize reusing a group as “fresh.” It only defines
what a future source-wide audit must prove before an evaluation cohort can be
admitted. Do not rewrite the ledger or change checker behavior based on this
partial review.

## Reproduction and inputs

- Source audit report: `R2_PERCEPTION_TEST_MCQA_EXPOSURE_AUDIT_20261111.md`
  on `codex/r2-perception-mcqa-exposure-audit-20261111` at commit
  `7c517235`.
- External scan report:
  `D:/CodexArtifacts/pretrained-reuse-r2/perception-test-mcqa-exposure-audit-20261111-attempt-05/exposure-report.json`,
  SHA-256
  `8da8b59847ddad68da09e029735a248661fb1d68ac95a8442aa2c10b452e076b`.
- Ledger/checker source: `codex/r2-perception-cohort-exposure-20261112`,
  commit `e9f54d3649e2923999d6a6e80d64b3f9b979a3d6`.
- Source revision: `google-deepmind/perception_test@3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
- The candidate-to-ledger join hashed IDs in memory and emitted only counts.
  It read no media, model weights, feature arrays, test data, or sealed audit.
  No inference, training, GPU, cloud, or paid compute was used.

The join uses the ledger's canonical identity function:
`SHA256(json.dumps([source_id, revision, video_id], sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))`.
For every candidate video ID, its digest was compared against the union of
existing source/revision ledger group digests. The result was only the three
counts recorded above; raw IDs were not written to Git.

## Next gate

Find a different rights-cleared source/task or an independently verified new
cohort. Before any model run, freeze group and content splits, prove both-input
necessity for an M-2 task, and record modality-only, metadata-only, shuffled,
stale, and missing-input controls. Perception MCQA cannot fill that gate, and
its source-candidate AV results remain development diagnostics only.
