# ATOMIC 2020 text-data quality screen — 2026-11-26

## Disposition

**Not admitted as a direct single-target Decision benchmark.** ATOMIC 2020 is
a large, openly licensed commonsense knowledge graph that may be useful as
auxiliary text-representation data. The checked release has enough head
overlap between its official train and dev files to invalidate dev as an
independent event-held-out selector without further filtering. Most
head/relation groups have multiple annotated tails, so converting a row into a
one-correct-answer multiple-choice question would silently omit known valid
targets. The `none` tail's task semantics are not explained by the included
data README; it is not treated as a negative class.

No BERT, model, benchmark test split, project sealed audit, or GPU was loaded.
The official `test.tsv` was present in the downloaded archive, but it was not
extracted or read. The separate live Orbitune job was left untouched.

## Pinned source and rights

- Publisher code repository: `allenai/comet-atomic-2020`, commit
  `33129ff9fbab126a98ce03f10017c405a553aeab`.
- Official archive: `atomic2020_data-feb2021.zip`, publisher Google Drive file
  ID `1uuY0Y_s8dhxdsoOe8OHRgsqf-9qJIai7`, 12,580,048 bytes, SHA-256
  `47c5f362ab4a3ea58c4962eebfdfd1c5420d3780e74cd3fe09efeef64f941c2b`.
- The publisher README says the dataset is CC-BY; the archived `LICENSE`
  identifies Attribution 4.0 International. Preserve attribution and citation.
  This records the publisher's grant, not a separate legal opinion.
- The included README says each TSV row is a distinct `(head, relation, tail)`
  tuple. The actual merged files contain duplicate rows, as quantified below.

Primary references: [official publisher repository and license](https://github.com/allenai/comet-atomic-2020/tree/33129ff9fbab126a98ce03f10017c405a553aeab), [ATOMIC 2020 paper](https://ojs.aaai.org/index.php/AAAI/article/view/16792), and [publisher data archive](https://drive.google.com/file/d/1uuY0Y_s8dhxdsoOe8OHRgsqf-9qJIai7/view).

## Train/dev integrity and label shape

The audit reads only `train.tsv` and `dev.tsv`. Head and triple comparisons
normalize strings with Unicode NFKC, case-folding, and collapsed whitespace.
No row text or targets are emitted by the audit script.

| Split | Rows | Distinct normalized event heads | Exact duplicate rows | Normalized duplicate rows | `none` tails |
|---|---:|---:|---:|---:|---:|
| Train | 1,076,880 (1,076,864 valid) | 36,940 | 68,626 | 70,649 | 120,282 (11.17% of raw rows) |
| Dev | 102,024 (102,024 valid) | 2,962 | 7,410 | 7,616 | 13,156 (12.90%) |

Sixteen train rows have an empty third (tail) field; all rows have exactly
three tab-separated columns. Those rows are counted as malformed and excluded
from duplicate, head/relation, and triple-overlap calculations. Their
non-empty event heads remain in the event-group overlap check. No dev rows have
an empty field.

The official files share 404 raw and normalized event heads: 13.64% of dev
heads also occur in train. They also share two exact normalized triples. The
official splitter groups source components by head, but the merged files
still contain cross-component head collisions. Consequently the official dev
file must not be used as an independent event-held-out validation set as-is.

After excluding the 16 empty-tail rows, train contains 243,354 normalized
`(head, relation)` groups. Of these,
177,663 (73.01%) have more than one distinct normalized tail. A single-target
choice adapter would need a multi-positive scoring contract or would need to
exclude/alter valid annotations. Neither is acceptable for a faithful generic
Decision benchmark. Random negative tails would define graph-membership
classification, not necessarily real-world commonsense truth, because graph
absence is not evidence that a statement is false.

## Reproducibility

- Source-code audit: `split/split_atomic2020.py` and `split/utils.py` at the
  pinned repository commit; the source uses head sets while assembling the
  constituent graphs.
- Train TSV: 54,921,553 bytes, SHA-256
  `6a39e07cc35960b9b84330ca78011a69d722d04c0a45224b33fd3192441692b1`.
- Dev TSV: 5,484,092 bytes, SHA-256
  `ce18c961fbbf5b2fa2ef5b718999483bea1d85353db8e39f0dc8014c3c330501`.
- The audit reports malformed empty fields and excludes those rows from label,
  duplicate, and head/relation calculations. Their non-empty heads are retained
  for cross-split event overlap; blank tails are never coerced to `none`.
- Archived license SHA-256:
  `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`.
- Analysis implementation: `scripts/audit_atomic2020_train_dev.py`.
  It accepts only train and dev paths; it has no test-split input. The source
  archive, extracted train/dev data, and full machine-readable receipt remain
  outside Git under
  `C:\CodexArtifacts\pretrained-reuse-r2-local\atomic2020-data-screen-20261126-attempt-01\`.
- The final receipt was regenerated from the pinned train/dev hashes after
  correcting the audit to retain malformed rows' non-empty heads for split
  overlap checks while excluding those rows from label and duplicate counts.
- Verification: Ruff passed for the audit and focused test; pytest result:
  `2 passed`. `git diff --check` passed. No model or GPU test was run.

## Decision for BERT exploration

This screen does **not** justify another BERT-family fit. It establishes a
potential large training source but not a clean supplied-option task. If
ATOMIC is reconsidered, first create a source-head-disjoint split solely from
the source train partition, retain every annotation, represent multiple valid
tails correctly, and predeclare graph-membership versus commonsense-validity
as the actual target. A graph-membership result would remain narrow evidence;
it cannot establish general text Decision quality or replace the four-modality
Path A gates.

The existing BERT evidence therefore remains: use compact BERT as a text
teacher/reference only where the task data and identical readout support a
fair, fresh comparison. Prior exposed MASSIVE, JCoLA, QASC, Social IQa, and
common-corpus validation results are not reopened or retuned here.
