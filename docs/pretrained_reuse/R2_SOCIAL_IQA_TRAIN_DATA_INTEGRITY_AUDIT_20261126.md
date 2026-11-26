# Social IQa train data integrity audit — 2026-11-26

## Disposition

The supplemental Social IQa v1.4 train file is useful for task-dimension and
answer-origin stratification, but it does not resolve source-event grouping.
It also exposes a small number of duplicated questions with inconsistent
targets across two publisher train releases. Keep the existing model result as
historical exploratory evidence. Do not silently rewrite labels, remove hard
examples, or make another model-selection run on the already exposed split.

## Pinned inputs and boundary

- Author-provided archive:
  `https://maartensap.com/social-iqa/data/socialIQa_v1.4_withDims.tgz`
- Archive SHA-256:
  `5efd8c27d4edfd3b3aba807b0c157b1cc84a6a415c5537d9abec1342261f7224`
- Supplemental train JSONL SHA-256:
  `73334ac195909d112107a993fd7e6505d5a864c7e21f27f30fde2138e49dbdce`
- Supplemental README SHA-256:
  `1ed56d3ca3db5aebccf00b0b7fb5912d2cb7e4c56336b78029cf52b47bad5d6d`
- Included CC BY 4.0 license text SHA-256:
  `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`
- Compared publisher-train JSONL SHA-256:
  `4c5333dea02be6d212114a83f17f7bbcbdb53b3d692f476bd40dcd242b6233de`
- Existing custom train/validation split hashes are recorded in the machine
  receipt; the split artifacts were read-only.

Only publisher-train data and the existing custom train/validation artifacts
were read. The supplemental dev/test members were listed but not extracted or
opened. No model, predictions, publisher dev/test labels, or project sealed
audit was accessed.

## Findings

| Check | Result |
|---|---:|
| Publisher train records in each release | 33,410 |
| Unique normalized context/question/options keys | 33,381 |
| Exact duplicate excess rows | 29 |
| Duplicate-key groups with cross-release target-multiset disagreement | 5 groups / 10 rows |
| Conflicting groups in existing custom train / validation | 4 / 1 |
| Exact context overlap in existing custom split | 0 |
| Answer-source annotation arrays shorter than 3 choices | 31 rows (12 length 1, 19 length 2) |
| Stable ATOMIC event ID in supplemental schema | absent |

The five conflicting duplicate groups each have two identical text-and-option
rows. In the baseline release those copies carry different target labels; in
the supplemental release both copies carry the same target. One such group is
in the existing validation split, so its two identical inputs have opposing
targets there. The correct resolution is unknown; neither release was silently
preferred.

The supplemental file adds `promptDim`, `answerSourcesOrigins`,
`answerSourcesWithCor`, and character mapping fields. The nine prompt
dimensions can support reporting task mix, and answer-source fields can
support annotation-provenance strata. They do not establish annotator agreement
per example, and the 31 short source arrays must remain flagged rather than be
used as a filter. There is no stable source-event key, so the internal
context-group split still cannot be certified ATOMIC-event-disjoint. The
publisher paper states that its official partitions group contexts by their
originating ATOMIC event; that guarantee does not automatically apply to the
custom random split within publisher train ([paper, §3.5](https://aclanthology.org/D19-1454/)).

## Reproduction

The external attempt directory is
`C:\CodexArtifacts\pretrained-reuse-r2-local\socialiqa-source-group-audit-20261126-attempt-01\`.
It contains the pinned archive, train-only extracted members, the audit script,
and `source-group-audit.json`. The receipt hashes the input and split files;
the script compares normalized row-key target multisets so repeated identical
questions are not accidentally collapsed.

This is a data-quality diagnostic only. It does not revise prior scores, select
a model, or close an R2 task-quality gate.
