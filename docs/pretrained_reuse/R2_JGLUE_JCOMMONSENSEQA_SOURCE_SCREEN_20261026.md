# JGLUE JCommonsenseQA source and split screen

Date: 2026-10-26
Status: **REVIEW — no fitting or quality evaluation admitted**

## Scope

Screened the official `yahoojapan/JGLUE` release tag `v1.1.0` at commit
`6dbf79c766d3e64a1a575ef474cb654b6c9385d4`. Only the published train and
validation JSONL files were fetched in memory for schema, hash, and split
identity checks. The test file/split was not requested, opened, or evaluated.
No labels were used to fit, tune, or score a model. No GPU was used and no
dataset files were retained in the repository.

## Rights and task fit

The official JGLUE README declares CC BY-SA 4.0 for the benchmark. LINE Yahoo's
official software/data catalog says JGLUE may be used for model training and
evaluation. The publisher paper describes JCommonsenseQA as Japanese
multiple-choice commonsense QA built with crowdsourcing from ConceptNet seeds;
two distractors are also crowd-authored. The task is compatible with a
text-only supplied-option Decision interface, but it is a narrow Japanese
commonsense QA task, not evidence of general text understanding or multimodal
quality.

The available source states support research/training use, but do not resolve
whether a trained model or this repository's distribution must carry
ShareAlike obligations. Keep the dataset and any derivative model under REVIEW
until that product/distribution question is resolved. No data or model
artifacts are redistributed here.

## Pinned split integrity findings

| Split | Rows | Normalized unique question strings | SHA-256 |
|---|---:|---:|---|
| train | 8,939 | 8,817 | `9b55fae5ecb3aedd6f8ce5bc09196c3b629864668ec6c18eee4d65c0aa48229e` |
| valid | 1,119 | 1,117 | `0d8d76f3bfa0d174866939882faccdd01fbc2bcd5a76c43748ba0c40a7b3b8d4` |

All 10,058 rows had the expected `q_id`, question, five choices, and label
fields. `q_id` ranges are contiguous and disjoint (`0–8938` train,
`8939–10057` valid). However, after Unicode NFKC normalization, case folding,
and removal of whitespace/punctuation/symbols:

- 17 question-string groups cross the train/valid boundary.
- Those groups account for 24 train rows and 19 validation rows.
- Train itself has 92 repeated normalized-question groups (214 rows); valid
  has one repeated group (3 rows).
- The distributed records expose no ConceptNet seed, source-concept, or
  annotator/group identifier with which to establish deeper source-group
  separation.

Therefore the official valid split is not an independent group-held-out
validation for this project. Removing just the exact overlapping questions
would not prove ConceptNet-seed independence. A possible future research-only
development split could be carved from official train by grouping normalized
question text, but it would still have unknown source-concept overlap and
would need an explicit limitation; the official valid split should remain a
legacy/reference split. The public official valid labels are also exposed and
must not be presented as blind evidence.

## Decision

Do not download or retain the corpus, do not train on it, and do not run a
quality probe in this phase. The source is a plausible task-shape match, but it
does not currently satisfy the project gate for rights-cleared derivative
distribution plus independent group-level validation. Revisit only if the
model/data ShareAlike implications are resolved and source-concept grouping or
a newly collected independent evaluation set becomes available.

## Primary sources

- [Official JGLUE v1.1.0 README](https://github.com/yahoojapan/JGLUE/blob/v1.1.0/README.md)
- [Official JGLUE v1.1.0 JCommonsenseQA training split](https://github.com/yahoojapan/JGLUE/blob/v1.1.0/datasets/jcommonsenseqa-v1.1/train-v1.1.json)
- [Official JGLUE v1.1.0 JCommonsenseQA validation split](https://github.com/yahoojapan/JGLUE/blob/v1.1.0/datasets/jcommonsenseqa-v1.1/valid-v1.1.json)
- [LINE Yahoo official software/data catalog](https://research.lycorp.co.jp/jp/softwaredata)
- [JGLUE construction paper, LREC 2022](https://aclanthology.org/2022.lrec-1.317/)
