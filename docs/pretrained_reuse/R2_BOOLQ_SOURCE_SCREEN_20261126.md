# BoolQ source screen — 2026-11-26

## Disposition

**`REVIEW_NOT_ADMITTED` for model training or quality evaluation.** BoolQ is a
good task-shape match for a BERT-family text Decision probe: a natural yes/no
question is paired with a passage, and the intended output is a binary answer.
The pinned Hugging Face card declares CC BY-SA 3.0, which is not in this
repository's explicit permissive-license allowlist. The source also does not
document a stable group identifier suitable for proving article-disjoint
splits, and its publisher test labels are unavailable. Do not ingest its split
files or train/evaluate a model unless the rights and grouping gates are
resolved under the project policy.

## Pinned metadata inspected

- Official dataset repository: `google-research-datasets/boolean-questions`,
  commit `90af34107399cc7a446b373dc4ee35b8001da7c2`.
- Hugging Face dataset repository: `google/boolq`, revision
  `35b264d03638db9f4ce671b711558bf7ff0f80d5`.
- Pinned Hub README: 6,570 bytes, SHA-256
  `07447668f46c7c1e62e24b650ea2a73f1ab34f2b3759b1587c7b7a81b4bf1cf4`.
- Pinned Google repository README: 2,675 bytes, SHA-256
  `0afe70d7cdbb644637592212f0cb15339389910bf00f35764ec3c545257aa663`.
- The Hub card declares `cc-by-sa-3.0`. The project allowlist in
  `src/tiny_omni_decision/dataset.py` accepts CC0-1.0, CC-BY-4.0, MIT,
  Apache-2.0 and BSD-3-Clause; it does not accept this license.
- The official README reports 9,427 labeled train rows and 3,270 labeled
  development rows. It describes 3,245 test rows as unlabeled; the pinned Hub
  file listing contains train and validation Parquet files only.
- The official release schema describes question, passage, answer and page
  title, but does not document persistent Wikipedia page IDs, annotator IDs,
  or a source-group split. The title might support an approximate article
  grouping if rights are cleared, but no group-overlap audit was performed.

## Label and task quality

The paper describes naturally occurring search questions paired with a
Wikipedia passage. Crowd annotators judged whether a question was clear,
located evidence in the passage, excluded unanswerable questions, and labeled
the answer yes or no. This is a stronger and more task-grounded label process
than a synthetic teacher-agreement target. It still does not establish
independent annotator reliability in the released rows, source-disjoint
generalization, or high accuracy on the project's broader Decision tasks.

If rights and grouping gates are later cleared, a valid use would require a
frozen article-grouped train/validation protocol, audits of label and split
quality, and a BERT-versus-existing-text-path comparison on identical examples
and probability metrics. No result should be generalized beyond passage-based
binary text decisions.

## Access boundary

This was a metadata-only screen. The pinned README/card and paper metadata were
read to assess task fit and licensing. No train, development or test data
files, model weights, features or labels were downloaded or processed. No
training, evaluation, checkpoint selection or GPU work was performed.

## Primary references

- [Official Google BoolQ repository at pinned commit](https://github.com/google-research-datasets/boolean-questions/tree/90af34107399cc7a446b373dc4ee35b8001da7c2)
- [Pinned Hugging Face dataset card and license](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5)
- [BoolQ paper, NAACL 2019](https://aclanthology.org/N19-1300/)
- [Project license allowlist implementation](../../src/tiny_omni_decision/dataset.py)
