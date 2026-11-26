# BANKING77 text Decision source-quality screen — 2026-11-26

## Disposition

**`REVIEW_NOT_ADMITTED` for a product-quality or model-selection run.** BANKING77
is a plausible English intent-decision task and may be useful for a bounded
development probe after its data-quality questions are resolved. No examples,
labels, model weights, or test records were downloaded or opened in this screen.

## Pinned source facts

- Hugging Face dataset: `PolyAI/banking77@1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`.
- Author repository: `PolyAI-LDN/task-specific-datasets@9d081458ff52e53cf7e848f414e6e9344e4e6696`.
- The [official dataset card](https://huggingface.co/datasets/PolyAI/banking77/tree/1846984189f2bc0ac51c6fc3a600fa9e1714e0c6)
  declares CC BY 4.0, 13,083 English customer-service queries, one banking
  domain, and 77 intent labels; its published split is 10,003 train / 3,080
  test, with 40 test examples per intent.
- The same card marks initial data collection, source language producers, and
  known limitations as missing information. It does not identify a source
  speaker, user, or conversation group field suitable for group-held-out
  validation.
- The [original task paper](https://arxiv.org/abs/2003.04807) presents a
  fine-grained single-domain intent task. A later ACL negative-results paper
  reports that two automated label-error detectors flagged more than 1,400 of
  the 10,003 training rows (about 14%) as *potentially* mislabeled; these are
  flags, not adjudicated corrections.

## Why it is not admitted yet

This is more relevant than the synthetic text corpora because it contains
customer-service utterances with human-assigned intent labels. However, the
source card does not document how those utterances were produced, and the
published label-noise analysis raises a substantial concern for fine-grained
classes. Treating every label as ground truth could make measured model quality
reflect annotation noise and overlapping intent definitions.

The 77-way task also exceeds the current `DecisionExample` option limit of 20.
Reducing the label set or choosing a favorable subset of distractors would alter
the task. A valid comparison would need a separately specified 77-way JSON
contract or a prospectively frozen candidate-generation protocol, plus label
quality review that does not inspect the official test labels for model
selection. The official test must remain unopened until the encoders, readouts,
and evaluation protocol are frozen.

## Decision and next gate

Do not start BERT training or score the official test from this screen. Revisit
only after either (a) an independently adjudicated, rights-cleared subset with
source-group metadata is available, or (b) the project defines a 77-option
Decision contract and a label-noise protocol that preserves all official
examples and leaves test data untouched until final evaluation. Any result must
remain a single-domain English intent task, not general text or Omni quality.

## Sources

- [Pinned official PolyAI dataset card](https://huggingface.co/datasets/PolyAI/banking77/tree/1846984189f2bc0ac51c6fc3a600fa9e1714e0c6)
- [Pinned author repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696)
- [Casanueva et al., original BANKING77 task paper](https://arxiv.org/abs/2003.04807)
- [Ying and Thomas, label-error analysis](https://aclanthology.org/2022.insights-1.19/)
