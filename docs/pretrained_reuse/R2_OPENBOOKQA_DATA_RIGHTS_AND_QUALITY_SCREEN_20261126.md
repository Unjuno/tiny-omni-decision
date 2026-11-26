# OpenBookQA source and data-quality screen — 2026-11-26

## Disposition

**REVIEW_NOT_ADMITTED. No dataset payload was downloaded or opened.** The
task is potentially relevant as a difficult, four-choice scientific Decision
task, but the public evidence inspected does not establish permission to use
the question/fact corpus in this product-directed research. Do not train or
evaluate on it unless the data rights are clarified and the source-disjoint
protocol is frozen first.

## Pinned source evidence

- Official code repository: `allenai/OpenBookQA`, `main` observed at
  `b51971646e9371a61508d9953fc706645e194a71` on 2026-11-26.
- The repository identifies itself as baseline **code** for the EMNLP 2018
  OpenBookQA paper and displays an Apache-2.0 license. The README directs users
  to download the dataset separately; the data is not included in that code
  repository.
- Official issue [#7, “Data license”](https://github.com/allenai/OpenBookQA/issues/7)
  remains open. The issue asks the publisher to add a data license distinct
  from the existing repository license; no resolution was visible in the
  inspected issue. Therefore the code's Apache-2.0 license is not evidence of
  a license grant for the dataset itself.
- The EMNLP paper describes 5,957 four-choice questions and 1,326 elementary
  science facts, with questions requiring both a supplied fact and broader
  common knowledge. It reports human performance close to 92% and notes a
  substantial gap between humans and pretrained methods. This makes it a
  useful possible stress task, but not evidence that a compact model can meet
  95–98% on a broader text capability target.

## Data-quality and split requirements if rights are clarified

The publisher paper describes questions built from a relatively small fact
book. A random question split could therefore place questions derived from
the same fact or concept on both sides and overstate generalization. Before
admission, obtain the official data and metadata under a verified license,
then audit:

1. question, answer-choice, and normalized-content duplicates;
2. originating core-fact/concept identity and train/dev/test crossings;
3. ambiguity, human-accuracy, and clarity metadata where supplied;
4. option position, answer length, lexical overlap, and other shortcut
   predictability;
5. whether a fact-grouped train/validation split retains enough independent
   groups for a meaningful estimate.

Do not inspect publisher test labels during source admission or model
selection. The existing CommonsenseQA development split is already exposed;
it must not be used to select an OpenBookQA candidate.

## Sources

- [Official OpenBookQA repository](https://github.com/allenai/OpenBookQA)
- [OpenBookQA data-license issue #7](https://github.com/allenai/OpenBookQA/issues/7)
- [EMNLP 2018 paper](https://aclanthology.org/D18-1260/)

This is a metadata/source-screen only. No questions, labels, facts, publisher
validation/test files, checkpoints, or sealed-audit data were accessed.
