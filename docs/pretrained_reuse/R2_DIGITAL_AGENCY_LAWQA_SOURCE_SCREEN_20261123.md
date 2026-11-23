# Digital Agency Law QA source screen — 2026-11-23

## Disposition

**Do not admit this source as the next BERT-family quality comparison.** The
official repository contains 140 original multiple-choice questions, only 114
distinct raw context strings, and 15 distinct reference identifiers/URLs. It
has no publisher train/validation/test split or demonstrated independent
question-source groups. The questions are described as created and checked by
multiple LLMs, so the labels are not independent human ground truth. These
limits make it too small and weakly independent for a useful BERT-family
generalization claim or a model-selection experiment.

This is a bounded source screen, not a model result. One 381,345-byte
`selection.json` payload was read in memory to verify schema and aggregate
counts; no randomized-order file, model, media, or hidden test set was read.
No data were saved to the repository or local artifact store, and no training
or GPU operation occurred.

## Source and rights

- Official source: [`digital-go-jp/lawqa_jp`](https://github.com/digital-go-jp/lawqa_jp),
  observed `main` commit `fce599797ce9b582045a32a5710d9197e7200959`.
- `data/selection.json`: 381,345 bytes; GitHub blob SHA-1
  `90b5e2ca40a6a5c104216f2d28a708342173258c`; payload SHA-256
  `157a196ed236e745c8c133ab2eff83af8c839f4dbf1d02f21da453fa71f19d03`.
- The repository README declares Public Data License 1.0 (PDL 1.0). The
  Digital Agency describes PDL as compatible with CC BY 4.0 and permits
  commercial reuse with source attribution. PDL also leaves users responsible
  for third-party rights; the question contexts cite both e-Gov laws and some
  external documents, so this screen does not establish rights for every
  embedded excerpt. See the [official release notice](https://www.digital.go.jp/en/news/9d06cea7-4b62-4d27-b036-d39cf65dad73),
  [dataset README](https://github.com/digital-go-jp/lawqa_jp), and
  [PDL 1.0 terms](https://www.digital.go.jp/resources/open_data/public_data_license_v1.0).

## Task and independence limits

- The README describes four-choice legal QA built and checked using multiple
  LLMs. It is a narrow Japanese legal-knowledge task, not general decision
  behavior.
- `selection.json` has 140 rows / file identifiers, 114 distinct context
  strings, and 15 distinct reference identifiers/URLs. There is no official
  train/validation/test split in the repository schema. The randomized files
  are documented as option-order variants, so those variants must remain
  grouped with their original question if ever used.
- The cited law sources do not imply that question labels were independently
  adjudicated. The QA has not been checked against current amended law here.

## BERT-family decision

This source does not justify another BERT run. Preserve the existing evidence:
BERT-base is a useful Japanese text teacher/reference on the tested
teacher-labeled task, but its roughly 110.6M parameters consume more than half
of the whole-model budget before the other modality towers. The compact
27.6M comparison was weaker there, while the grouped JCoLA OOD comparison
showed a majority-class collapse for both BERT-base and the tested DistilBERT
variant. Those results do not establish a family-wide ranking.

Only consider a single compact BERT-derived student/distillation experiment
after admitting a new, sufficiently large, source-disjoint task with an
independently justified target and a validation split frozen before fitting.
Do not reuse the already exposed Japanese typed-decision, Social IQa, or
JCoLA development/test labels for tuning. This screen neither changes nor
closes the multimodal Path A and joint-input requirements.
