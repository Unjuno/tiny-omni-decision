# Social IQa fresh text Decision source screen — 2026-11-23

## Disposition

`ALLOW_SCOPED_EXPLORATORY_TEXT_DECISION` for a training-only, group-split
English three-option probe. This source is a candidate for comparing compact
BERT-family text encoders on supplied-option scoring; it does not qualify any
model for product use or establish multimodal quality.

## Official source and rights

- Dataset: [AI2 Social IQa](https://allenai.org/data/socialiqa), pinned dataset
  repository revision `8835ceb9141d7896d9d968634a9b21ae440e3ec5`.
- The pinned dataset card body declares `CC-BY-4.0`. The Hub API's structured
  `license` field is absent, so the card-body statement is retained verbatim as
  the license evidence rather than inferred from the repository loader code.
- The paper says contexts and question types were seeded from ATOMIC. The
  official [ATOMIC dataset card](https://huggingface.co/datasets/allenai/atomic)
  declares CC BY 4.0 as well.
- The pinned upstream loader points to
  `https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip`.
  Its archive SHA-256 is
  `ee073914a0fc33265cfbcfc50ec20df9b2e07809c3f420f599138d1f394ef5c3`.

The source provides 33,410 human-annotated, three-choice training records.
The paper describes everyday social reasoning (motives, reactions and likely
actions), making it relevant to a text Decision interface. This is a scoped
English commonsense task, not general reasoning coverage.

## Train-only split preflight

Only the archive's `train.jsonl` and `train-labels.lst` members were read. The
archive's dev members were not extracted or opened; no publisher test or project
sealed audit was accessed. To avoid relying on the publisher's small dev set,
we created a deterministic seed-17 80/20 split over NFKC-normalized exact
`context` groups:

| Split | Decisions | Context groups | Target 0 | Target 1 | Target 2 |
|---|---:|---:|---:|---:|---:|
| Train | 26,751 | 26,400 | 9,039 | 8,885 | 8,827 |
| Validation | 6,659 | 6,600 | 2,235 | 2,291 | 2,133 |

There are 33,000 unique normalized contexts; 410 contexts have two questions,
and none has more than two. Train/validation context overlap, full normalized
example overlap, and normalized question-plus-options overlap are all zero.

### Remaining source-identity limitation

Social IQa rows have no stable ATOMIC seed-event/source ID. The split therefore
groups exact normalized contexts, but **cannot rule out paraphrased contexts
originating from the same ATOMIC event across splits**. Treat these metrics as
development evidence with that limitation; do not call the split fully
source-event-disjoint. A future probe should try to reconstruct conservative
source-event groups from the source annotations or use another fresh source
before making a broad text-quality claim.

## Frozen artifacts

All downloaded data and generated JSONL remain outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\socialiqa-train-only-screen-20261123-attempt-01\`.
The complete machine-readable receipt is `source-screen-report.json` (SHA-256
`fde44c1685c1b4a67f382922138d22fbc68858648f0f4daf82ba08f658afc90c`). It
contains hashes and sizes for the source archive, pinned card and loader, split
JSONL files, and the generation script.

No model was loaded, trained, selected, or evaluated during this source screen.
The next permitted step is a small, predeclared text-only comparison on this
fixed train/validation split, with no use of the untouched official dev/test or
project sealed audit. Selection should be limited because the train split does
not expose ATOMIC source-event identity.
