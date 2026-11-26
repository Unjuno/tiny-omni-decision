# OpenBookQA source screen — 2026-11-26

## Disposition

**`REVIEW_NOT_ADMITTED` for model training or quality evaluation.** OpenBookQA
is a promising text Decision task: four supplied choices are selected from a
question and an elementary-science context, and its expanded release exposes
core-fact, human-score, clarity and anonymized-worker fields that could support
better grouping and label audits. However, the pinned dataset card declares
the data license `unknown`. The upstream Apache-2.0 license is for the
OpenBookQA code repository and is not evidence of a dataset license. The
published source also derives its core facts from WorldTree, so fact-source
rights need to be covered explicitly. Do not ingest split files, train, or
score a model until the dataset and fact-source rights are resolved.

## Pinned source metadata

- Official project/code repository:
  `allenai/OpenBookQA` at commit
  `b51971646e9371a61508d9953fc706645e194a71`.
- Pinned repository README: 16,536 bytes, SHA-256
  `9482b3bd2c9f507214e1e19e9813bea4b688b78691a819af37f95b074b4df2db`.
- Repository `LICENSE`: 11,357 bytes, SHA-256
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
  It identifies the code repository's Apache-2.0 terms; it is not a grant for
  the separately published question/fact data.
- Hugging Face dataset repository: `allenai/openbookqa`, revision
  `388097ea7776314e93a529163e0fea805b8a6454`.
- Pinned Hub README: 9,061 bytes, SHA-256
  `a58e68a31e20e7cbac02f80f44f3282ffbb8a40aeb20488670e5ffa975687960`.
- The Hub's machine-readable `cardData.license` is `unknown`. The dataset card
  reports 4,957 train, 500 validation and 500 test examples for each of the
  `main` and `additional` configurations.

## Task and data-quality fit

The paper describes 5,957 four-choice questions built around 1,326 elementary
science facts. Crowd workers generated questions and distractors; five other
workers answered each question, and a question was retained when at least four
answered correctly. The paper says Dev and Test received additional expert
filtering. The expanded `additional` schema associates rows with a core fact,
human accuracy score, clarity score and anonymized worker ID. Those fields
could support group-aware splits and label-quality strata, but the official
split's fact/worker overlap has not been audited. The paper also warns that its
auxiliary second-fact annotations can be incomplete or weakly related; they
must not be treated as gold evidence.

This is a narrow elementary-science text task, not generic multi-domain
reasoning or multimodal quality. The paper's estimated human accuracy is about
92%, so this source cannot substantiate the project's 95–98% aspiration without
separate evidence that the target population and task differ. If rights are
resolved, use it only as a task-specific BERT-versus-existing-text-path
comparison, group by core fact and worker before fitting, preserve the human
score/clarity fields for audit rather than performance-based filtering, and
keep an untouched group-disjoint evaluation set.

## Access boundary

This was a metadata/source screen only. The Hub viewer rendered example rows;
those previews are treated as exposed and must not be used as an evaluation
set. No Parquet shard was downloaded or processed, and no features, model,
checkpoint, training, evaluation, or GPU work was performed.

## Primary references

- [Pinned OpenBookQA dataset card](https://huggingface.co/datasets/allenai/openbookqa/tree/388097ea7776314e93a529163e0fea805b8a6454)
- [Pinned OpenBookQA code repository](https://github.com/allenai/OpenBookQA/tree/b51971646e9371a61508d9953fc706645e194a71)
- [OpenBookQA paper and annotation procedure](https://aclanthology.org/D18-1260/)
- [TensorFlow Datasets schema/split catalog](https://www.tensorflow.org/datasets/catalog/openbookqa)
