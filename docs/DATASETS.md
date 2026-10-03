# Phase 2 dataset provenance and decisions

This document records source facts verified from dataset repositories/cards at the pinned
revisions below. The audit command reports this project's conservative use policy; it is
not legal advice. An `ALLOW` applies only to the declared use and the explicit permissions
in that manifest. Unknown terms fail closed for training-safe normalization.

## Candidate inventory

| Source | Pinned revision | Size / modalities | Source facts | Project decision |
|---|---|---|---|---|
| [Typed Decisions Synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | `5ece89a225b23c4cd5c4bab5735a0819d61dd7d5` | 6,682 train cases / 23,319 train questions; text | Dataset card says MIT, all examples are synthetic and LLM-generated; labels are unreviewed. | **ALLOW training** under the project policy. Only hard-gold questions normalize to v0.1; the original gold and teacher values are retained. Never treat as factual benchmark data. |
| [Open-Jev](https://huggingface.co/datasets/TypeSafeAI/Open-Jev) `release-v2-redistributable` | `538ce45e3e888d3e9147d425509e06ff2d05813c` | 79,116 train rows / 113,568 all splits; text | Card says CC0-1.0 and documents exact split counts. Mostly controlled/synthetic data; target is often a probability vector. | **ALLOW text training candidate.** Adapter accepts only single-winner one-hot targets and stores the unmodified original target. Do not combine the overlapping browser/drone expansion config with this release. |
| [OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data) | `c991b20e70359acb4448147c2ce99eecccd08ac4` | 94,707 rows; text, image, video | Dataset card says 127 source datasets and per-row `license`; pinned `licenses.csv` has 127 source rows and 25 license labels. Media is present in Parquet rows. | **REVIEW; no training ingestion.** Do not use the whole aggregate. Per-row source/license is retained; a source is not training-eligible until commercial, model-training, redistribution, and media rights are explicitly resolved. See the pinned [license table](../manifests/candidates/onejev-licenses.csv), SHA-256 `27b0b90e143b5abd4116c0a81438253d9250f99ea2699c16e974c8f5d142f9a8`. |
| [MMAU test-mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) | `ccd9696c0111ea7060827598f310558df0b71b0a` | 1,000 test items; audio | Official MMAU README identifies 1,000 test-mini examples. Dataset card declares CC-BY-NC-4.0; it is a benchmark, and source audio remains third-party media. | **Evaluation-only / DENY for commercial training and media redistribution.** Metadata adapter stores a pinned media reference and does not download audio. Do not train on this benchmark. |
| [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) | `230a2d4fac8900333c61754641c7a13e069ac9c6` | 4,000 QA rows across 20 task JSON files; video QA | HF card declares MIT for the repo but states video copyrights belong to source creators and are for academic research only; 320 NTU clips require separate manual access. The pinned repo includes several large per-source video archives. | **Evaluation-only / DENY for commercial training and video redistribution.** Not all clips are available: 320 NTU RGB+D clips need separately obtained access. |
| [Oxford Clevr-4](https://www.robots.ox.ac.uk/~vgg/data/clevr4/) | code `cddc78fb2a8359dc958987b2c750bfdd4bfd2c73`; archive SHA-512 pinned in the manifests | nominal 10k image set; annotation archive contains 10,531 images (8,424 train / 2,107 val); image | Official source offers the dataset under CC BY 4.0. The pinned code defines four ten-class taxonomies: texture, shape, color and count. The archive annotation JSON SHA-256 and official checksum-list SHA-256 are recorded in each manifest. | **ALLOW controlled synthetic image classification training** with attribution. Train uses only the official train split; the separate val manifest is evaluation-only. The adapter emits four decisions per image and stores source-image references, never image bytes. |
| [Google Speech Commands v0.02](https://huggingface.co/datasets/google/speech_commands/tree/a751309c0fd613e8a5d30d77900f30e8b42bc2da) | Hub snapshot `a751309c0fd613e8a5d30d77900f30e8b42bc2da`; TFDS builder `470d259ad213ac458c5013f35e6fd01716c567a5` | Hub card: 84,848 train / 9,982 validation / 4,890 test; one-second audio; ten keyword classes | Google/TensorFlow documentation describes CC BY data collected for Speech Commands; the pinned Hub card declares CC-BY-4.0. The manifest records SHA-256 for each train/test Parquet shard. | **ALLOW keyword classification training** with attribution. Adapter pins the card's class-index mapping, keeps ten keyword labels, excludes auxiliary words and `_silence_`, and prohibits speaker identification. Test is evaluation-only. |
| [CLEVRER](https://clevrer.csail.mit.edu/) | code `98b842082ba4f7c18b6b9e3f39145871782a65ef`; official question JSON SHA-256 pinned per split | 10,000 train / 5,000 validation videos; 152,572 train / 76,368 validation questions; synthetic video | Official MIT-hosted README states CC0 and documents split ranges. Question metadata is separate from video archives. | **ALLOW descriptive single-answer synthetic video training** under CC0. Adapter includes supported categorical descriptive questions only; validation is held out. Video archives are about 12.4 GB train / 6.2 GB validation and were not downloaded. |

The HF repository `apple/mmau` is **not** the audio benchmark: its pinned files are
CodeContests and math/tool-use JSONL. MMAU's official README links the correct
`gamma-lab-umd/MMAU-test-mini` repository. Do not confuse its NC benchmark terms with the
license of the separately MIT-licensed evaluation code repository.

## Schema and provenance

`DecisionExample` is JSON-serializable JSONL with a 2–20 unique option range and a target
that must exactly match one option. `media` is always a list of `MediaRef`; each reference
stores one image/audio/video kind and a relative local path or remote/source URI, optional
SHA-256, and optional source license. Binary media is never embedded. Per-record
`provenance` and `source_target` preserve row license facts, upstream component identity,
the source's original annotation, and soft targets where available. Revisions are exact
40-character commit IDs.

The training and evaluation candidate manifests are frozen separately under
`manifests/training-candidates.yaml` and `manifests/evaluation-candidates.yaml`. The
evaluation list includes only held-out benchmark candidates; the mixed OneJev aggregate is
excluded from both generated release lists pending source review.

The Clevr-4 archive is described as the 10k version upstream; the exact pinned annotation
file has 10,531 records. Its four labels are flattened into four independent ten-option
decisions. References include the official archive SHA-512 so downstream media resolution can
verify the source archive before opening image files.

Speech Commands is a bounded keyword-classification candidate, not general speech
understanding. Its source rows require an explicit train/test split. `_silence_` and
`_unknown_` rows are intentionally excluded because v0.1 options are the ten official
keyword classes. The adapter stores a pinned source reference and does not copy audio
bytes. The pinned Hub Parquet metadata includes shard SHA-256 values in the manifests.

CLEVRER normalization reads official question JSON and creates option sets from fixed
subtype taxonomies (`exist`, `query_color`, `query_material`, `query_shape`, and `count`).
It excludes non-descriptive, answerless, unsupported, and non-categorical questions. Scene
indices are checked against the official train (0–9999) and validation (10000–14999)
ranges; media references identify the upstream video without downloading it.

## License policy

`tiny-omni-decision audit-dataset-manifest <manifest>` prints the pinned revision,
declared license and permission facts, unresolved fields, component decisions and final
`ALLOW`, `REVIEW` or `DENY` project policy. The policy allows only an explicit small
permissive-license set with affirmative commercial-use, derivative-training and
redistribution facts; media datasets also need affirmative media redistribution rights.
Explicit `false` is `DENY`; missing, unknown, custom, non-commercial or unreviewed facts are
`REVIEW` or `DENY`, never an implicit pass.

This rule does not adjudicate copyrights in upstream works. A manifest's source facts and
this repository's policy are shown separately so a reviewer can make that distinction.

## Reproduce a small run

Install `python -m pip install ".[dev]"`; pinned JSONL and JSON files stream directly from
their immutable Hub revision. Parquet-only sources need `python -m pip install ".[data]"`.
This example uses three verified source cases and writes no media bytes:

```bash
tiny-omni-decision audit-dataset-manifest manifests/dataset.example.yaml
tiny-omni-decision dataset-normalize n4ze3m/typed-decisions-synth \
  --manifest manifests/dataset.example.yaml --adapter typed-decisions-synth \
  --limit 3 --seed 17 --output data/processed/typed-synth-sample.jsonl
```

The `--limit` bounds source rows read; the adapter can emit several typed questions per
case. To normalize a local `.json` array or `.jsonl` fixture, pass its path as the source
and use `--adapter generic`. Hub loading uses each manifest's pinned revision and source
file paths; JSONL/JSON sources stream directly. Parquet-only sources use the optional
`datasets` reader with streaming. `dataset-check-splits <train.jsonl> <eval.jsonl>` fails on shared source IDs or
normalized fingerprints. Fingerprints normalize Unicode/case/whitespace and sort options,
so option-order-only and normalized-text duplicates collide; media identity is part of the
fingerprint.

For Clevr-4, download the official archive only when image media is needed, verify it against
the manifest's SHA-512, then pass its `clevr_4_annots.json` to the local adapter. A three-image
sample emits 12 decisions and does not copy image bytes into JSONL:

```bash
tiny-omni-decision audit-dataset-manifest manifests/candidates/clevr4.yaml
tiny-omni-decision dataset-normalize data/raw/clevr4-10k/clevr_4_annots.json \
  --manifest manifests/candidates/clevr4.yaml --adapter clevr4 \
  --limit 3 --seed 17 --output data/processed/clevr4-train-sample.jsonl
tiny-omni-decision dataset-normalize data/raw/clevr4-10k/clevr_4_annots.json \
  --manifest manifests/candidates/clevr4-validation.yaml --adapter clevr4 \
  --limit 3 --seed 17 --output data/processed/clevr4-val-sample.jsonl
```

Audio and video candidates can be audited and sampled as follows. Install `.[data]` to
read Speech Commands Parquet from the Hub; local CLEVRER question JSON is metadata-only:

```bash
tiny-omni-decision audit-dataset-manifest manifests/candidates/speech-commands.yaml
tiny-omni-decision dataset-normalize google/speech_commands \
  --manifest manifests/candidates/speech-commands.yaml --adapter speech-commands \
  --limit 3 --seed 17 --output data/processed/speech-commands-train.jsonl
tiny-omni-decision dataset-normalize data/raw/clevrer/train-questions.json \
  --manifest manifests/candidates/clevrer.yaml --adapter clevrer \
  --limit 3 --seed 17 --output data/processed/clevrer-train.jsonl
tiny-omni-decision dataset-normalize data/raw/clevrer/validation-questions.json \
  --manifest manifests/candidates/clevrer-validation.yaml --adapter clevrer \
  --limit 3 --seed 17 --output data/processed/clevrer-validation.jsonl
```

The pinned Speech Commands Parquet shards are read directly in streaming mode, bypassing
the legacy dataset script in the Hub repository. Audio decoding is disabled because the
adapter needs only the path and label. A live Hub smoke run normalized three train rows;
the checked-in CPU tests use tiny source-shaped fixtures. CLEVRER examples were sampled
from the official train/validation question JSON. Combined smoke sample files (including
text, Clevr-4, CLEVRER and speech fixtures) contain 61 train and 66 evaluation decisions;
the split checker reported zero shared source IDs and zero shared content fingerprints.
These bounded outputs are not frozen full-corpus training data.

## Remaining gates before durable Decision/Omni training

- Decide whether further component-level OneJev sources have complete permissions; do not
  ingest `REVIEW` or `DENY` rows into training.
- Clevr-4, Speech Commands, and CLEVRER provide controlled image/audio/video candidates
  under their recorded grants. Natural-image QA ingestion remains blocked for OneJev
  components until each source's commercial, training, redistribution and media rights are
  resolved; the aggregate remains REVIEW.
- Keep benchmark evaluation records completely out of train manifests; the overlap CLI is
  mandatory when producing a frozen pair.
- Review the label quality/fit of synthetic text candidates against the intended task.
- Phase 2 is closed at candidate/catalog level. Before durable training, generate and freeze
  full-corpus outputs, verify splits over those outputs, review label fit, and freeze
  held-out benchmarks. No model training, teacher-logit generation, Recovery training,
  ternary conversion, benchmark media download, or rented GPU work was performed in Phase 2.
