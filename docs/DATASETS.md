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

The initial text training and evaluation candidate manifests are frozen separately under
`manifests/training-candidates.yaml` and `manifests/evaluation-candidates.yaml`. The
evaluation list includes only held-out benchmark candidates; the mixed OneJev aggregate is
excluded from both generated release lists pending source review.

The Clevr-4 archive is described as the 10k version upstream; the exact pinned annotation
file has 10,531 records. Its four labels are flattened into four independent ten-option
decisions. References include the official archive SHA-512 so downstream media resolution can
verify the source archive before opening image files.

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

## Remaining gates before Phase 3

- Decide whether further component-level OneJev sources have complete permissions; do not
  ingest `REVIEW` or `DENY` rows into training.
- Clevr-4's controlled synthetic image taxonomy is eligible for training under its explicit
  CC BY 4.0 grant and attribution. Natural-image QA ingestion remains blocked for OneJev
  components until each source's commercial, training, redistribution and media rights are
  resolved; the aggregate remains REVIEW.
- Keep benchmark evaluation records completely out of train manifests; the overlap CLI is
  mandatory when producing a frozen pair.
- Review the label quality/fit of synthetic text candidates against the intended task.
- No model training, teacher-logit generation, Recovery training, ternary conversion,
  benchmark media download, or rented GPU work belongs to Phase 2.
