# Teacher Quality Next: Fresh Text Data Baseline

## Scope and status

This is development infrastructure for the active autonomous multimodal
Teacher-quality goal. It does not promote a model, alter Teacher v1 or prior
Video Teacher artifacts, evaluate a sealed audit, or perform quantization.
The current work is data preparation and Decision-runtime support; no model
training has run yet.

Working branch: `codex/teacher-quality-next`  
Source code baseline: `455f56e2373f9f6133388133c5f469337d4ecef9`  
Local preparation seed: `17`

## Text candidate: Amazon MASSIVE en-US

The official MASSIVE archive was fetched from Amazon's published S3 URL. Its
size is 39,500,415 bytes and SHA-256 is
`7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8`.
The pinned source repository revision is
`f966f21846043aabef9b0f974fa7970027f43738`. The upstream notice identifies
CC BY 4.0. Only en-US is used so parallel translations of the same utterance
cannot cross splits.

The importer reads official train and dev records. It inspects only the
partition marker on test rows, then skips them without parsing their labels or
utterances. No test examples, counts, labels, or metrics are stored in the
generated corpus. The fixed transformation produced:

| Split | Examples | Unique normalized utterances | Intent coverage |
|---|---:|---:|---:|
| Train | 11,502 | 11,456 | 60 |
| Validation | 2,045 | 2,032 | 60 |

Each example retains all 60 intent options. The option list is sorted from the
official train taxonomy and hashed as
`d04b663b407e9f5b5be80c9d11160c391c7b68f516c9da957aaca026138fc86d`.
Normalized utterance groups use Unicode NFKC, casefolding, and collapsed
whitespace. If a group appears in official dev and train, the entire group is
assigned to validation. Official dev lacks one train intent, so one seed-17
train-only utterance group for that intent was moved to validation. The result
has zero normalized utterance overlap between train and validation.

Output corpus hashes:

- Train: `f67bb230cffd9ab3da3af2175fe6118667dc2793d15d05a2c6affe8a61be2476`
- Validation: `79d1559478c0b86b7d7ed2c1209ba1062dca8c08e9a83ea8056ab0a4a989505f`

Exact normalized utterance-state hashes were compared with 12 existing
train/validation JSONL files from the prior safe corpora and experiment
corpora. No exact overlaps were found. Paths containing sealed, audit,
evaluation, test, or heldout markers and all heldout-candidate files were
excluded from that scan.

The original official partitions had 11 normalized utterances represented in
both train and dev, and official dev covered 59 of the 60 train intents. The
custom grouping above repairs those two integrity/coverage issues without
dropping any train/dev source record. Four train utterance groups and one
validation utterance group contain multiple upstream intent labels; these
ambiguous records are retained and reported rather than silently removed. The
official test partition remains untouched.

## Decision runtime change

The schema now permits up to 62 choices, and prompt labels use unique
single-character labels (`A-Z`, `a-z`, then `0-9`). The pinned Gemma 4
tokenizer was tested against a 60-choice prompt: all 60 answer labels resolve
to distinct single continuation tokens. Existing runs with up to 26 options
retain their former `A-Z` labels.

## Artifacts and verification

- Source manifest: `manifests/candidates/amazon-massive.yaml`
- Development validation manifest: `manifests/candidates/amazon-massive-validation.yaml`
- Importer: `scripts/import_massive.py`
- Split and provenance implementation: `src/tiny_omni_decision/massive.py`
- Local ignored data and hashes: `data/processed/teacher-quality-next/massive/metadata.json`

Verification so far: Ruff passes; 131 CPU tests pass; the source manifest
validates with project policy `ALLOW`; all 60 labels pass the pinned tokenizer
continuation check. No model training, candidate selection, audio/image/video
corpus assembly, or final audit has been performed as part of this work.

## Primary source references

- [Official MASSIVE repository](https://github.com/alexa/massive)
- [Official license notice](https://github.com/alexa/massive/blob/main/NOTICE.md)
- [Amazon Science dataset announcement](https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding)
