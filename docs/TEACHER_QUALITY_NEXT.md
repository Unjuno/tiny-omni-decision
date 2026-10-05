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

## Audio candidate: OpenSLR LibriSpeech

Only the official SLR12 `train-clean-100` and `dev-clean` archives were
downloaded. Their official MD5 checks and recorded SHA-256 hashes are in the
local metadata; the raw archives were removed after extraction. No test
archive was downloaded or read. The pinned checksum snapshot has SHA-256
`efe6522682d078fdcfd2c3b2d418a53a89683b342dd5147a769a41a7e021fe3f` and the
derived source revision is `96519bc4ce8e8a57f84fd03b8833553f51e76dfc` (a
checksum-snapshot fingerprint, not a Git commit).

The seed-17 transformation uses every unique train-clean-100 utterance once
and all dev-clean utterances for development validation. It makes a four-way
spoken-transcript selection decision with three distinct distractor
transcripts from the same split. The corpus contains 28,539 train and 2,703
validation examples, with 251/40 speakers and 28,539/2,703 unique FLAC assets.
Speaker overlap and normalized-transcript overlap between its splits are both
zero. Corpus SHA-256 values are:

- Train: `a0e0d0d12296a956e9d1aed56e92680d6570184ab4b2d190b394cf49e8eab198`
- Validation: `82d88224fc99cd85cc08568038e2f8f137e028b277f8cf310edf4e809164f4ac`

An independent local integrity pass read every referenced FLAC and compared
its SHA-256 with the corpus record: 31,242/31,242 assets exist and match,
covering 6,970,829,637 bytes. The LibriSpeech processor path was also smoke
tested against a real local FLAC with the pinned Gemma 4 processor. A safe
cross-corpus content scan found one three-character lexical collision:
`YES` occurs as a LibriSpeech target transcript and as an existing
Typed-Decisions-Synth gold answer. This is not a shared source record, speaker,
audio asset, or utterance; it is retained and disclosed rather than silently
removing a valid audio example. Candidate manifests for both splits validate
with project policy `ALLOW`.

Implementation and builder files are `src/tiny_omni_decision/librispeech.py`,
`scripts/download_librispeech.py`, and
`scripts/build_teacher_quality_librispeech.py`. FLAC decoding is supported by
the Decision processor through PyAV and resampled to the required mono 16 kHz
waveform.

## Image candidate: fresh Clevr-4 train identities

The pinned annotation SHA-256 is
`fba922a00216bbc3641f7f03a117a9f5e6a0956edde1f66062ff399225219fd6`. A fresh
seed-17 grouping pass excludes 2,433 image identities present in the prior
safe train/validation corpora. From the remaining 6,146 official-train images,
the deterministic whole-image split assigns 4,318 to training (17,272
taxonomy questions), 927 to development validation (3,708 questions), and
901 to an unmaterialized reserve. The reserve identity hash is
`0ed536ec4d1f019ade1efbbb2b1b31cfc1612b77fb05a2c55b0a84fe5325c058`. All four
taxonomy labels are represented in validation, and train/validation/reserve
image identity overlap is zero. Manifests for both candidate splits validate
with project policy `ALLOW`.

The corpus has **not** yet been materialized: the local C: volume ran out of
space during a full archive fetch. The run was stopped at 83/227 downloaded
ranges. Its archive could not be hash-verified or used; no generated Clevr-4
image candidate corpus is claimed. The range journal and partial archive are
local ignored files, not repository artifacts. The intended archive SHA-512
is pinned in the manifest as
`769465d90b6550a242f6d5940b52f2e0944c65af09e17e6f3f2cb65701af66057ab06485b2194fa9d8932e03a03e87b3fadece43049e1af1ddd9f8e4990ed19a`.
Continue after enough local disk space is available, verify the complete
archive, materialize only train/validation image identities, and leave reserve
images unmaterialized.

The split implementation is `src/tiny_omni_decision/clevr4_quality.py`; the
local materialization path is `scripts/build_teacher_quality_clevr4.py` and
accepts only a SHA-512-verified archive. The dedicated downloader resumes
verified byte ranges in `scripts/download_clevr4_archive.py`.

Verification so far: Ruff passes. The repository CPU suite passes 137 tests
when run with the repository's `src` directory on `PYTHONPATH`; the checked-in
`.venv` currently fails during Python site initialization on a CP932 decoding
error in its `.pth` processing. MASSIVE and LibriSpeech train/validation
manifests, plus Clevr-4 train/validation candidate manifests, validate with
project policy `ALLOW`. No model training, candidate selection, Clevr-4 media
corpus, video candidate corpus, fresh final audit, or final Teacher has been
produced.

## Primary source references

- [Official MASSIVE repository](https://github.com/alexa/massive)
- [Official license notice](https://github.com/alexa/massive/blob/main/NOTICE.md)
- [Amazon Science dataset announcement](https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding)
