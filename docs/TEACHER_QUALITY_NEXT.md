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

### Read-only Teacher v1 reference on LibriSpeech validation

The frozen Teacher v1 adapter was evaluated once on all 2,703 LibriSpeech
development examples. The adapter SHA-256 is
`4e664a5500a3a109a48de7ec7c807d2b09b61cf386fd124ca8368e197802f951`; the base
revision is `6befbaca7398925921802abd1f277b495b78b738`. The validation corpus
SHA-256 is the value above, and the ordered validation ID hash is
`2adf72fcbf3f8eef3ab46ad39fb4fd45c326d26f56f0deac1e1ff8831506ebf6`.

| Modality/source | N | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Audio / LibriSpeech | 2,703 | 0.99334 | 0.02910 | 0.00959 | 0.01527 | 0.97979 |

The v1 training config has a 1,024-token input cap, but the complete validation
set's measured maximum processor length is 1,072; three examples exceed 1,024.
To retain every validation item, the evaluation used 1,072 tokens. This is a
read-only baseline on a new development source and does not replace or update
the historical v1 metrics. It is a clean audiobook transcript-selection task
from one source, so this high score is not evidence of broad audio reasoning
or of the four-modality goal being achieved. Full per-example predictions and
metrics are saved locally under the ignored path
`artifacts/teacher-quality-next/v1-librispeech-validation/`; no model weights
were changed. Evaluation wall time was 762.2 seconds on the local GPU; peak
VRAM was not instrumented for this run.

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

The full 3,797,490,816-byte archive has now been fetched and its SHA-512
verified against the manifest:
`769465d90b6550a242f6d5940b52f2e0944c65af09e17e6f3f2cb65701af66057ab06485b2194fa9d8932e03a03e87b3fadece43049e1af1ddd9f8e4990ed19a`.
Only candidate train/validation media were materialized; all 901 reserve
identities remain unmaterialized. The resulting corpus hashes are:

- Train: `2db15927f9b5d163f3be849f4442a2275e7653f29b053f7f9b1cf1b1d560a39a`
- Validation: `73c5fa8de932877750614d73542630937f90a297e419538ffe813374778d098d`

Independent verification covered every candidate media file, hashing and PNG
decoding each image: train 17,272 examples / 4,318 unique images / 0 missing /
0 hash mismatches / 0 invalid PNGs; validation 3,708 / 927 / 0 / 0 / 0. The
train and validation image IDs, media hashes, and task groups are disjoint;
the 901 reserve IDs also have zero overlap with either corpus. Their identity
hash matches the split metadata above. An interrupted earlier materialization
had left a zero-byte `CLEVR_new_000117.png`, which the independent decode pass
found. The materializer now validates PNG structure and writes extracted files
atomically; the builder repaired the image from the verified archive and
regenerated the corpus hash. A regression test covers repair of an invalid
existing image and verifies that reserve media are not extracted.

The split implementation is `src/tiny_omni_decision/clevr4_quality.py`; the
local materialization path is `scripts/build_teacher_quality_clevr4.py` and
accepts only a SHA-512-verified archive. The dedicated downloader resumes
verified byte ranges in `scripts/download_clevr4_archive.py`.

Verification so far: Ruff passes. The repository CPU suite passes 137 tests
when run with the repository's `src` directory on `PYTHONPATH`; the checked-in
`.venv` currently fails during Python site initialization on a CP932 decoding
error in its `.pth` processing. MASSIVE and LibriSpeech train/validation
manifests, plus Clevr-4 train/validation candidate manifests, validate with
project policy `ALLOW`. No new model training, candidate selection, fresh
final audit, or final Teacher has been produced.

### Teacher v1 on fresh Clevr-4 development validation

The read-only Teacher v1 adapter was evaluated once on the complete 3,708-row
Clevr-4 development validation (927 unique images). This set is disjoint from
the previously used image identities and the reserve set documented above; it
is development evidence, not the final audit. The run used base and processor
revision `6befbaca7398925921802abd1f277b495b78b738`, adapter SHA-256
`4e664a5500a3a109a48de7ec7c807d2b09b61cf386fd124ca8368e197802f951`, and
config SHA-256
`a68fae06267078dbbbcc8afc0ce4680cb553ed28b968e64e2bde50955c5a6409`. The
validation JSONL SHA-256 is
`73c5fa8de932877750614d73542630937f90a297e419538ffe813374778d098d`; its
sample-ID order hash is
`5dbeef2b479d65a20e23ba1100211faeb026657f9c1cc54a7f22655add026d1e`.

| Modality | N | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Image (`sgvaze/clevr4`) | 3,708 | 0.7201 | 0.7937 | 0.3822 | 0.0880 | 0.8073 |

Inference took 2,393.11 seconds with peak VRAM 11,031,089,664 bytes. All
3,708 predictions were saved, and the prediction count, corpus row count,
validation file hash, and adapter hash were verified against the metrics JSON.
This one-source result is 18 percentage points below the per-modality 90%
goal. It does not establish a general Image ceiling; further development must
diagnose source/task coverage and improve using a separate candidate without
reusing any final audit. Raw predictions and machine-readable metrics are in
the ignored local path
`artifacts/teacher-quality-next/evaluations/v1-clevr4-validation/`.

### Fresh CLEVRER video development corpus

The deterministic seed-17 split contains 500 train scenes and 100 validation
scenes from unused official CLEVRER train-source videos. It excludes the 446
historical scenes found in Teacher v1 and Candidate E train/validation
corpora. All 600 selected MP4s were independently checked for existence and
their SHA-256 values match the corpus manifest. An initial materialization
attempt stopped with `ENOSPC`; after disk space recovered, the exact same
selection was resumed, existing videos were verified and reused, and the
builder completed without changing the split or sampling policy.

| Split | Scenes / unique videos | Examples | Temporal descriptive | Explanatory | Predictive | Counterfactual |
|---|---:|---:|---:|---:|---:|---:|
| Train | 500 | 12,372 | 5,347 (43.2%) | 3,083 (24.9%) | 704 (5.7%) | 3,238 (26.2%) |
| Validation | 100 | 2,538 | 1,067 (42.0%) | 638 (25.1%) | 132 (5.2%) | 701 (27.6%) |

The official question source fingerprint is
`11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4` and the
video source revision is recorded in the manifest. Corpus SHA-256 values are
`6196cc1f508210016430b97e3ddab5e72ef677d84f276bb11e4c3f5d31b38d26` (train)
and `16b3b9a45c7c430cab81c74b7bf633ba0c4aea3ac120587820e33ae879a3dd4b`
(validation); the manifest SHA-256 is
`dfd6e3246c7f252982a57341913f872c2e23be5e4e6dc527731b5e7596bd6e1f`.
Independent validation confirmed exact manifest and corpus hashes, 12,372 /
2,538 rows, 500 / 100 disjoint scene IDs, 600 present nonempty media files,
and zero shared source IDs, source assets, media identities, or normalized
content fingerprints. The sealed-audit flag is false; no sealed audit was
loaded. The machine-readable corpus and manifest are in the ignored local
path `artifacts/teacher-quality-next/clevrer-fresh-scenes-v1/`.

### Teacher v1 on fresh CLEVRER development validation

The read-only Teacher v1 adapter was evaluated on all 2,538 validation
examples from the fresh CLEVRER corpus above (100 unique validation videos).
The evaluation used the existing 4-frame config and made no model, data, or
preprocessing changes. The corpus split verifier had already confirmed
zero train/validation source, asset, media-identity, and normalized-content
overlap; the audit corpus was not loaded. This is development evidence and
must not be presented as a blind audit.

| Task type | N | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Temporal descriptive | 1,067 | 0.4977 | 1.1374 | 0.6413 | 0.1672 |
| Explanatory | 638 | 0.5376 | 0.7498 | 0.5459 | 0.1472 |
| Predictive | 132 | 0.5379 | 0.8303 | 0.6100 | 0.2227 |
| Counterfactual | 701 | 0.4650 | 0.7763 | 0.5734 | 0.1570 |
| **Video / macro** | **2,538** | **0.5008** | **0.9243** | **0.5970** | **0.1584** |

The validation and adapter hashes match the metrics record. Validation JSONL
SHA-256:
`16b3b9a45c7c430cab81c74b7bf633ba0c4aea3ac120587820e33ae879a3dd4b`;
sample-ID order SHA-256:
`b34e827212f089b44b72edf4fb06f7fcf9d447bd943419d4e56c66edd3677e18`;
adapter SHA-256:
`4e664a5500a3a109a48de7ec7c807d2b09b61cf386fd124ca8368e197802f951`;
config SHA-256:
`a68fae06267078dbbbcc8afc0ce4680cb553ed28b968e64e2bde50955c5a6409`.
Base and processor revision are both
`6befbaca7398925921802abd1f277b495b78b738`. Configured and effective maximum
sequence length were both 1,024 tokens. All 2,538 prediction IDs match the
validation IDs in exact order, and the prediction file contains 2,538 unique
rows. Metrics JSON SHA-256 is
`49bbf1935ecea2dfca8464feac4b4907d3765cf9ee0e2c491f67cc1e76b7e78e`;
predictions JSONL SHA-256 is
`0bcb9d9ba9c29b9e544c284dfa0cfe64ccc31776f3579d009d6f208dc4f64ac3`.

Inference took 1,259.44 seconds; total wall time from process start through
artifact write was about 23 minutes 24 seconds. Peak allocated VRAM was
10,797,883,392 bytes. No cloud compute was used. The local TorchVision
fallback was used because `torchcodec` is unavailable. The first uncached
attempt completed its redundant processor preflight but was stopped before
inference output; it produced no score and is not counted as an evaluation.
Commit `2c3ff39` adds a bounded per-video sampled-frame cache to the
development evaluator. Unit tests and a real-processor smoke confirmed cached
and uncached tensors are identical, and that repeated rows reuse the decoded
frames under the fixed evaluation config.

The 50.08% score is far below the 90% per-modality target and identifies
video as a major current weakness. It does not establish a model-family
ceiling: task type, representation, coverage, and model adaptation still need
separate development experiments. Metrics and predictions are in the ignored
local path
`artifacts/teacher-quality-next/evaluations/v1-clevrer-video-validation-cached/`.

### Candidate E on the same fresh CLEVRER development validation

The immutable Candidate E best checkpoint (step 512; 8 frames;
decoder-all-linear rank 16) was evaluated on the exact same 2,538 validation
IDs, in the same order and with matching targets/options. This is a matched
development comparison: Teacher v1 used 4 frames, while Candidate E used 8;
the adapters and training procedures also differ, so the comparison does not
isolate a causal frame-count effect. Both prediction sets have been observed
on this development generation. It is not a final audit.

| Candidate / task | N | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Teacher v1, all video | 2,538 | 0.5008 | 0.9243 | 0.5970 | 0.1584 |
| Candidate E, all video | 2,538 | 0.4704 | 0.9326 | 0.5736 | 0.0572 |
| Candidate E, temporal descriptive | 1,067 | 0.4236 | 1.2526 | 0.6650 | 0.0514 |
| Candidate E, explanatory | 638 | 0.4828 | 0.7059 | 0.5126 | 0.0832 |
| Candidate E, predictive | 132 | 0.5000 | 0.7039 | 0.5106 | 0.0790 |
| Candidate E, counterfactual | 701 | 0.5250 | 0.6950 | 0.5018 | 0.0406 |

Candidate E accuracy is 3.03 percentage points below v1. A paired bootstrap
resampling 100 video scenes (10,000 replicates, seed 17, percentile 95% CI)
put the E-minus-v1 accuracy difference at `[-0.0524, -0.0087]`. Brier score
improved by 0.0234 (95% CI `[-0.0402, -0.0070]`); NLL increased by 0.0084
(95% CI `[-0.0180, 0.0348]`). Task-level differences are mixed: Candidate E
improved counterfactual accuracy but declined on temporal descriptive,
explanatory, and predictive questions. Temporal descriptive NLL also rose by
0.1152. The overall accuracy target remains unmet by a wide margin.

The validation SHA-256 and ordered-ID SHA-256 are the same values recorded
for the v1 comparison above. Candidate E adapter SHA-256 is
`c0d483b89798bea0d20f50a2335a0fa658c1794cd31c4ceb245f079b0e8ae851` and its
config SHA-256 is
`0635c29cc761412cebec53f1569fecfc296c647a2f7a11bdbb217f88cfa7ac5c`. The
base and processor revision is
`6befbaca7398925921802abd1f277b495b78b738`; effective sequence length was
1,024 tokens. Inference took 2,178.69 seconds and process wall time through
artifact write was 38 minutes 49 seconds. Peak allocated VRAM was
11,426,737,152 bytes. It ran locally without cloud compute. The TorchVision
fallback was used because `torchcodec` is unavailable.

Prediction count, unique IDs, exact validation order, targets, options,
validation hash, adapter hash, and config hash all passed. Metrics JSON SHA-256
is `5e05c4b693d0fbe2a3afe9540469066507bc652eba7769cec1638dc77c54b094`;
predictions JSONL SHA-256 is
`229e256cbc707f44d57c6ed19864b54cc1e5953eb5df0e3dae56ec4f009fdfa8`;
paired-comparison JSON SHA-256 is
`6964fe1159f6eb0de97633fd63d5d3ed9aa3d8408cd51139e515310750a7fe75`.
Artifacts are in the ignored local path
`artifacts/teacher-quality-next/evaluations/candidate-e-clevrer-video-validation/`.
These results do not show that 8 frames caused a regression or that the model
family has reached a ceiling. Any further broad candidate selection should
use another clean development generation to limit repeated selection on these
observed 100 scenes.

## Primary source references

- [Official MASSIVE repository](https://github.com/alexa/massive)
- [Official license notice](https://github.com/alexa/massive/blob/main/NOTICE.md)
- [Amazon Science dataset announcement](https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding)
