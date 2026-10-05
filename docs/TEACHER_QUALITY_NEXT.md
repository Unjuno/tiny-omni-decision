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

## Teacher v1 on MASSIVE English development validation

The read-only Teacher v1 adapter was evaluated on all 2,045 examples in the
ALLOW-approved Amazon MASSIVE en-US development split. Each example selects
from the full set of 60 intents; no options or examples were removed. The
official test partition was not read. This is development evidence, not a
final audit.

| Modality / source | N | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Text / `alexa/massive` | 2,045 | 0.6509 | 1.8286 | 0.4976 | 0.1141 | 0.7648 |

The evaluation used the immutable Teacher v1 selected adapter
(`4e664a5500a3a109a48de7ec7c807d2b09b61cf386fd124ca8368e197802f951`), base
and processor revision `6befbaca7398925921802abd1f277b495b78b738`, and the
explicit reference config (`dc44f7dfa9bf1f5cf171ea707f3c8a5e0049ca2cc2064216f1c9c288650ae6e7`).
The validation JSONL SHA-256 is
`79d1559478c0b86b7d7ed2c1209ba1062dca8c08e9a83ea8056ab0a4a989505f`; ordered
sample-ID SHA-256 is
`6e409a3455522318b31cbd263b3d573a8185cfc89528522e65cd6b207aadb8e8`.
Effective sequence length remained 1,024. Evaluation took 547.765 seconds
and peak allocated VRAM was 10,738,208,768 bytes on the local RTX 3080 Laptop.
Metrics JSON SHA-256 is
`3d7427f1b822f5938fb478546f61e101b210da5b33fe28ad231f3823771f6dec`; all
2,045 per-example predictions are in the ignored local evaluation directory,
with SHA-256
`809c16c3f23987898a7a6984693a9bd35b009100fd392a949633eaf668af39e1`.

At 65.09%, v1 is 24.91 percentage points below the Text goal on this
single-source 60-intent task. This identifies a substantial domain/task gap
for the current teacher; it does not establish a general Text ceiling or
predict the result on other text Decision sources. Future experiments may use
this development split for selection, with validation-generation reuse
tracked explicitly; no sealed audit was loaded.

### Candidate E matched Text diagnostic

The immutable Candidate E best checkpoint (step 512; adapter SHA-256
`c0d483b89798bea0d20f50a2335a0fa658c1794cd31c4ceb245f079b0e8ae851`) was
evaluated on the identical ordered MASSIVE validation IDs with the same
preprocessing. It scored 0.6274 Accuracy, 1.8936 NLL, 0.5051 Brier, and
0.0924 ECE (mean confidence 0.7197), compared with v1's 0.6509 Accuracy,
1.8286 NLL, and 0.4976 Brier. Candidate E's ECE was lower than v1's by 0.0217,
but its accuracy and both proper scoring rules were worse. Its config SHA-256
is `0635c29cc761412cebec53f1569fecfc296c647a2f7a11bdbb217f88cfa7ac5c`; the
base/processor revision, validation JSONL hash, and ordered sample-ID hash
match the v1 evaluation above. Evaluation took 655.5 seconds; peak allocated
VRAM was 10,824,126,464 bytes. Metrics JSON SHA-256 is
`7e0a55e9f1bf779d8ae5c0e58be79a5dc2b3a418df06ad46c01c656a1359e9e6`; the
2,045-row predictions JSONL SHA-256 is
`9e28a96a672b2a68e33d46dcf460b38605ca683c6db549518f8c9563cb48e31`.

A paired cluster bootstrap over the 2,032 normalized-utterance groups (10,000
resamples, seed 17, percentile interval) estimated Candidate E minus v1
Accuracy at -0.0235, with 95% CI [-0.0362, -0.0113]. This compares two already
trained adapters and does not isolate any single training change. Across the
60 target intents, unweighted class Accuracy was 0.7019 for v1 and 0.6458 for
Candidate E. Per-intent validation counts range from 1 to 131, so that macro
class summary is noisy; it is supplementary and does not replace the
full-set sample-weighted result. Both candidates scored 0/8 on
`music_settings`, 0/16 on `email_querycontact`, 5/106 on `general_quirky` for
v1 (2/106 for E), and 1/26 on `recommendation_events`. These errors and the
long-tailed validation counts motivate source/task analysis and broader clean
text data, not deleting difficult intents. This validation generation is now
observed for v1 and Candidate E; avoid repeated candidate selection on it.

The split includes all 60 intents, but is long-tailed: per-intent counts range
from 4 to 810 in training and from 1 to 131 in validation. This reflects the
available source distribution and is retained without reweighting the reported
metrics. For v1, frequent confusions include `calendar_set` to
`calendar_query` (34), `play_music` to `music_query` (30), and `calendar_set`
to `alarm_set` (26). Candidate E has the same first confusion (36) and makes
more `cooking_recipe` to `cooking_query` errors (33 versus v1's 15). The
evidence points to fine-grained intent discrimination and long-tail coverage;
it does not yet distinguish data scarcity from prompt/label representation or
adapter interference.

## Fresh CLEVRER development generation v2

A second seed-23 development generation was built from the pinned official
CLEVRER training questions, excluding all 1,046 scenes in the durable v1,
Candidate E, and fresh-scenes-v1 corpora. The deterministic split has 500
training scenes (12,509 examples) and 100 validation scenes (2,491 examples).
The train and validation corpora have zero shared scene IDs. All 600 selected
MP4 files were present and SHA-256 verified. A cross-corpus check against
106,182 historical records found zero overlapping source IDs, assets, media
identities, or normalized content. The sealed audit was not loaded.

| Split | N | Temporal descriptive | Explanatory | Predictive | Counterfactual |
|---|---:|---:|---:|---:|---:|
| Train (500 scenes) | 12,509 | 5,349 | 3,047 | 734 | 3,379 |
| Validation (100 scenes) | 2,491 | 1,070 | 628 | 144 | 649 |

Corpus hashes are `78fde1efac834ff371dde233da60fec1d04aa6f7dee75d31fefdc36681e25549`
(train) and `0dde757fee675c72489016170d2018fcf08ee309da76ec01eb7885a2375532f5`
(validation). The corrected manifest SHA-256 is
`8630c3d723018e02536d785cc682395fc1ace986e341491b4ae0ddafd7114c37`. The
builder initially wrote a v1 dataset ID into the v2 manifest; this was caught
before training, corrected using the output generation name, and recorded in
the local `manifest-correction.json`. Corpus bytes and their hashes did not
change. A regression test now covers generation-specific IDs.

## Multimodal data-coverage candidate corpus and run

The new local candidate corpus combines the durable v1 train rows for
non-video modalities, MASSIVE train, LibriSpeech train-clean-100, fresh
Clevr-4 train, and the fresh CLEVRER v1/v2 train scenes. It has 117,828 train
examples and 10,947 validation examples. Validation combines MASSIVE,
LibriSpeech, Clevr-4, and fresh CLEVRER v2. The full train/validation gate
reports zero source-record, asset, media, and normalized-content overlap.
The exact source counts and hashes are in the local corpus manifest at
`data/processed/multimodal-data-coverage-v1/`:

| Modality / source | Train examples | Validation examples |
|---|---:|---:|
| Text / `alexa/massive` | 11,502 | 2,045 |
| Text / `TypeSafeAI/Open-Jev` | 6,527 | — |
| Text / `n4ze3m/typed-decisions-synth` | 23,319 | — |
| Image / `sgvaze/clevr4` | 18,544 | 3,708 |
| Audio / `google/speech_commands` | 4,516 | — |
| Audio / `openslr/LibriSpeech` | 28,539 | 2,703 |
| Video / `MIT-IBM/CLEVRER` | 24,881 | 2,491 |

The combined train SHA-256 is
`7fe73a0039fd4b651b323f1cee0de12f6a429ff69605e485dce2ef538a6762e3`; the
validation SHA-256 is
`1a8691b7b09b5e37d17a46340519bbd67bca6480d1d887714b4bd1b8fa330db0`.
The legacy v1 train corpus contributes no video rows because it includes
CLEVRER records without the task labels required by the video task-weighted
sampler. Its 3,492 video rows were excluded as a group; the candidate uses the
two scene-disjoint, task-labeled fresh CLEVRER training generations instead.

An actual sampler dry-run consumed 8,192 unique examples with zero repeats.
Two explicit source policies kept modality weights and video task weights
fixed. Equal source weights consumed 683 MASSIVE examples; weighting
`alexa/massive` at 2.0 consumed 1,024 and proportionally reduced Open-Jev and
Typed Decisions Synth to 512 each. The selected 2x policy is a text-coverage
hypothesis, not a validation-proven improvement. Under it the dry-run
consumed 1,365 audio, 2,048 image, 2,048 text, and 2,731 video rows; video task
counts were 546 temporal descriptive, 819 explanatory, 820 predictive, and
546 counterfactual, with 975 unique video scenes and zero repeated examples.
Sample-ID order SHA-256 is
`4313ae5c498e6dd60b4c5ad0d066d83174f2d51e296b961da2bd1b39b0baea86`.

Candidate config is `configs/decision/teacher_quality_data_coverage_v1.yaml`
(seed 17; 8 frames; decoder-all-linear rank 16; cosine schedule with 3% warmup;
2,048 steps; early-stopping patience 17; one sample per example; frozen
modality encoders and projector). Training constructs a fresh LoRA adapter on
the pinned pretrained base. The read-only Teacher v1 selected adapter is used
only for a same-validation reference evaluation, then unloaded; it is not a
training initialization. No resume checkpoint is supplied.

The first local launch stopped before baseline evaluation or optimizer updates:
its nested corpus path made the legacy media-root resolver choose `data/processed`,
so a Clevr-4 path resolved incorrectly. That failed attempt is retained under
`artifacts/teacher-quality-next/teacher-data-coverage-v1/seed17-2048/` and is
not a model result. The unchanged corpus files were copied to the standard
`data/processed/multimodal-data-coverage-v1/` depth; their SHA-256 hashes match.
A full preflight over 128,775 train/validation rows then found zero missing
media files and zero paths escaping `data/`. Train/validation integrity checks
also passed. A retry uses a distinct output directory; its result is pending.

The retry's fresh-base pre-training selection baseline completed on 512
validation examples (128 per modality). Accuracy was Audio 0.9531, Image
0.1875, Text 0.0547, and Video 0.4844; macro Accuracy was 0.4199. Macro NLL,
Brier, and ECE were 3.0106, 0.7664, and 0.2845. These are development
validation baseline metrics, not final-audit results. The 2,048-step fresh
LoRA training run is now in progress; its history and checkpoints are kept in
the separate `seed17-2048-media-root-fix/` output directory.

The first scheduled evaluation completed at step 128. The validation selector
chose this checkpoint (selection score 0.749918; no overfit warnings) and
saved it at `checkpoints/step-000128/` plus `best/`. Across 512 validation
examples (128 per modality), metrics were:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 0.9844 | 0.0413 | 0.0136 | 0.0297 |
| Image | 0.6406 | 1.0133 | 0.4660 | 0.1520 |
| Text | 0.6484 | 1.9495 | 0.5134 | 0.1654 |
| Video | 0.5234 | 0.8683 | 0.5485 | 0.0681 |
| Macro | 0.6992 | 0.9681 | 0.3854 | 0.1038 |

Relative to the fresh-base selection baseline on the same IDs, all four
modalities improved at step 128; Video remains the weakest and Text has the
worst NLL. Video question-type results were temporal descriptive 10/26
(0.3846 accuracy, 1.5576 NLL), explanatory 21/38 (0.5526, 0.6879), predictive
20/38 (0.5263, 0.7059), and counterfactual 16/26 (0.6154, 0.6801). At this
checkpoint the run had consumed 512 unique examples with no repeats, 496
unique underlying assets, 158 video scenes, and 171 CLEVRER parent questions.
Source counts were CLEVRER 171, Clevr-4 128, MASSIVE 64, Speech Commands 42,
LibriSpeech 43, Open-Jev 32, and Typed Decisions Synth 32. The sample-ID order
hash for these 512 examples is
`19a34fe6a6432c4b8534d2e6fcd02fb2522f21ea21c7b2816b6946c908761382`.
The base and step-128 validation prediction files contain the exact same 512
unique sample IDs in the same order; their validation ID-order SHA-256 is
`00754c90a2155aa8a8a949489a3d7a80e6205943f21153047f434be61fda92a0`.
The measured validation pass took 390 seconds; elapsed run time at that point
was 2,037 seconds and peak allocated VRAM was 11,916,105,728 bytes. Rolling
train-window metrics are recorded in `validation-step-128.json`; they are not
a fixed-example generalization-gap estimate. Further training and scheduled
validation remain in progress; these development-validation gains do not
promote the candidate to product or establish final-audit performance.

At step 256, the selector chose a new best checkpoint with score 0.704172
(adapter SHA-256 `bebc6c880f8c9cc188db49f6635e64e45ac851e36d5d4b0d81011fa3c75ecb6a`).
The same fixed 512 validation IDs were used, in the same order. Metrics were:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.0011 | 0.0001 | 0.0011 |
| Image | 0.7266 | 0.9915 | 0.4269 | 0.1662 |
| Text | 0.6563 | 1.8058 | 0.5071 | 0.2215 |
| Video | 0.4922 | 0.8826 | 0.5580 | 0.0931 |
| Macro | 0.7188 | 0.9203 | 0.3730 | 0.1205 |

Compared with step 128, Image improved by 8.6 points and Audio reached 100%;
Text gained 0.8 points, while Video fell by 3.1 points. Macro Accuracy,
NLL, and Brier improved, while macro ECE worsened. The selector also recorded
`weak_modality_degraded` and `train_loss_falling`. Video task-type results were
temporal descriptive 6/26 (0.2308 accuracy, 1.6667 NLL), explanatory 21/38
(0.5526, 0.6850), predictive 20/38 (0.5263, 0.6899), and counterfactual 16/26
(0.6154, 0.6689). The candidate remains in training under the predeclared
2,048-step budget; one weak-modality decline does not trigger a stop.

By step 256 the run had consumed 1,024 unique examples with no repeats, 972
unique assets, 300 video scenes, and 341 CLEVRER parent questions. Unique
assets by modality were Audio 167, Image 249, Text 256, and Video 300. Source
example counts were CLEVRER 341, Clevr-4 256, MASSIVE 128, Speech Commands 85,
LibriSpeech 86, Open-Jev 64, and Typed Decisions Synth 64. The validation ID
order SHA-256 remains
`00754c90a2155aa8a8a949489a3d7a80e6205943f21153047f434be61fda92a0`. This
validation pass took 404 seconds; elapsed run time was 3,295 seconds and peak
allocated VRAM was 11,916,105,728 bytes. The 128-step rolling train window
reported CE 0.8515 and Accuracy 0.6836; it is not a fixed-example gap measure.

At step 384, the validation evaluation completed and the validation-only
selector chose the checkpoint (score 0.419176; adapter SHA-256
`c942706eb00fc9f1058c536f5724365ff52fca5cbfbca5d3d5f10d2f358eca80`). The
512 validation sample IDs remain unique and in exactly the same order as the
baseline, step 128, and step 256 predictions (ordered-ID SHA-256
`00754c90a2155aa8a8a949489a3d7a80e6205943f21153047f434be61fda92a0`). Metrics:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.0040 | 0.0004 | 0.0038 |
| Image | 0.7188 | 0.7006 | 0.3362 | 0.0963 |
| Text | 0.7188 | 1.1038 | 0.3755 | 0.0988 |
| Video | 0.5234 | 0.8539 | 0.5453 | 0.0674 |
| Macro | 0.7402 | 0.6656 | 0.3143 | 0.0666 |

Compared with step 256 on the identical validation examples, macro Accuracy
rose by 2.15 points and macro NLL/Brier/ECE all improved. Video Accuracy rose
by 3.13 points, Text by 6.25 points, Audio remained at 100%, and Image fell by
0.78 points. The selector reported no overfit warnings. Video task-type results
were temporal descriptive 10/26 (0.3846 accuracy, 1.4844 NLL), explanatory
21/38 (0.5526, 0.6931), predictive 20/38 (0.5263, 0.6931), and counterfactual
16/26 (0.6154, 0.6931). These estimates use small per-type counts and the same
development validation repeatedly used for checkpoint selection.

By step 384, training had consumed 1,536 unique examples with no repeats,
1,412 unique underlying assets, 412 video scenes, and 512 CLEVRER parent
questions. Unique assets by modality were Audio 247, Image 369, Text 384, and
Video 412. Source examples were CLEVRER 512, Clevr-4 384, MASSIVE 192, Speech
Commands 128, LibriSpeech 128, Open-Jev 96, and Typed Decisions Synth 96. The
sample-ID order hash was
`83c0c0402d10bf3ecebb05f2ba030a2af09f858bebbdcdddf3e7b3b997d766d4`. Video
task counts were counterfactual 102, explanatory 154, predictive 154, and
temporal descriptive 102. The evaluation took 421 seconds; elapsed run time
was 4,645 seconds and peak allocated VRAM remained 11,916,105,728 bytes. The
128-step rolling training window had CE 0.8950 and Accuracy 0.6406; it is not
a fixed-example train/validation gap. Training continued after this evaluation
under the existing 2,048-step budget.

At step 512, validation Accuracy rose in aggregate, but the fixed selector did
not improve and kept step 384 as best. The step-512 adapter was retained as a
non-selected checkpoint (SHA-256
`d716424676465e0045be8738b0ab8987f0cec7c647b36d27737f9f2cace1334d`). The same
512 unique validation IDs and order were used (ordered-ID SHA-256
`00754c90a2155aa8a8a949489a3d7a80e6205943f21153047f434be61fda92a0`). Metrics:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.0007 | 0.0000 | 0.0007 |
| Image | 0.7813 | 0.7320 | 0.3198 | 0.0890 |
| Text | 0.7500 | 1.4155 | 0.4015 | 0.1459 |
| Video | 0.5078 | 0.8473 | 0.5530 | 0.0874 |
| Macro | 0.7598 | 0.7489 | 0.3186 | 0.0807 |

From step 384 to 512, Image Accuracy rose 6.25 points and Text 3.13 points;
Audio stayed at 100%, while Video fell 1.56 points. Macro Accuracy gained
1.95 points, but macro NLL, Brier, and ECE worsened, and the selector score
rose from 0.419176 to 0.503766 (higher is worse under this rule). The selector
recorded `train_loss_falling`, `validation_nll_rising`, and
`weak_modality_degraded`; it did not flag rising train Accuracy with stalled
validation Accuracy. Video task-type results were temporal descriptive 8/26 (0.3077
accuracy, 1.4519 NLL), explanatory 21/38 (0.5526, 0.6931), predictive 20/38
(0.5263, 0.6931), and counterfactual 16/26 (0.6154, 0.6931). The temporal
descriptive decline and Video Accuracy regression remain visible in the
record; this checkpoint was not selected.

Training had consumed 2,048 unique examples with no repeats, 1,837 unique
underlying assets, 514 video scenes, and 683 CLEVRER parent questions. Unique
assets by modality were Audio 327, Image 485, Text 511, and Video 514. Source
example counts were CLEVRER 683, Clevr-4 512, MASSIVE 256, Speech Commands 170,
LibriSpeech 171, Open-Jev 128, and Typed Decisions Synth 128. Video task counts
were counterfactual 136, explanatory 205, predictive 205, and temporal
descriptive 137. The sample-ID order hash was
`2a6125de48ac552a9bb0738e9f1ff29f183a0e9b476eb0741d1e8e84e7e6d048`. Evaluation
took 404 seconds; elapsed run time was 5,929 seconds and peak allocated VRAM
remained 11,916,105,728 bytes. The 128-step rolling training window had CE
0.7441 and Accuracy 0.6875; these are not fixed-example generalization-gap
metrics. Training continued after evaluation under the existing budget.

At step 640, the selector again kept step 384 (score 0.478155 versus best
0.419176). The step-640 adapter checkpoint was retained but not selected
(SHA-256 `c6187617f860747a6fa0ce37f80ab8561085986c9dba32f9465cb29edf310c52`).
All 512 validation IDs were unique and in the same order as earlier evaluations
(ordered-ID SHA-256
`00754c90a2155aa8a8a949489a3d7a80e6205943f21153047f434be61fda92a0`). Metrics:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.0022 | 0.0004 | 0.0020 |
| Image | 0.7422 | 0.7621 | 0.3520 | 0.1270 |
| Text | 0.7578 | 1.1704 | 0.3566 | 0.1151 |
| Video | 0.5078 | 0.9310 | 0.6137 | 0.1782 |
| Macro | 0.7520 | 0.7164 | 0.3307 | 0.1056 |

Against step 512, Video Accuracy was unchanged, Image fell 3.91 points, Text
rose 0.78 points, and Audio stayed at 100%. Macro Accuracy fell 0.78 points;
macro NLL improved while macro Brier/ECE worsened. The selector reported
`train_accuracy_rising_validation_stalled`; the other three monitored signals
were false. This is a validation overfitting warning, not a fixed-example
training/generalization gap. Video task-type metrics were temporal descriptive
8/26 (0.3077 Accuracy, 1.6864 NLL), explanatory 21/38 (0.5526, 0.7557),
predictive 20/38 (0.5263, 0.7580), and counterfactual 16/26 (0.6154, 0.6844).

For the paired Video Accuracy change from step 512 to 640, a diagnostic
scene-cluster bootstrap resampled 77 validation scenes with replacement,
20,000 replicates, Python `random.Random(17)`, percentile 95% interval. The
observed difference was 0.0000; the interval was [-0.0488, 0.0472]. The small
number of validation scenes and repeated use of this development set limit
the inference; this is not a blind audit result.

At step 640, training had consumed 2,560 unique examples with no repeats,
2,245 unique underlying assets, 601 video scenes, and 853 CLEVRER parent
questions. Unique assets by modality were Audio 407, Image 599, Text 638, and
Video 601. Source examples were CLEVRER 853, Clevr-4 640, MASSIVE 320, Speech
Commands 213, LibriSpeech 214, Open-Jev 160, and Typed Decisions Synth 160.
Video task counts were counterfactual 170, explanatory 256, predictive 256,
and temporal descriptive 171. The consumed sample-ID order hash was
`6edefab4888ac867b26bf81f84f1a952b516ec8dfc00bf368657ca45890e9d23`. The
evaluation took 408 seconds; elapsed run time was 7,193 seconds and peak
allocated VRAM was 12,008,189,952 bytes. The rolling 128-step train window
reported CE 0.7452 and Accuracy 0.6992; it is not a fixed-example gap. Training
continued under the predeclared 2,048-step limit and patience of 17 evaluations.

At step 768, validation improved under the fixed selector and this checkpoint
became the new best (score 0.409989; adapter SHA-256
`17b81b8c3e8888e136f6bc0c4a832e75ecfe53578aa254e18e66a5c26ed6004b`). The
same 512 unique validation IDs and order were used (ordered-ID SHA-256
`00754c90a2155aa8a8a949489a3d7a80e6205943f21153047f434be61fda92a0`). Metrics:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.0012 | 0.0002 | 0.0011 |
| Image | 0.7734 | 0.7010 | 0.3208 | 0.1120 |
| Text | 0.7500 | 1.1003 | 0.3737 | 0.1503 |
| Video | 0.5156 | 0.8313 | 0.5454 | 0.0724 |
| Macro | 0.7598 | 0.6584 | 0.3100 | 0.0839 |

Against step 640, macro Accuracy increased 0.78 points and macro NLL, Brier,
and ECE improved. Video Accuracy increased 0.78 points (one additional correct
answer among 128); Image increased 3.13 points, Text fell 0.78 points, and
Audio remained at 100%. The selector reported only `train_loss_falling`; the
validation NLL did not rise, and neither weak-modality-degradation nor stalled
validation accuracy was flagged. Video task-type results were temporal
descriptive 9/26 (0.3462 Accuracy, 1.3731 NLL), explanatory 21/38 (0.5526,
0.6931), predictive 20/38 (0.5263, 0.6931), and counterfactual 16/26 (0.6154,
0.6931).

For the paired Video Accuracy change from step 640 to 768, a diagnostic
scene-cluster bootstrap resampled 77 validation video IDs with replacement,
20,000 replicates, Python `random.Random(17)`, percentile 95% interval. The
observed difference was +0.0078; the interval was [-0.0317, 0.0496]. This
development validation was also used for selection, so the interval is a
descriptive uncertainty check, not blind-audit inference.

Training had consumed 3,072 unique examples with no repeats, 2,629 unique
underlying assets, 671 video scenes, and 1,024 CLEVRER parent questions. Unique
assets by modality were Audio 484, Image 709, Text 765, and Video 671. Source
examples were CLEVRER 1,024, Clevr-4 768, MASSIVE 384, Speech Commands 256,
LibriSpeech 256, Open-Jev 192, and Typed Decisions Synth 192. Video task counts
were counterfactual 205, explanatory 307, predictive 307, and temporal
descriptive 205. The consumed sample-ID order hash was
`ed9a665a55073abdb6c4b17ce3d6aa57ab7889431b57eef4c4a26c2d93488050`. Evaluation
took 396 seconds; elapsed run time was 8,440 seconds and peak allocated VRAM
was 12,008,189,952 bytes. The rolling 128-step training window had CE 0.7323
and Accuracy 0.6953; it is not a fixed-example generalization gap. Training
continued under the unchanged 2,048-step budget.

At step 896, the selector chose a new best checkpoint (score 0.397738; adapter
SHA-256 `a4622774abe48477e68b6f9aab1dca83cd0c1e880abb1805aa70bf731215f9bd`).
All 512 validation IDs remained unique and in the same order as the previous
evaluations (ordered-ID SHA-256
`00754c90a2155aa8a8a949489a3d7a80e6205943f21153047f434be61fda92a0`). Metrics:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.0004 | 0.0000 | 0.0004 |
| Image | 0.8359 | 0.5096 | 0.2359 | 0.0849 |
| Text | 0.7500 | 0.9843 | 0.3469 | 0.1361 |
| Video | 0.4297 | 1.0012 | 0.6168 | 0.1709 |
| Macro | 0.7539 | 0.6238 | 0.2999 | 0.0981 |

Against step 768, Image rose 6.25 points and Text was unchanged; Audio remained
at 100%, but Video fell 8.59 points. Macro Accuracy declined 0.59 points while
macro NLL, Brier, and ECE improved; the selector score therefore improved and
selected step 896 even as its minimum-modality Accuracy fell from 0.5156 to
0.4297. It recorded `train_loss_falling`,
`train_accuracy_rising_validation_stalled`, and `weak_modality_degraded`.
This is an explicit trade-off and overfitting warning, not evidence that the
Video decline is acceptable. Video task-type metrics were temporal descriptive
10/26 (0.3846 Accuracy, 2.1109 NLL), explanatory 17/38 (0.4474, 0.7242),
predictive 18/38 (0.4737, 0.7086), and counterfactual 10/26 (0.3846, 0.7237).

For the paired Video Accuracy change from step 768 to 896, a diagnostic
scene-cluster bootstrap resampled 77 validation video IDs with replacement,
20,000 replicates, Python `random.Random(17)`, percentile 95% interval. The
observed difference was -0.0859; the interval was [-0.2339, 0.0615]. The
interval is wide and includes zero; it is a development-validation diagnostic,
not a blind-audit result.

At step 896, training had consumed 3,584 unique examples with no repeats,
2,995 unique underlying assets, 728 video scenes, and 1,195 CLEVRER parent
questions. Unique assets by modality were Audio 559, Image 818, Text 890, and
Video 728. Source examples were CLEVRER 1,195, Clevr-4 896, MASSIVE 448, Speech
Commands 298, LibriSpeech 299, Open-Jev 224, and Typed Decisions Synth 224.
Video task counts were counterfactual 239, explanatory 358, predictive 359,
and temporal descriptive 239. The consumed sample-ID order hash was
`ff17c276ebcd9afe1d909c82b0d2e020cdf64b90ad243d2e16d7c5ce035d226d`. Evaluation
took 380 seconds; elapsed run time was 9,669 seconds and peak allocated VRAM
was 12,008,189,952 bytes. The rolling 128-step training window had CE 0.6440
and Accuracy 0.7383; it is not a fixed-example generalization gap. Training
continued under the unchanged 2,048-step budget and early-stopping patience.

At step 1024, validation improved under the fixed selector and the checkpoint
became the new best (selection score 0.373490; adapter SHA-256
`2d617045b1a953ff248e8508f63e03354adbc319f712754fdde503d76e630be6`). The
same 512 validation IDs were used in the same order (ordered-ID SHA-256
`00754c90a2155aa8a8a949489a3d7a80e6205943f21153047f434be61fda92a0`). Metrics:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.0032 | 0.0004 | 0.0030 |
| Image | 0.7578 | 0.6223 | 0.3122 | 0.1009 |
| Text | 0.7813 | 0.9764 | 0.3521 | 0.1613 |
| Video | 0.5000 | 0.8673 | 0.5626 | 0.1256 |
| Macro | 0.7598 | 0.6173 | 0.3068 | 0.0977 |

Compared with step 896, Video Accuracy increased 7.03 points, Text increased
3.13 points, Image fell 7.81 points, and Audio stayed at 100%. Macro Accuracy
increased 0.59 points, macro NLL improved by 0.0065, macro Brier worsened by
0.0069, and macro ECE improved slightly. Minimum-modality Accuracy rose from
0.4297 to 0.5000. The selector emitted no warnings at this checkpoint; this
does not erase the earlier step-896 warning or establish a fixed-example
train/validation gap. The rolling training window had CE 0.7130 and Accuracy
0.6914; it covers a different rolling set of examples and is not a direct
generalization-gap estimate.

Video question-type metrics were temporal descriptive 7/26 (0.2692 Accuracy,
1.5875 NLL), explanatory 21/38 (0.5526, 0.6731), predictive 20/38 (0.5263,
0.6936), and counterfactual 16/26 (0.6154, 0.6847). Thus the overall Video
gain was mixed: temporal descriptive declined from 0.3846 at step 896 while
the other three task types improved.

For the paired Video Accuracy change from step 896 to 1024, the saved
predictions were matched by sample ID and scene-cluster bootstrap resampled 77
validation scenes with replacement, 20,000 replicates, Python
`random.Random(17)`, percentile 95% interval. The observed difference was
+0.0703; the interval was [-0.0744, 0.2143]. This is a wide development-set
diagnostic, and the same validation data participates in checkpoint selection;
it is not blind-audit evidence.

At step 1024, the run had consumed 4,096 unique examples with no repeats,
3,370 unique underlying assets, 782 video scenes, and 1,365 CLEVRER parent
questions. Unique assets by modality were Audio 637, Image 934, Text 1,017,
and Video 782. Source examples were CLEVRER 1,365, Clevr-4 1,024, MASSIVE 512,
Speech Commands 341, LibriSpeech 342, Open-Jev 256, and Typed Decisions Synth
256. Video task counts were counterfactual 273, explanatory 409, predictive
410, and temporal descriptive 273. The consumed sample-ID order hash was
`33950ba758dd8767587c6a97868f9169308f463582cb16eb5b8dbc64112a3a80`. Evaluation
took 386 seconds; elapsed run time was 10,864 seconds and peak allocated VRAM
was 12,008,189,952 bytes. Evaluation completed and the same training process
continued beyond step 1024 under the unchanged run configuration.

At step 1152, validation did not improve under the fixed selector; step 1024
remained best (selection score 0.373490). The step-1152 adapter was retained
separately (SHA-256
`e3915cfb2848909f4c531b42e36603465eb4244763dafee2db4cf3091561fcdb`). The
512 validation IDs and their order remained unchanged. Metrics:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.0014 | 0.0002 | 0.0013 |
| Image | 0.7422 | 0.6118 | 0.3195 | 0.1462 |
| Text | 0.7891 | 1.0967 | 0.3526 | 0.1685 |
| Video | 0.5078 | 0.8735 | 0.5646 | 0.0741 |
| Macro | 0.7598 | 0.6459 | 0.3092 | 0.0975 |

Compared with step 1024, Video and Text Accuracy each rose 0.78 points, Image
fell 1.56 points, and Audio stayed at 100%; macro Accuracy was unchanged.
Macro NLL worsened from 0.6173 to 0.6459, Brier worsened from 0.3068 to
0.3092, and ECE changed only slightly from 0.0977 to 0.0975. The selector
reported `train_loss_falling`, `validation_nll_rising`, and
`train_accuracy_rising_validation_stalled`; `weak_modality_degraded` was false.
The rolling 128-step training window had CE 0.6531 and Accuracy 0.7402; this
rolling metric is not a fixed-example generalization gap.

Video question-type metrics were temporal descriptive 8/26 (0.3077 Accuracy,
1.6008 NLL), explanatory 21/38 (0.5526, 0.6797), predictive 20/38 (0.5263,
0.6931), and counterfactual 16/26 (0.6154, 0.6931). Relative to step 1024,
only temporal-descriptive Accuracy rose; the other three task-type accuracies
were unchanged. Temporal descriptive NLL increased slightly.

For the paired Video Accuracy difference from step 1024 to 1152, predictions
were matched by sample ID and a scene-cluster bootstrap resampled 77 validation
scenes, 20,000 replicates, Python `random.Random(17)`, percentile 95% interval.
The observed difference was +0.0078 and the interval was [0.0000, 0.0252].
This small change is a development-validation diagnostic on data also used for
selection, not blind-audit evidence.

At step 1152, training had consumed 4,608 unique examples with no repeats,
3,733 unique underlying assets, 827 video scenes, and 1,536 CLEVRER parent
questions. Unique assets by modality were Audio 715, Image 1,049, Text 1,142,
and Video 827. Source examples were CLEVRER 1,536, Clevr-4 1,152, MASSIVE 576,
Speech Commands 384, LibriSpeech 384, Open-Jev 288, and Typed Decisions Synth
288. Video task counts were counterfactual 307, explanatory 461, predictive
461, and temporal descriptive 307. The consumed sample-ID order hash was
`4fc14bdc87723d691b1b3969983430eaf5d65d6e25b7895fd50259f18c00d0ee`. Evaluation
took 377 seconds; elapsed run time was 12,079 seconds and peak allocated VRAM
was 12,008,189,952 bytes. The run continued beyond step 1152 under the same
configuration.

At step 1280, validation did not improve under the fixed selector, so step 1024
remained best (selection score 0.373490). The step-1280 checkpoint was retained
separately (adapter SHA-256
`e7113d4c71946c7ce628339c5b60bb3830ae774ff2065591c89fccaa1b70f3e4`). The
same 512 validation IDs and order were used. Metrics:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.0013 | 0.0001 | 0.0012 |
| Image | 0.7891 | 0.5258 | 0.2806 | 0.1162 |
| Text | 0.8047 | 1.0909 | 0.3379 | 0.1618 |
| Video | 0.5156 | 1.0114 | 0.6371 | 0.2357 |
| Macro | 0.7773 | 0.6573 | 0.3139 | 0.1287 |

Compared with step 1152, macro Accuracy increased 1.76 points; Image rose
4.69 points, Text 1.56 points, Video 0.78 points, and Audio stayed at 100%.
However, macro NLL worsened from 0.6459 to 0.6573, Brier from 0.3092 to
0.3139, and ECE from 0.0975 to 0.1287. Video Accuracy increased slightly
while its NLL, Brier, and ECE worsened, indicating a calibration trade-off.
The selector recorded `train_loss_falling` and `validation_nll_rising`.
Its `train_accuracy_rising_validation_stalled` and
`weak_modality_degraded` signals were false. The rolling training window had
CE 0.6286 and Accuracy 0.7422; it is not a fixed-example train/validation gap.

Video question-type metrics were temporal descriptive 9/26 (0.3462 Accuracy,
1.8815 NLL), explanatory 21/38 (0.5526, 0.9481), predictive 20/38 (0.5263,
0.7333), and counterfactual 16/26 (0.6154, 0.6404). Relative to step 1152,
only temporal-descriptive Accuracy changed (+3.85 points); other type
accuracies were unchanged, while NLL worsened for three types and improved for
counterfactual questions.

For paired Video Accuracy from step 1024 to 1280, predictions were matched by
sample ID. A scene-cluster bootstrap resampled 77 validation scenes, 20,000
replicates, Python `random.Random(17)`, percentile 95% interval. The observed
difference was +0.0156 with interval [0.0000, 0.0407]. From step 1152 to 1280,
the difference was +0.0078 with interval [0.0000, 0.0250] under the same
procedure. Both are repeated-development-validation diagnostics; checkpoint
selection and many experiment looks limit inference, and neither is a blind
audit result.

At step 1280, training had consumed 5,120 unique examples with no repeats,
4,082 unique underlying assets, 866 video scenes, and 1,707 CLEVRER parent
questions. Unique assets by modality were Audio 794, Image 1,155, Text 1,267,
and Video 866. Source examples were CLEVRER 1,707, Clevr-4 1,280, MASSIVE 640,
Speech Commands 426, LibriSpeech 427, Open-Jev 320, and Typed Decisions Synth
320. Video task counts were counterfactual 341, explanatory 512, predictive
512, and temporal descriptive 342. The consumed sample-ID order hash was
`28be172f2f3e9384116a4470023adc3607e5413f1c1bea12eb58b9a80a72ccf1`. Evaluation
took 378 seconds; elapsed run time was 13,270 seconds and peak allocated VRAM
was 12,008,189,952 bytes. The unchanged run continued beyond step 1280.

## Primary source references

- [Official MASSIVE repository](https://github.com/alexa/massive)
- [Official license notice](https://github.com/alexa/massive/blob/main/NOTICE.md)
- [Amazon Science dataset announcement](https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding)
