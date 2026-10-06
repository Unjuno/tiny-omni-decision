# Physion++ video development reference

## Why this was evaluated

The remote `main` plan now prioritizes an EmbeddingGemma 2 ternary-first
student experiment, with the frozen Gemma 4 Decision Teacher providing
option-level supervision. That plan does not establish the separate project
quality gate of 90% Accuracy on each modality. This report therefore records a
development-only video generalization check for the existing frozen Teacher
references. It does not change Teacher v1, Candidate E, or the E-long run, and
it does not promote an artifact.

## Source and split

The candidate source is the reviewed Physion++ `readout_data.zip` RGB media
archive (MIT; cite Hsiao-Yu Tung et al., Physion++). The archive metadata and
labels have been inspected, so this material is not eligible as a fresh sealed
audit. The adapter defines one retrospective binary question over each full
clip: whether a marked target was contacted at any time. It is not a
future-prediction task and is not a paper-faithful reproduction of the
published readout benchmark.

The pinned archive identity is size 2,971,891,321 bytes, ETag
`"4cf84f667bcc3eaf991dbcc10dd31177-355"`, and Last-Modified
`Sat, 10 Jun 2023 23:02:20 GMT`. Metadata-record SHA-256 is
`e673b6ba5ee3f96a75051b946003e0dafafdeb8bc12b034c7699cf2934d0b84c`; the
mapped media-ID SHA-256 is
`59abe4999b4485139ce925108dd96eca8bdb1f08e2ebf2aae04ae9768f39dc77`.
The archive inventory differs from the paper summary: it contains 832 metadata
rows, 800 mapped videos, and 615 `(scenario, trial_seed)` groups. The split
keeps each scenario/seed group together.

The frozen corpus is outside Git at
`C:\CodexArtifacts\tqpp\fullclip-v1`:

| Split | Examples | Unique videos | Groups | SHA-256 |
|---|---:|---:|---:|---|
| Train | 652 | 652 | 492 | `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b` |
| Validation | 148 | 148 | 123 | `1254f01256197f1c2e23f8fa13dc03f87789c328f32bf768300134dd6b70d8dc` |

The validation ID-order hash is
`5c1ded5d955f55cc5632227a1f1d33d12fc768ca549c32ad5039f3ccc65d8052`.
Train and validation have no source-ID, source-asset, media-identity, or
normalized-content overlap. The corpus manifest records per-video hashes.
This is a development split, not a blind final audit.

## Evaluation method

Teacher v1 and the E-long selected checkpoint were each evaluated once on the
same 148 validation examples and in the same order. Both used the pinned
Gemma 4 base and processor revision
`6befbaca7398925921802abd1f277b495b78b738`, the same option-scoring code, and
the same 8-frame preprocessing. V1 was trained with 4 frames, so the shared
8-frame comparison is not a reproduction of its training-time preprocessing.
No model was trained on Physion++ for these measurements.

| Frozen reference | Video Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Teacher v1 | 0.5405 | 0.8466 | 0.6034 | 0.2486 | 0.7557 |
| E-long selected, step 1,536 | 0.5743 | 0.9398 | 0.6277 | 0.2401 | 0.7969 |
| E-long minus v1 | +0.0338 | +0.0932 | +0.0243 | -0.0085 | +0.0412 |

Predictions were paired by sample ID and resampled by the 123
`(scenario, trial_seed)` clusters. A 10,000-replicate percentile bootstrap
(seed 17) gave the following 95% intervals for E-long minus v1:

| Metric delta | 95% interval |
|---|---:|
| Accuracy | [-0.0208, +0.0927] |
| NLL | [+0.0177, +0.1680] |
| Brier | [-0.0220, +0.0697] |
| ECE | [-0.0471, +0.0662] |

The accuracy interval includes zero. E-long has a higher point Accuracy but
also worse NLL and Brier; this small development comparison does not establish
a clear overall winner. The bootstrap is descriptive and does not account for
the source's label/domain limitations or earlier checkpoint selection on a
different validation set.

## Interpretation and next step

The two frozen teachers perform only modestly above chance on this novel
retrospective physical-interaction task, and E-long's higher accuracy comes
with worse probability quality. This could reflect limited temporal reasoning,
domain shift from CLEVRER, the narrow binary task, or label/source properties;
these results do not isolate a cause and do not demonstrate the 90% Video goal.

The 652 unique Physion++ training examples are a rights-cleared development
candidate for a separately versioned data-coverage experiment. Any such run
must keep the current Teacher artifacts and this validation corpus immutable,
record non-video regression on a matched multimodal validation suite, and must
not describe this already-inspected source as a sealed audit. The remote
ternary-first student plan remains a separate downstream path; no compression
or Recovery training was performed here.

## Reproduction artifacts

- Corpus builder: `scripts/build_physionpp_readout.py`
- Frozen Teacher evaluator: `scripts/evaluate_physionpp_reference.py`
- Paired cluster comparison: `scripts/compare_physionpp_references.py`
- Candidate manifest: `manifests/candidates/physionpp-readout.yaml`
- External metrics/predictions: `C:\CodexArtifacts\tqpp\evaluations\`
- External paired summary: `C:\CodexArtifacts\tqpp\physionpp-reference-comparison.json`
- Teacher v1 adapter SHA-256: `4e664a5500a3a109a48de7ec7c807d2b09b61cf386fd124ca8368e197802f951`
- E-long adapter SHA-256: `0497a25742451daeb2e08c7879dc221d174a17bf451e4ab2de0f3af6223b5009`
- Local compute: RTX 3080 Laptop 16 GB; each evaluation took about 10.4 minutes,
  with peak allocated VRAM 11.34 GB (v1) and 11.43 GB (E-long). Cloud cost $0.
