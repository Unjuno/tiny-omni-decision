# Perception Test source and task screen — 2026-11-11

## Decision

**Do not treat this as a new source admission.** Perception Test was already screened and used in multiple R2 development probes. This branch adds only a strict MCQA schema adapter and one train-only frozen-encoder runtime smoke. It does not authorize another quality-selected fit or reuse of exposed validation groups. Any future video-QA experiment would require an exact per-video exposure audit and a predeclared untouched development cohort; it still would be video-plus-text QA, not evidence that audio is necessary.

Do not inspect, download, or evaluate the official test split. Do not claim participant-disjoint evaluation: the paper reports around 100 participants, while participant IDs are absent from the inspected annotations. The group unit available for this task is the full video/scene ID. The paper also describes 37 script templates with 2–5 variations, so a video-level split does not test unseen-script generalization.

## Source and rights evidence

- Official repository: [`google-deepmind/perception_test`](https://github.com/google-deepmind/perception_test), inspected at commit `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`. This source was already used for prior R2 diagnostics; see [`R2_PERCEPTION_TEST_SOURCE_SCREEN_20261021.md`](R2_PERCEPTION_TEST_SOURCE_SCREEN_20261021.md), [`R2_PERCEPTION_AV_QUESTION_AUDIT_20261013.md`](R2_PERCEPTION_AV_QUESTION_AUDIT_20261013.md), [`R2_PERCEPTION_AV_TEMPORAL_MATCH_20261021.md`](R2_PERCEPTION_AV_TEMPORAL_MATCH_20261021.md), and [`R2_PERCEPTION_SOURCE_TASK_DISPOSITION_20261029.md`](R2_PERCEPTION_SOURCE_TASK_DISPOSITION_20261029.md). The R0 task battery explicitly closes the prior AV1/AV2 cohorts to further tuning.
- Pinned README SHA-256: `9f739a07616e3f60af0723800d813ca87953496ba9e288fc62b4bfd7578afa85`.
- Pinned repository `LICENSE` SHA-256: `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`. The README states software is Apache-2.0 and other materials are CC BY 4.0. The NeurIPS paper also identifies the public fine-tuning and validation splits as CC BY and says participants consented to use, publication, and storage. The paper says faces and voices were intentionally not recorded.
- Primary references: [official repository and data instructions](https://github.com/google-deepmind/perception_test), [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Paper-Datasets_and_Benchmarks.pdf).
- The public multiple-choice video-QA task uses three options, with questions labeled by Memory/Abstraction/Physics/Semantics and Descriptive/Explanatory/Predictive/Counterfactual reasoning. The authors report 91.4% human accuracy and 46.2% for the evaluated SOTA video-QA model; this is a hard diagnostic task, not a 95%-accuracy expectation.

## Annotation-only preflight

Only the official `train_annotations.zip` and `valid_annotations.zip` were fetched. No video/audio files, precomputed features, test annotations, or test media were accessed. ZIP metadata supplied ETags and lengths; local SHA-256 receipts are recorded in [`manifests/pretrained_reuse/perception-test-source-screen-20261111.json`](../../manifests/pretrained_reuse/perception-test-source-screen-20261111.json).

| Split | Archive video groups | Groups with MCQA | Video-QA questions | Sound segments | Visible single-source events | Video-ID overlap |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,184 | 1,955 | 7,392 | 25,757 | 6,170 | 0 |
| Validation | 5,900 | 5,260 | 19,140 | 69,767 | 16,824 | 0 |

Reasoning support (question counts):

| Split | Descriptive | Explanatory | Predictive | Counterfactual |
|---|---:|---:|---:|---:|
| Train | 6,144 | 852 | 247 | 149 |
| Validation | 15,858 | 2,286 | 639 | 357 |

The sound-source candidate requires one resolvable parent object, `is_visible=1`, a parent object track whose timestamp intersects the annotated sound interval, and at least two object tracks whose timestamps intersect that interval. This produced 4,066 candidate events across 1,437 train video groups and 11,211 across 3,997 validation video groups. The proposed decision is to choose the object label that produced an event from synchronized video/audio and supplied object labels. This is a candidate adapter design only; audio-only, video-only, both-input, and source-shuffled controls have not been evaluated.

## Adapter preflight and annotation anomaly

A strict annotation-only adapter and schema tests now exist in
`src/tiny_omni_decision/perception_test_dataset.py` and
`tests/test_perception_test_dataset.py`. The adapter accepts only the public
train/valid annotations, requires matching split and video IDs, preserves the
publisher's answer index, and rejects unsafe/missing media paths. Every choice
is labeled `A:`, `B:`, or `C:` uniformly. This is necessary because a complete
annotation scan found four validation questions with duplicate distractor
text; their correct `answer_id` points to a distinct third choice. Train has
0 such rows (7,392 total), validation has 4 (19,140 total). No rows are
excluded or merged. The fixed choice labels retain the source index while
satisfying the repository's unique-option decision schema.

The adapter was exercised over all 7,392 train and 19,140 validation QA rows
using synthetic relative media references. This validates annotation/schema
conversion, not media presence, decoding, feature extraction, or model quality.

## One train-only frozen-feature smoke

To verify that the local pinned Path A encoder could process a real source clip,
one official train member (`video_9431`) was fetched from the 28,404,087,974-byte
official `train_videos.zip` using HTTP byte ranges. The archive ETag, ZIP member
CRC, expanded size, and media SHA-256 are recorded in the external receipt
`D:\CodexArtifacts\pretrained-reuse-r2\perception-test-patha-smoke-20261111\media-receipt.json`
(SHA-256 `91a3131c03016956f8f8e662d5d2b6682bdbd0028b397fa91a24b12980b649af`).
The media is retained at
`D:\CodexArtifacts\pretrained-reuse-r2\perception-test-patha-smoke-20261111\media\video_9431.mp4`;
no other archive member was fetched. The receipt reports 1,050 H.264 frames at
30fps and an AAC stream. PyAV decoded eight uniformly spaced frames. Frozen
VideoPrism-B at the already pinned checkpoint/source produced finite
`[2048, 768]` FP32 tokens in the existing Python 3.11.16/JAX 0.6.2/Flax
0.11.1/TensorFlow 2.18.1/PyAV 18.1.0 CPU environment. First JIT forward was
5.89s; decode, preprocess, and forward took 13.36s; peak working set was
4.20GB. GPU, fitting, and quality scoring were not used. The feature array is
`D:\CodexArtifacts\pretrained-reuse-r2\perception-test-patha-smoke-20261111\videoprism-video_9431-tokens.npz`
(SHA-256 `2a13aa09175558291e334c218cef825a9b6ecb118cf348653456dc7a5b35fc4f`).
The runtime JSON receipt at
`D:\CodexArtifacts\pretrained-reuse-r2\perception-test-patha-smoke-20261111\videoprism-video_9431-feature-smoke.json`
has SHA-256 `3e2ad825c68b8ab8f35c52cc0e01445a9de3e12e22bb8a78430146239fbc823d`.

This smoke verifies a frozen video encoder path only. It does not evaluate any
MCQA answer, does not establish that `video_9431` was untouched by every prior
experiment, and does not clear the held-out group-exposure gate. Preserve the
existing R2 result artifacts and do not score validation as part of this smoke.

## Limits and next gate

- The official train/validation partition is disjoint by `video_id`; one exact train media hash now exists. Duplicate-content and cross-project media-identity gates remain open for any proposed cohort.
- Participant and script/variation IDs are not present in the annotation metadata. A video-ID split establishes clip/scene separation, not participant separation or unseen-script generalization.
- Video-QA is a text-conditioned video decision task. It is not a direct text-free video probe. The sound-source candidate is more explicitly audio-video, but its joint-input value is unproven.
- The paper's 91.4% human score makes the project-wide 95% target unsuitable as a guaranteed threshold for this diagnostic. Report actual scores and difficulty; do not redefine the project target.
- Full train and validation video archives are large (26.5 GB and 70.2 GB compressed); separate audio archives are 12.3 GB and 33.1 GB. A single verified train member was range-fetched to a new D: artifact directory; no full archive, validation media, audio-only archive, or test asset was fetched.
- Before any question-answer scoring or fitting: reconcile exact group IDs against every prior Perception Test artifact, freeze a new video-group cohort, preserve the participant/script limitation, and verify local media hashes. Do not use previously scored AV1/AV2 cohorts for selection. Keep official test split unopened.

The next gate is an exposure audit against all prior Perception Test group IDs
and a fresh-task/source decision. Do not fit or score this MCQA validation until
the audit proves a sufficient untouched video-group cohort and its selection
protocol is frozen. Prefer a different admitted source if that cannot be shown.
