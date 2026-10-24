# Perception Test audio-to-visual source candidate preflight

## Status

This is a train-only task-format and annotation-feasibility result. It does not
admit Perception Test as a product-quality benchmark, establish that both
modalities are required, or report model quality. No validation or audit
annotations were loaded, no media was decoded, and no model was fitted.

## Source and reproducibility

- Dataset: Google DeepMind Perception Test, pinned repository revision
  `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
- Inspected member: official `all_train.json` only.
- Archive: `C:\CodexArtifacts\pretrained-reuse-r2\perception-test-source-audit-01\train_annotations.zip`.
- Archive SHA-256:
  `9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
- The annotation archive and generated corpus stay outside Git.
- Generated corpus manifest and JSONL are under
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-preflight-20261024-04\`.
- Generated JSONL SHA-256:
  `d20f0b7256503d7486fa9c4e439a4a8e70398c2dbfd8396db7ee0a45f98394c4`.
- Adapter SHA-256:
  `3a239d1fc092bd6242357500214c2b7ce47af48add4493d987ad9403d4c7f1d1`.
- Corpus builder SHA-256:
  `7b53b2b63c99d4626d5c183526a14247dfccd6a033e38a4ee0f2ef5103488fb8`.

The dataset describes sound intervals with inclusive start/end frame IDs and
microsecond audio timestamps. Each tracked object has frame-aligned normalized
bounding boxes. The adapter validates frame/box/mask alignment and rejects
malformed values.

## Task definition

Given an audio interval and object-crop proposals from overlapping video frames,
select every visible object track annotated as a source of that sound. This is
**candidate re-ranking conditional on an oracle event interval and oracle object
tracks**. It does not detect event boundaries, discover object proposals, or
provide an end-to-end real-time audiovisual decision.

The task retains all annotated parent objects as positives; it does not collapse
multi-source events into an arbitrary single target. An event is admitted only
when it is annotated visible, every source parent is known and has an unmasked
box in the event interval, and at least one non-parent object has an unmasked
box in the interval. Masked boxes, unknown parent IDs, missing parents, and
no-distractor events are held out with reasons.

Runtime input contains the audio interval and candidates' sampled normalized
frame boxes. It excludes sound labels, object labels, annotation IDs, video IDs,
and source object IDs. Candidate choice identifiers are `option_N`, assigned
after a deterministic per-event hash ordering. Dataset identifiers used for
media retrieval, grouping, provenance, or supervision remain outside the
model-request fields. A downstream collator must enforce this boundary and
must not serialize `media_reference`, `group_id`, or `supervision` as model
features.

At most four evenly spaced visible boxes per track are retained. This limits
record size but is an implementation convenience, not a validated temporal
sampling policy. The crops still need to be decoded from the original frames;
no crops or media tensors are in the current corpus.

## Train-only annotation yield

The deterministic preflight adapter processed all 25,757 annotated sound
intervals across 2,184 train videos:

| Status | Events |
|---|---:|
| Candidate | 10,154 |
| Held | 15,603 |

Candidates span 1,982 videos. Candidate set sizes range from 2 to 49 objects.
The candidate-set distribution and positive-count distribution are in the
external generated manifest. There are 3,255 single-parent events, 5,618 with
two parents, and a long tail through 11 parents. These are source annotation
counts, not accuracy results.

The earlier exploratory count of 10,156 used a less strict bbox eligibility
check. This adapter excludes masked boxes, which reduces the eligible count by
two. The stricter count is used here. It also refuses unknown `-1` parent IDs
instead of treating them as negatives.

## Remaining gates before model work

1. The source inventory must identify the corresponding audio/video files and
   prove every event crop lies within the decoded media duration. The current
   source archive alone does not establish that.
2. Freeze a new video-grouped train/validation partition before fitting. Do not
   use previously inspected validation cohorts or any sealed audit annotations.
3. Define audio-only, video-only, question/label-prior, candidate-order-shuffle,
   audio-shuffle, stale-window, and interval-offset baselines/controls before
   model selection. Match the candidate universe across modalities.
4. Verify the task does not reduce to sound-category or object-order priors.
   Candidate count is highly variable (2–49), and the 15 observed sound labels
   are imbalanced; stratified source reporting and balanced metrics are needed.
5. Establish whether annotated parent objects have consistent causal/source
   semantics for the intended decision. A positive association is not itself a
   counterfactual proof that both modalities are necessary.
6. Reuse only media explicitly admitted for this train-only purpose; do not
   decode old validation media or begin GPU work as part of this preflight.

Only after these gates pass should a small frozen-feature readout be considered.
The preflight must not be presented as validation performance, a modality
complementarity result, or evidence toward the 95–98% product target.

## Implementation

- Adapter: `src/tiny_omni_decision/perception_source_candidate.py`.
- Deterministic writer: `scripts/build_perception_source_candidate_corpus.py`.
- Regression tests: `tests/test_perception_source_candidate.py`.
- Existing visibility-proposal format remains unchanged and separate.
