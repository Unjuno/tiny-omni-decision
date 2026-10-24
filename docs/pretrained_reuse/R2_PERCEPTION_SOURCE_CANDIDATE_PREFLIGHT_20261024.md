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
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-preflight-20261024-05\`.
- Generated JSONL SHA-256:
  `613daa46964f7595568fb32311e50fe59d0e88a4f02ce77be22a24aee9f03b16`.
- Generated manifest SHA-256:
  `2d41b7bb0dccf01298f4218d2ff3fd88110a350d1ba093a855314a191a7a4274`.
- Adapter SHA-256:
  `2d89893026a043ec6dcbe24d52ee5ffb2568766ad124e22312de14fd1f928d7a`.
- Corpus builder SHA-256:
  `7b53b2b63c99d4626d5c183526a14247dfccd6a033e38a4ee0f2ef5103488fb8`.

The dataset describes sound intervals with inclusive start/end frame IDs and
microsecond audio timestamps. Each tracked object has frame-aligned normalized
bounding boxes. The adapter validates frame/box/mask alignment, rejects
malformed values, and holds every event whose end exceeds the shorter of the
metadata audio and video durations. This matches the existing source audit's
boundary warning; no media was decoded in this step.

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
intervals across 2,184 train videos. It held all 247 events whose end exceeded
metadata audio duration (none exceeded the metadata video duration); 15 of the
otherwise eligible candidates were removed by this endpoint gate:

| Status | Events |
|---|---:|
| Candidate | 10,139 |
| Held | 15,618 |

Candidates span 1,982 videos. Candidate set sizes range from 2 to 49 objects.
The candidate-set distribution and positive-count distribution are in the
external generated manifest. There are 3,255 single-parent events, 5,618 with
two parents, and a long tail through 11 parents. These are source annotation
counts, not accuracy results.

An earlier exploratory count of 10,156 used a less strict bbox eligibility
check. This adapter excludes masked boxes and media-boundary violations and
refuses unknown `-1` parent IDs instead of treating them as negatives. Its
initial pre-boundary count was 10,154; applying the endpoint gate removes 15
otherwise eligible events.

Read-only filename matching against the existing Perception Test
`perception-av-event-train-media-01` directory found 400 local train videos
covering 2,339 of the current candidate events across all 400 candidate groups.
This confirms file presence only. Those files were not decoded or scored, and
their duration/synchronization checks remain outstanding for this task.

A deterministic CPU-only smoke selected 32 candidate events from 32 distinct
official-train video groups (seed 17) in that existing media directory. All 32
MP4 sizes and SHA-256 values matched the acquisition receipt. PyAV decoded all
77 requested bbox frame IDs; decoded presentation times fell within the
annotated event interval with a two-frame tolerance. The audio decoder covered
all 66.536017 seconds of the 32 annotated intervals, with less than 1e-9 seconds
aggregate difference. The report is
`C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-media-smoke-20261025-03\media-smoke.json`,
SHA-256 `9d557f74fc97e459a63c6fe40edcd38e9ade15723e848210801b2c2a89660a98`.
This checks 32 of 400 files; it does not decode candidate crops or calculate
model metrics and is not a representative sample.

## Provisional development split

Before fitting, a seed-17 deterministic hash split was frozen over the 400
candidate-bearing official-train videos with local media. The 80 held-out
groups contain 435 events; the 320 training groups contain 1,904 events. Video
ID overlap and exact media SHA-256 overlap are both zero. This is a
development-only split of a previously inspected source, not a blind validation
or audit. The prior source/task work used the same video media for other labels;
participant-level grouping is unavailable, and normalized near-duplicate
content has not been checked.

External artifacts:

- Directory: `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-dev-split-20261025-01\`.
- Manifest SHA-256:
  `506fe453cff369885dc543774ee224db50f59b881e58d5768f17b67c11b693d7`.
- Train JSONL SHA-256:
  `a979f9dce76e52f74b4295dab5fd0e59feddf7fbe6f460de5b4cf020b490f0cb`.
- Development validation JSONL SHA-256:
  `8d2a159b21fe24fe91ce5ff30ac7d8f3a62c0c669e09e7a010271bec5ea8ad1c`.
- Splitter script SHA-256:
  `c5a713fd3af6b5bdb10c81fe4a45e723a4888ea143c54d5a4e76d0d1b5417947`.
- CPU media smoke script SHA-256:
  `2fda873f9aebc27e2ae607f7d084878514ff2926acacca1ca33bf9aa8aa9ed7c`.

The remaining 1,582 candidate-bearing official-train video groups without
matching local media are excluded from this frozen pilot; they were not
silently assigned to either partition.

## Remaining gates before model work

1. Verify decoded duration, synchronization, and event boundaries for the
   remaining 368 videos in this pilot. Do not use previously inspected
   validation cohorts or sealed audit annotations.
2. Define audio-only, video-only, question/label-prior, candidate-order-shuffle,
   audio-shuffle, stale-window, and interval-offset baselines/controls before
   model selection. Match the candidate universe across modalities.
3. Verify the task does not reduce to sound-category or object-order priors.
   Candidate count is highly variable (2–49), and the 15 observed sound labels
   are imbalanced; stratified source reporting and balanced metrics are needed.
4. Establish whether annotated parent objects have consistent causal/source
   semantics for the intended decision. A positive association is not itself a
   counterfactual proof that both modalities are necessary.
5. Reuse only media explicitly admitted for this train-only purpose; do not
   decode old validation media or begin GPU work as part of this preflight.

Only after these gates pass should a small frozen-feature readout be considered.
The preflight must not be presented as validation performance, a modality
complementarity result, or evidence toward the 95–98% product target.

## Implementation

- Adapter: `src/tiny_omni_decision/perception_source_candidate.py`.
- Deterministic writer: `scripts/build_perception_source_candidate_corpus.py`.
- CPU media smoke: `scripts/preflight_perception_source_candidate_media.py`.
- Group split freezer: `scripts/freeze_perception_source_candidate_split.py`.
- Regression tests: `tests/test_perception_source_candidate.py`.
- Existing visibility-proposal format remains unchanged and separate.
