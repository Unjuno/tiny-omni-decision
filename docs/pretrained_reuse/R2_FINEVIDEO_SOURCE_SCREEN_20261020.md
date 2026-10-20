# FineVideo source screen — 2026-10-20

## Decision

FineVideo is a **conditional development-data candidate**, not yet admitted
for training or model selection. Its official dataset card describes 43,751
YouTube videos with video, speech transcripts, time-coded transcript segments,
scene/activity annotations, and video-level question/answer pairs. The card
attributes the upstream videos/transcripts to CC-BY YouTube-Commons and
provides provenance. This is a better rights and task starting point than
sources whose media rights are unknown, but it is not yet a complete source
packet for this product.

No dataset rows, QA, media, or labels were loaded in this screen. The dataset
is gated, its current usable revision must be confirmed against maintainer
removal updates, and its declared layout exposes only a `train` split. Do not
accept the gate terms, download data, or start an experiment until the exact
revision and per-video rights/provenance fields can be audited locally.

## What the official metadata establishes

- The card declares 43,751 videos and says the videos and transcripts were
  obtained from YouTube-Commons, whose source videos were shared under CC-BY.
- Records include video metadata, transcripts, time-coded speech, scene and
  activity annotations, audiovisual-correlation metadata, and five QA pairs
  per video. The annotation pipeline used Gemini for video descriptions and
  GPT-4o to structure them; this means QA/annotation labels are generated and
  need quality checks before being treated as truth.
- The only configured data split shown is `train`. A new development split
  could be created from untouched video IDs, but it must be frozen by video
  identity before reading QA targets, then checked for channel/source and
  normalized-content overlap. It must not be described as an upstream split.
- FineVideo's Terms require respecting each original license and attribution,
  keeping the dataset current with validated removal requests, and reproducing
  the Terms if redistributing or hosting it. The gated card also states that
  access shares the user's contact details with maintainers.

## Fit to the open R2-M gate

FineVideo could support *candidate* synchronized audiovisual decisions around
time-localized speech/activity, sound-versus-visual evidence, or scene-level
context. Its existence does not establish that an answer needs both streams:
the available QAs are video-level generated annotations and may be answerable
from the transcript alone, frames alone, or annotation wording. A valid task
packet must declare questions whose target can be independently checked and
include paired audio-only, video-only, shuffled-audio/video, missing-stream,
and timing-offset controls. Retain only examples where the label and controls
are defensible; report exclusions and realized task mix without tuning on the
held-out group.

Before admitting even a small development corpus, require:

1. explicit user acceptance of the gated terms if account access is needed;
2. a pinned dataset commit/revision and current-removal-status check;
3. row-level original video ID, channel/source identity, original license,
   attribution, and media hash;
4. a deterministic, video-grouped train/development split from unused upstream
   `train` records, with source/channel and content-overlap audits;
5. a predeclared, auditable joint-input decision task with validated labels and
   enough independent video groups for uncertainty estimates.

Until these checks pass, FineVideo is **REVIEW**. No GPU reservation through
HotGPU is justified. This screen made no model/data/checkpoint changes, did not
open any test or sealed-audit data, and incurred no compute cost.

## Official references

- Dataset card, structure, provenance, license discussion, and Terms:
  <https://huggingface.co/datasets/HuggingFaceFV/finevideo>
- Official data-preparation repository:
  <https://github.com/huggingface/fineVideo>
- Official creation and annotation methodology:
  <https://huggingface.co/blog/fine-video>

