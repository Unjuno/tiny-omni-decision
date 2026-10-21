# AVE source screen (2026-10-22)

## Disposition

**Review only; not admitted for acquisition, fitting, evaluation, or product
claims.** The Audio-Visual Event (AVE) dataset is a synchronized audiovisual
temporal-localization benchmark with 4,143 ten-second clips, 28 event
categories, and event-boundary annotations. Its source paper defines each
target event as both visible and audible. That makes it useful for studying
event localization, but the class/event label may often be recoverable from
either stream alone. The dataset definition therefore does not establish a
decision task where both inputs are necessary. The official split is described
in the paper as train/validation/test, but underlying video-group identity,
source overlap, and content-duplicate behavior have not been audited here.

The official repository instructions say that the authors do not own the movie
clips and that clips are publicly available through YouTube. The reviewed
official materials do not grant rights to the underlying media or establish
permission for downloading, derivative-model training, or commercial use.
Do not download or use the media absent source-specific authorization. No
dataset files, annotations, features, or video/audio media were accessed in
this screen.

## Why it does not close the joint-input gate

- The task is temporal localization of a preselected event category, rather
  than a caller-supplied-option decision with independently adjudicated proof
  that audio and video are jointly required.
- A joint-vs-audio-only/video-only comparison could be informative only after
  source rights, split grouping, duplicate checks, and class-wise unimodal
  baselines are established. It cannot be presumed to demonstrate fusion.
- AVE annotations are aligned to the AudioSet ontology. The separate AudioSet
  overlap and upstream pretraining contamination would also need an explicit
  audit before treating a held-out score as independent evidence.
- This is an older, single-event benchmark; its task construction does not
  directly test multi-event temporal flow or state-conditioned Decision reuse.

## Evidence reviewed

- The original paper defines an audiovisual event as both visible and audible,
  and describes 4,143 ten-second clips and temporal boundaries.
- The official project repository links its annotations/features and instructs
  users to retrieve movie clips from YouTube; it states the authors do not own
  those clips.
- The official materials reviewed here do not specify an applicable license
  for the underlying clips or authorization for derivative model use.
- No data were opened or acquired. No evaluation or model run was performed.

## Sources

- [Original AVE paper (Rochester author-hosted PDF)](https://www.cs.rochester.edu/~cxu22/p/eccv2018_ave_paper.pdf)
- [Official AVE-ECCV18 repository](https://github.com/YapengTian/AVE-ECCV18)
- [Official Rochester audiovisual research project page](https://sites.google.com/view/audiovisualresearch/)

This screen is a source/task rejection for the current goal, not evidence that
audio-video fusion is generally unhelpful. Keep the synchronized AV task gate
open until a rights-cleared, group-audited task establishes conditional
multi-input necessity.
