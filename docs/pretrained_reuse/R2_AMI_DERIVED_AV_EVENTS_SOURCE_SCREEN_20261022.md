# AMI-derived audiovisual event segments source screen — 2026-10-22

## Decision

**REVIEW ONLY; do not access, ingest, fit, or score the hosted payload yet.**
The public dataset card reports a potentially useful AMI-derived set of
speaker/event-centered audio and video segments and identifies fields needed
for meeting/speaker grouping. However, the host requires explicit agreement to
share contact information before file access. The card's Apache-2.0 metadata
does not explain how the original AMI CC BY 4.0 media attribution and terms are
preserved in the redistributed audiovisual clips. No gated files, labels, or
archives were accessed.

## Public metadata and task fit

The public card describes 35,731 segments with `meeting_id`, `speaker_id`,
timestamps, event labels, and availability flags for audio, video, and lip
video. It reports 35,536 video segments and approximately 28,263 lip-video
segments. For the laughter subset it reports 469 events with audio, 450 with
video, and 346 with lip video. It describes ten disfluency classes, with all
disfluent words having audio but variable video/lip-video availability.

This could support a compact temporal speaker/event Decision probe. Laughter
segments could test whether a visible participant is the source of a vocal
event, while disfluency segments could test whether audio and visible speaking
activity agree. But laughter may be solvable from audio alone, and the public
metadata does not establish how many distinct meetings and speakers contribute
to each label, whether a task truly requires both inputs, or whether a valid
meeting-disjoint validation cohort has adequate power. Row count is not a
substitute for independent meeting count.

## Rights, access, and provenance questions

- The original AMI corpus owner states that the released signals and
  annotations are CC BY 4.0. The source license must be retained for any
  original AV media and derivatives built from it.
- The Hugging Face dataset card labels the derivative dataset Apache-2.0 but
  gates its files behind agreement to share contact information. The card does
  not provide a component-level rights statement explaining whether that
  Apache label applies only to new annotations/code or also purports to cover
  the original audiovisual media.
- Its card points to an author preprocessing repository, but this screen did
  not establish a pinned source revision, complete reproducible pipeline,
  per-asset provenance, or legal permission to redistribute each media crop.
- The labels appear event/speaker-associated. They do not by themselves prove
  that visual evidence is necessary or that candidate-speaker answers cannot
  be recovered from audio/transcript shortcuts.

Before reconsidering, require an explicit component-level license/provenance
statement from the dataset author, preserve AMI CC BY attribution for AMI
media, resolve what the hosted contact-information agreement permits, pin the
dataset and preprocessing revisions, and establish per-meeting/per-speaker
counts plus meeting-disjoint train/validation/audit splits. Then predeclare a
both-required task and audio-only, video-only, shuffled-video, and missing-input
controls. Do not accept the Apache-2.0 badge alone as sufficient evidence.

## Evidence inspected

- Hosted dataset card and metadata summary:
  <https://huggingface.co/datasets/hhoangphuoc/ami-disfluency>
- Author-referenced preprocessing repository:
  <https://github.com/hhoangphuoc/AVSL/blob/main/docs/Preprocess.md>
- Official AMI download and signal release information:
  <https://groups.inf.ed.ac.uk/ami/download/>
- Official AMI corpus license:
  <https://groups.inf.ed.ac.uk/ami/icsi/license.shtml>

The hosted card was read without accepting its access terms. No gated payload,
archive member, media, label row, or metadata file was read. No training,
feature extraction, evaluation, or GPU work was performed.

## Roadmap effect

This is a higher-specificity AV source lead that may make AMI media acquisition
more practical, but it is not an admitted task or benchmark and does not close
R2 M-2. Its immediate blockers are gated access, unclear derivative-media rights
and provenance, unknown meeting-level coverage, and unproven both-required task
semantics. Preserve the original AMI preflight unchanged.
