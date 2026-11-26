# Ego4D Social “Talking-to-me” source screen — 2026-10-22

## Decision

**REVIEW / NOT ADMITTED.** This is the strongest newly identified naturalistic
AV Decision task in the current source search: determine whether a supplied,
tracked visible person is speaking to the camera wearer from a synchronized
conversation segment. It maps naturally to a binary or caller-ID choice output.
The task is promising because it combines speech activity, visible-person tracks,
and a directed-to-camera-wearer label. This is not yet evidence that both model
streams are indispensable on this dataset; audio-only, video-only, joint, missing,
and timing controls remain necessary.

Do not acquire annotation/media payloads, train, or evaluate until the applicable
Ego4D license agreement is reviewed and accepted by an authorized party. The
official access instructions require an accepted agreement for both annotations
and data. The public agreement endpoint was not readable in this screen (HTTP
403), so permitted research, derivative-model, redistribution, and product use
are all **UNKNOWN**. No agreement was accepted, no gated content was accessed,
and no media or annotation files were fetched.

## Pinned public source evidence

| Source | Pin / hash | What it establishes |
| --- | --- | --- |
| Official Ego4D benchmark docs — [AV diarization](https://ego4d-data.org/docs/benchmarks/av-diarization/) and [Social](https://ego4d-data.org/docs/benchmarks/social/) | Public pages checked 2026-10-22; pages do not expose a release commit | AV diarization includes face boxes/tracking, active-speaker labels, voice activity, and transcription. The Social `Talking to me` task uses a video and audio segment plus tracked faces and speaker status to label whether a person is speaking to the camera wearer. |
| Official Ego4D access instructions | [Start Here](https://ego4d-data.org/docs/start-here/), checked 2026-10-22 | Annotation and media acquisition require accepting the license agreement and approval for credentials. The full-scale release is over 5 TB; the official CLI can select benchmark-scoped subsets. The terms themselves could not be reviewed here. |
| Official AVD benchmark implementation | `EGO4D/audio-visual`, default `HEAD` `da19e63569758186bacfca4f53d92b211f6e373a`; README SHA-256 `a70118be22273baebdc447187617d46a722d825e28de072ef1071152627e584f` | Its README documents a first-version AVD corpus of 572 five-minute clips, 389 train, 50 validation and the remaining test; it describes separate face localization/tracking, active speaker, diarization and transcription subtasks. These are **AVD v1** counts, not verified Social `Talking-to-me` usable counts. No split files or data were fetched. |
| Official Social benchmark, TTM branch | `EGO4D/social-interactions@b2dc098c1067bd84fdf8ed5927254ca7bae22741`; README SHA-256 `f59178c09b6d455e2305a661d7d7f00ab3c114af2771d15104312078315c3c84` | The branch README specifies `av_train.json` / `av_val.json`, train/validation split lists and videos as required inputs; it says the baseline is trained from scratch. Its source reports an old validation baseline, which is not a result for this project and is not used for selection. The branch root has no `LICENSE` file at that exact revision; do not infer the default-branch license applies to this branch or its data. |
| Social repository default branch | `EGO4D/social-interactions@033fefb2b15518ebd634e034780f405994f00744`; README SHA-256 `2a90d37a6d965759eafe8e38c99a63d00bc61c587f54fa5f283c49e5d5e55dad` | Its README says data, annotations, and benchmark details will be made available later, while linking the TTM/LAM branches. This differs in maturity from the current official benchmark docs and reinforces the need to pin the actual data release before use. |

The TTM and AVD code are public source repositories, but source-code visibility
does not grant rights to the Ego4D media or annotations. We did not clone or copy
their code into this project. The historical AVD v1 corpus statistics must not
be substituted for exact TTM subset coverage or independent group counts.

## Why this task fits the Decision interface

For an annotated face track `person_id` over a fixed time interval, the product
form can be a typed binary decision:

```json
{
  "question": "Is this person speaking to the camera wearer during this interval?",
  "options": [
    {"id": "no", "label": "No"},
    {"id": "yes", "label": "Yes"}
  ]
}
```

The supplied track avoids adding an unmeasured face detector/identity model to
the resident parameter and runtime budget. If admitted, report event-level and
person/video-group metrics and probabilities; compare identical examples for
audio-only, video-only, and joint models. Keep audio availability, video
visibility, overlap, and camera-wearer identity explicit. Do not merge speech
activity with “talking to me”: the latter is a different, directed-social label.

## Admission gates

1. An authorized user reviews the actual agreement and confirms permitted uses;
   record exact agreement/version and scope. No automated acceptance.
2. Obtain only the approved train/development subset. Record annotation/data
   release revision, hashes, media IDs and source manifest hashes.
3. Reconcile current Social docs and the pinned TTM/AVD code with the actual
   annotation schema. Verify labels, missingness, intervals, face-track joins,
   audio/video timestamp mapping and availability without opening test content.
4. Split by complete grouped video and camera-wearer/session identity. Ego4D v2.1
   says 210 videos were found to belong to combined groups; enforce those groups
   across train/calibration/validation and keep test untouched. Measure actual
   group counts before deciding whether this task can meet the provisional 200
   independent-group floor.
5. Audit source/media/content overlap and pretraining overlap for every frozen
   encoder used. If overlap cannot be bounded, label it UNKNOWN and exclude the
   affected source from model selection.
6. Freeze a task-specific DecisionExample builder with caller-supplied person
   IDs and `[no, yes]` options. Run matched audio-only, video-only, joint,
   missing, stale, shuffled and time-offset controls on the same valid examples.
   A sensitivity change alone is not proof that both streams improve truth.

## Result and next step

**Source/task fit: promising. Data/license admission: not met.** Continue with
public-source review only. The next action is to obtain an authorized review of
the actual agreement and, only if permitted, inspect the official TRAIN and
development metadata without opening test annotations. Do not start a quality
fit based on the README baseline or the historical AVD v1 counts.

No model, encoder, GPU, dataset, existing worktree, or sealed audit was changed
or accessed for this source screen. No training or evaluation was run.
