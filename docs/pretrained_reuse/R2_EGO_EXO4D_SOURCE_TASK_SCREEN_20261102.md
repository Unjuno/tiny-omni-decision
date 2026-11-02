# Ego-Exo4D source/task screen — 2026-11-02

**Disposition: `CONDITIONALLY_PROMISING_FOR_VIDEO_M1`; not admitted for data
use.** This screen used only the publisher's public documentation and license
template. No agreement was accepted, no metadata/annotation files or media were
downloaded, and no model was run.

## Why it may fit a video-only Decision benchmark

The official overview reports Ego-Exo4D V2 at 5,035 takes, more than 800
participants, 131 natural scene contexts, and 1,286.3 hours of video. It
describes synchronized first-/third-person video, seven-channel audio, IMU,
gaze and 3D environment data. Publisher metadata is documented with
`participant_uid`, `capture_uid`, `take_uid`, task fields, and validated
time-sync fields.

The official Demonstrator Proficiency benchmark is a concrete fixed-choice
video task: predict one of four labels (Novice, Early Expert, Intermediate
Expert, Late Expert) from egocentric video, optionally with synchronized
exocentric video. The benchmark explicitly excludes audio, gaze, text/narration,
and subject information. Therefore this is a possible **video M1** task only;
it provides no evidence for AV or sensor fusion and must not be presented as an
M2 test.

The publisher reports enough top-level participant/take volume to justify a
metadata feasibility check after access, but the actual label coverage,
participant-disjoint split counts, and valid evaluation sample count remain
unknown. Entry gates remain at least 1,000 fixed evaluation decisions and at
least 200 independent participant groups, with all split assignments frozen
before fitting. The official docs show identity/time-sync fields but do not
document a participant-disjoint benchmark split.

## License and access gate

The publisher requires review and acceptance of its custom agreement before
access to the dataset **or annotations**. The official template grants research,
training, evaluation, and product-development use for object/event/activity
understanding, explicitly including commercial product development. It also
states that the database is proprietary, cannot be sublicensed or provided to
third parties, and that dataset content may not appear in a program, dataset,
or product. It preserves the licensee's IP in derived models, subject to the
agreement. Credentials are issued after acceptance (typically about 48 hours);
the agreement page reports no fee. This is a legally binding, individual or
authorized-organization agreement, not an open data license. **No one accepted
or signed it in this screen.**

The publisher's default recommended download is approximately 14 TiB. The CLI
supports selecting parts, including annotations, metadata, and takes, but the
size of a minimal eligible subset or pre-extracted feature set has not been
verified. The observed C: free space during this screen was 1,402,617,856 bytes;
no download should begin before a size and storage-reserve preflight.

## Decision and next gates

- Keep the candidate unadmitted until the user personally reviews and accepts
  the official agreement. Do not submit the form or sign on the user's behalf.
- After access, inspect only allowed metadata first. Freeze a participant-
  disjoint train/validation/sealed split and count labels/groups before any
  media retrieval. Do not load sealed records during development.
- Continue only if validation has at least 1,000 decisions and 200 participants,
  label coverage is sufficient, media-feature rights and subset size are clear,
  and the selected video-feature source is itself license/runtime-audited.
- The proficiency task is explicitly video-only. Do not use it to claim audio,
  sensor, or multimodal necessity. A separate M2 task would need its own
  protocol and shortcut/ablation controls.

## Publisher references

- [Overview and release scale](https://docs.ego-exo4d-data.org/)
- [Proficiency Estimation task definition](https://docs.ego-exo4d-data.org/benchmarks/proficiency_estimation/)
- [Metadata schema](https://docs.ego-exo4d-data.org/data/metadata/)
- [Aria trajectory/MPS documentation](https://docs.ego-exo4d-data.org/data/mps/)
- [Access and license requirements](https://docs.ego-exo4d-data.org/getting-started/)
- [Official license template](https://ego4d-data.org/pdfs/Ego-Exo4D-Model-License.pdf)

No audit set was created or read. Local GPU use, cloud use, and media bytes:
none.
