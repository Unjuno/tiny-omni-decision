# Audio-Visually Indicated Actions source screen — 2026-10-26

## Decision

**REVIEW; not admitted for fitting or quality evaluation.** The source is a
small synchronized AV and acoustic-sensor dataset with 14 performed action
classes. Its task design could be converted to supplied-option decisions, but
the source is deliberately constructed so each action has a distinctive sound;
audio-only may solve much of the label. The official split code also does not
hold out the source recording or participant. Dataset access is restricted to
non-commercial research/education through an institution-linked request. No
request was sent, and no dataset payload, media or labels were accessed.

## Pinned source evidence

- Official dataset page checked 2026-10-26:
  [Audio-Visually Indicated Actions Dataset](https://pavis.iit.it/audio-visually-indicated-actions-dataset).
  It describes 378 synchronized 30–60-second AV sequences, nine people,
  fourteen actions, and three recording scenarios. Inputs include 640×480 RGB,
  12 kHz audio from 128 microphones, and beamformed acoustic images.
- The official page limits access to non-commercial research and education,
  requires a request from an email address associated with a research
  institution/university, and does not expose an open license for the media.
  The code's MIT license does not grant dataset rights.
- Official Git repository:
  [`afperezm/acoustic-images-distillation`](https://github.com/afperezm/acoustic-images-distillation/tree/c0152dc26587f814d5b891a4a813b8cd15626993),
  pinned commit `c0152dc26587f814d5b891a4a813b8cd15626993`, tree
  `ffa1e362cb239f7e78f5eec6dad7a06e40355de1`. A read-only no-checkout clone
  was made under `C:\CodexArtifacts\pretrained-reuse-r2\av-actions-code-screen`
  and checked out detached at that revision; worktree status is clean.
- The pinned README says each 30-second `Data_x` sequence is stored as 30
  one-second TFRecord shards. The pinned `utils/generate_lists.py` shuffles
  contiguous chunks (`--num_samples`, default 1) across all location/subject/
  action directories. Even at `--num_samples 10`, the three chunks from a
  single 30-second recording can be assigned to different splits. Subject and
  recording identity are not split groups. The repository also targets Python 2
  and TensorFlow 1.x, not the current runtime.

## Admission gates and consequence

Do not send an access request, download the dataset, or train unless a properly
authorized institution-linked request and the exact usage terms are reviewed.
If access is later authorized, first build recording-level and participant-
level grouped manifests from source metadata; only nine participants and three
scenarios are reported, so independent-group precision will be limited. Before
any fit, preregister audio-only, video-only, acoustic-image-only and joint
controls to test whether the labels actually require both streams. Treat this
as a narrow research pilot, not evidence for general multimodal Decision
quality or the product accuracy target.

No model, GPU, media, annotation payload, dataset artifact, or existing run was
changed or accessed. No training or test evaluation occurred.
