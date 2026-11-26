# RoboMNIST episode-level joint-task feasibility — 2026-11-03

## Decision

RoboMNIST is a rights-cleared candidate for a **whole-repetition** audiovisual
Decision task. The official article explicitly describes the dataset's
episode-level synchronization and clarifies that `UNC###` encodes motion
uncertainty, not participant or asset identity. This corrects the tempting but
incorrect reading of that filename suffix as a group/person ID.

This does **not** establish a per-sample absolute audio-to-frame offset from the
local WAV/MP4 files. Use whole-clip features only; do not claim word/frame
alignment. This source does not provide scene-, room-, or device-held-out
generalization. No new feature extraction, fitting, or model evaluation is
reported here.

## Source evidence

The official [Scientific Data article](https://doi.org/10.1038/s41597-025-04636-2)
defines `U` as end-effector motion uncertainty; `UNC258` denotes `U = 2.58`.
It reports 60 primary action/robot/velocity combinations and at least 32
repetitions per combination. It states that video is synchronized with CSI and
that audio start/end times are synchronized with the other modalities. The
official dataset metadata is CC BY 4.0; see the pinned v1 source screen and
receipt in `R2_ROBOMNIST_SOURCE_20261014.md`.

The existing pinned 60-archive directory audit counted 2,135 complete
repetition groups, 32–44 per combination, and 17,080 media/supervision files.
Every repetition group contains three camera files, three microphone files,
one CSI file, and one position-supervision file. The earlier native input smoke
measured 450 video frames and 450 CSI timestamps over approximately 15 seconds;
the WAV files have no absolute start clock. The separate PositionGT clock scan
also found no timestamp field. These local format observations remain valid;
the article adds author-level support for **episode** association, not a
measured sub-frame audio offset.

`UNC` values recur across different primary combinations by design. They are
noise/uncertainty conditions, not repeated underlying media. Treat each full
primary-combination/repetition record as the source group and keep all of its
camera, microphone, CSI, and position files together. Never feed `UNC`, `Rp`,
source filenames, or position supervision to the model. The group keys and
publisher labels are split/accounting metadata only.

## Candidate task and controls

Candidate task: from a complete recording, answer **“Which digit did the robot
draw?”** with the fixed ten options `0` through `9`; the activity factor `Ac`
supplies the target. The proposed first joint comparison is one video view plus
its same-module audio (`Rx1`), with video-only, audio-only, question-only, and
within-split shuffled-input controls. CSI remains an optional follow-up rather
than silently enlarging the first candidate. This is a narrow controlled-robot
task, not broad Omni quality evidence.

If this candidate is frozen, the validation must contain at least 1,000 groups
for a provisional `PASS` decision. The existing global-`Rp` length diagnostic
has only 660 validation groups, so it cannot serve that purpose. Any new
development split must be preregistered, hash-ordered within each primary
combination, preserve the previously observed `Rp01` group on the training
side, keep every full repetition intact, and be described as **development**:
the same public source records were previously used in a metadata-only length
diagnostic. No result from such a split would be a blind or scene-OOD result.

## Transfer and local-capacity check

Using the existing ZIP central-directory compressed sizes for all 2,135 groups:

| Selected members | Compressed bytes | GiB | Linear time at prior native-smoke throughput |
| --- | ---: | ---: | ---: |
| One camera (`Rx1`) + one microphone (`Rx1`) | 17,536,296,858 | 16.332 | 34.6 h |
| One camera + one microphone + CSI | 20,729,339,672 | 19.306 | 40.9 h |

The time projection uses the previously measured one-repetition Figshare input
smoke (26,173,462 response bytes in 186.024 seconds, about 0.134 MiB/s). It is
an estimate, not a new throughput benchmark; remote throttling, request
overhead, and decode time may change it. C: had about 7.08
GiB, so retaining even the one-camera/one-microphone source set locally would
not fit while preserving the 2 GiB reserve. Any future full-quality attempt
would need bounded streaming feature extraction, small feature-only outputs,
and a measured transfer dry run before committing to the multi-day network
cost. The current PhysicalAI integrity acquisition remains untouched and no
competing download was started.

## Status

- Rights: allowed with attribution under the pinned dataset license.
- Episode-level AV association: supported by the dataset authors; suitable for
  whole-clip, not frame-aligned, use.
- Repetition grouping: available for all 2,135 records; full media/content
  overlap and scene independence are not established by ZIP metadata alone.
- Candidate split/task: not frozen or created.
- Model training/evaluation: not run; no quality claim.
- Local cost: $0; no GPU or cloud compute used in this feasibility check.
