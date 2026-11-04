# BBBD source and task screen — 2026-11-05

## Disposition

**REVIEW_NOT_ADMITTED for video + sensor Decision fitting. No dataset archive,
participant record, signal, video, or quiz-answer payload was downloaded or
opened.** The released physiological/behavioral data have a credible CC BY 4.0
grant, but the visual stimulus clips are third-party YouTube videos and their
rights for model training were not established. The source also does not yet
prove a frozen task in which both the video and sensor streams are necessary.

## Pinned source and rights

- The official Zenodo v1.1 record is
  [10.5281/zenodo.22968459](https://zenodo.org/records/22968459), published
  2026-09-25. Its public metadata API reports version `1.1`, license
  `cc-by-4.0`, and five experiment archives totaling about 27.0 GB. The
  archive metadata reports MD5 values, but no archive was downloaded, so those
  digests were not independently verified.
- The dataset authors' [license and usage page](https://bbbd.pythonanywhere.com/about-page)
  explicitly permits commercial use and adaptation of the neural,
  physiological, and behavioral measurements under CC BY 4.0, with attribution.
  The [dataset site](https://bbbd.pythonanywhere.com/) says its downloadable
  files use the same license.
- The stimuli page describes the 11 video stimuli as original YouTube videos
  and provides source links and timing maps. The Zenodo file inventory lists
  five experiment archives containing signals, behavioral data, and metadata;
  it does not list licensed video files. No evidence was found that the
  dataset's CC BY grant also licenses the underlying third-party video/audio
  works. Reusing those clips for feature extraction or model training therefore
  needs source-specific rights confirmation.
- The dataset landing page cites DOI `10.5281/zenodo.19241964`, while the
  immutable v1.1 record and API identify `10.5281/zenodo.22968459`. Use the
  latter versioned record for any future provenance; do not use the mutable
  website files without reconciling them against v1.1. The changelog says
  v1.1.0 corrected participant-answer mappings and quiz-score tables for
  Experiments 4 and 5.

## Task and split feasibility

The authors report 178 participants across five experiments, with participant
IDs restarting in each experiment. The data contain synchronized video-viewing
signals, per-video 4-choice questions, and participant responses. Domain
questions are asked before viewing; memory questions are asked after viewing.
The question bank is reused for a given video across experiments, and the
source set contains only 11 distinct video identities.

Two possible targets have different meanings:

1. Selecting the objectively correct video-quiz answer is a video-content task;
   the participant's physiological state is not needed to determine the
   answer. Including the sensor stream would not by itself establish useful
   multimodal fusion.
2. Predicting a particular participant's selected answer from the video and
   their signals is a human-response forecasting task. That is plausible, but
   it is not the same as choosing the correct answer. It needs a preregistered
   response-label contract and matched video-only, sensor-only, and perturbation
   controls before fitting.

For any later study, group subjects by `(experiment, participant_id)` because
IDs restart. Group video/question identities across experiments by the
underlying source-video identity, not local `stimNN` labels. A strict unseen-
video evaluation has only 11 source groups, and questions repeat across
participants and experiments. This is a substantial limitation for estimating
generalization to new videos and requires a split/uncertainty design before
admission. Demographic and ASRS fields are not needed for the proposed task and
should be excluded.

## Decision and constraints

- Do not use BBBD for the current video + sensor Path A fitting gate until the
  source-video rights and both-input task are resolved.
- Do not treat CC BY rights for the released signals as rights to the source
  YouTube clips.
- Do not download the 27 GB archive set under current storage conditions; C:
  had about 85 MiB free at screening time. No cleanup, cache creation, GPU work,
  or existing-run intervention was performed.
- A sensor-only behavioral study could be separately screened, but it would not
  establish the goal's video + sensor joint-input requirement.

## Evidence inspected

- [Official BBBD overview](https://bbbd.pythonanywhere.com/)
- [Official experiment descriptions and synchronization](https://bbbd.pythonanywhere.com/experiments-page)
- [Official quiz structure](https://bbbd.pythonanywhere.com/questionnaires-page)
- [Official stimulus identity and timing map](https://bbbd.pythonanywhere.com/stimuli-page)
- [Official v1.1.0 changelog](https://bbbd.pythonanywhere.com/changelog)
- [Official BBBD v1.1 Zenodo record](https://zenodo.org/records/22968459)
- [Scientific Data article](https://www.nature.com/articles/s41597-026-07215-1)
