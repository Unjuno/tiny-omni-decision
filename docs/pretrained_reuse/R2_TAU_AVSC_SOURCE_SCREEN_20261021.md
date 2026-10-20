# TAU Urban Audio-Visual Scenes 2021 source screen — 2026-10-21

## Decision

**REVIEW / not admitted to training.** TAU-UAVS is a well-documented,
synchronized audio/video scene-classification dataset, and its original paper
reports gains for joint modeling over its unimodal systems. However, Zenodo's
record metadata identifies the license as `other-nc` (non-commercial), the
development archive is 107.7 GB, and scene labels do not guarantee that both
signals are necessary for each example. Do not acquire the archive or start a
fit for this product-oriented R2 path on the current evidence.

## Primary-source findings

- The DCASE 2021 task page describes synchronized recordings from 10 acoustic
  scenes, with 10-second audio/video files and one-second decisions. It asks
  that all segments from a location remain in the same split. The development
  archive is 107.7 GB; the held-out evaluation data has no released labels.
- The same page reports the published case study at 84.4% joint accuracy,
  compared with 76.8% audio-only and 70.0% video-only. This supports that both
  modalities can help on average, but does not establish per-example joint
  necessity or the project's conditional option-ranking task.
- The official Zenodo record is open-access but its API metadata gives
  `metadata.license.id = other-nc`. Open download access is not a commercial
  training license. No license clarification was sought and no files were
  downloaded or opened.
- DCASE also names TAU Urban Audio-Visual Scenes 2021 Evaluation (61.3 GB), but
  its reference labels are not released. It cannot provide a labeled blind
  evaluation without a coordinator-provided scoring route.
- The related RAVEL and CAVA repositories describe research/non-commercial
  access. They are not replacement product-training sources under the current
  rights gate.

## Reproduction and scope

This was a read-only official-source check. The Zenodo record API was queried
for metadata and file sizes only. No audio, video, labels, evaluation data, or
sealed audit was read. No model, corpus, run, or checkpoint was changed; no GPU
work was started.

## Next gate

Continue screening for a bounded, rights-compatible source with fresh grouped
development identities and labels suitable for the actual decision interface.
Require a task-level intervention showing when audio and video are each useful
before training a fusion readout. If no such source passes, retain the existing
Perception Test scale result as narrow development evidence and do not spend
GPU time on another run over the already-observed validation cohorts.

## Sources

- [DCASE 2021 Acoustic Scene Classification, Subtask B](https://dcase.community/challenge2021/task-acoustic-scene-classification)
- [Official Zenodo development record](https://zenodo.org/records/4477542)
- [Official Zenodo evaluation record](https://zenodo.org/records/4767103)
- [RAVEL corpus page](https://perception.inrialpes.fr/datasets/Ravel/)
- [CAVA dataset page](https://perception.inrialpes.fr/Free_Access_Data/Data/CAVA_Dataset/index.html)
