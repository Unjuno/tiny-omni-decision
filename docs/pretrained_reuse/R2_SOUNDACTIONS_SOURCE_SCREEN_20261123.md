# 365 Sound Actions v2 source/task screen — 2026-11-23

## Disposition

**Rights: ALLOW for research and commercial use with CC BY 4.0 attribution.
R2-M joint-fit admission: NOT ADMITTED.** The source is a promising, small,
natural synchronized audio/video collection, but its published task is
primarily an audio-perception label and its prior modality comparison found
audio-only ahead of audio/video. It therefore does not pass the current R2
gate for a task where both modalities are genuinely required. No model or AV
media was loaded, and no training or evaluation was performed.

It may be retained as a possible narrow representation diagnostic if the
roadmap later admits non-complementarity probes. It must not be described as
evidence that the product can jointly reason over audio and video.

## Pinned source and local inspection

- Publisher record: [Zenodo v2](https://zenodo.org/records/23971067), DOI
  `10.5281/zenodo.23971067`, published 2026-11-05. The record states CC BY 4.0;
  its included `LICENSE.txt` explicitly permits commercial sharing/adaptation
  with attribution and indication of changes.
- Contents: 365 synchronized sound-producing action recordings, each with
  audio and video; 365 stills; and three independent annotators' labels. The
  full source is about 2,288 MB. The v2 release preserves the same 365 source
  items as v1 but restores most videos to the phone's original 60-fps timing;
  030 remains a 25-fps damaged-camera exception.
- The v2 label archive includes the publisher's correction script and
  `label-corrections.csv`. The README says older OSF copies and the earlier
  paper contain stale consensus cells; use the current v2 labels if this
  source is ever revisited.
- Local screen output:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\soundactions-source-screen-v2-20261123-attempt-02\`.
  Only the small `overview.csv`, `sound-actions-labels.zip`, `statistics.json`,
  `SHA256SUMS.txt`, `README.md`, and `LICENSE.txt` were retrieved. The audio,
  video, and still-image archives were not downloaded. Zenodo's MD5 values
  matched for all six retrieved files; local SHA-256 values are recorded
  below.

| File | Bytes | SHA-256 |
|---|---:|---|
| `overview.csv` | 126,126 | `6e1307f24b696ef8079374db1f4102dbf1b1028a1024d0b714316dbc919a2de5` |
| `sound-actions-labels.zip` | 58,315 | `c08d18a856c3ef10879e7a7c4b6b95d5ddc05f1f878670ee309013440e4ad601` |
| `statistics.json` | 857 | `1e01d6285f449b8306735379d3ec7eeb51262b6cbbddc110eca87542139b06d0` |
| `SHA256SUMS.txt` | 118,025 | `9648b000cfff5e9295ec59ae39346486e68362733d9fe89cfbd68e1d94bffec0` |
| `README.md` | 8,194 | `9aa763e114d398f87bf16e5c3df0738f3fdcb6c53f863b710ee22be1496b1c94` |
| `LICENSE.txt` | 345 | `c517ddfa462d4ee86c5f5ddc13ebfd3476c8e881f83558e7b1d4bf19ea99ae63` |

## Task and grouping evidence

The target `perception_type` is a three-class description of sound shape
(impulsive, iterative, sustained); it is not an answer label that requires
matching a sound to a visual event. The v2 annotation file has 365 unique
recording IDs: 126 impulsive, 125 iterative, 83 sustained, and 31 `unknown`
(no majority). Thus only 334 records have a consensus target. Of all 365,
158 have unanimous labels, 176 have two-of-three agreement, and 31 have no
majority. The publisher reports free-marginal multi-rater κ = 0.46.

A metadata-only grouping check found 365 distinct exact normalized
object/action annotation pairs and 365 distinct exact action sets; there were
362 exact object sets. This suggests the rows are not exact repeats of the
same annotated action, but it does not establish independent recording
conditions. All 365 actions were recorded by one creator, Alexander Refsum
Jensenius, over one year. A split by recording ID would hold out media assets
but would not test transfer to a new recorder or population. No official
train/validation/test split is provided.

The accompanying 2026 paper ran five-fold experiments on these same 365 items
with DG-SCT, a 460M-parameter AV model. For the `PerceptionType` target, its
frozen-backbone/final-head result was 0.54 with AV versus 0.55 audio-only and
0.44 video-only. With adapters trainable, it reported 0.60 AV versus 0.66
audio-only. Those are the authors' cross-validation results, not this
project's measurements, and they do not establish a video benefit. The paper
also describes the dataset as too small for modern transformer
generalization. The separate `Enjoyability` labels have still weaker
agreement (κ = 0.20) and are unsuitable as a replacement target.

## R2 decision

Do not fit a joint AV model on this source under the current roadmap's
both-required admission gate. It is rights-cleared, but its target is
audio-dominant, its labels are moderately noisy, its source diversity is one
creator, its public examples have already been used in published
cross-validation, and the prior AV arm did not outperform the audio-only arm.
This source screen does not close R2-M or the multimodal quality gates.

If revisited for a bounded development probe, freeze the exact v2 split by
recording ID and normalized action/object identity before loading media, keep
all questions derived from a clip in that clip's split, use the corrected v2
labels, report a majority-label baseline, and include audio-only, video-only,
AV, shuffled-video, and missing-modality controls. Any result must be called
development evidence and cannot serve as an independent release gate.

## Primary references

- [Zenodo v2 record and file manifest](https://zenodo.org/records/23971067)
- [Zenodo v2 metadata API](https://zenodo.org/api/records/23971067)
- [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode)
- [Guo, Tørresen & Jensenius (2026), article](https://doi.org/10.5334/tismir.223)
