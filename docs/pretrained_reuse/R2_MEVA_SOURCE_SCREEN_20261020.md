# MEVA source screen — 2026-10-20

## Decision

**Do not admit MEVA to an R2 fit or checkpoint-selection loop yet.** It is a
plausible natural-video activity-presence task, but its public train and
evaluation-level annotations share all seven capture dates and almost all
date-camera groups. A video-level or clip-level split would therefore not
provide the independent scene/session evidence required for this project.
The annotation repository could not be opened through its web interface, so
we used a shallow, blob-filtered Git metadata checkout and inspected file
paths only. We did not inspect annotation contents or acquire media.

## What the official source establishes

- The MEVA site says the dataset is licensed CC BY 4.0 and provides training-
  level and evaluation-level activity annotations. It describes 120.2 hours
  / 1,443 five-minute clips of training-level annotation and 64 hours / 769
  clips of evaluation-level annotation.
- The training README says the source has roughly 330 hours and 4,001 video
  clips, and explicitly excludes `drop-4-hadcv22` from training for NIST
  ActEV submissions. This project would not use that drop.
- The license is limited to rights the licensor can grant; it says publicity,
  privacy, and similar personality rights are not licensed. The source paper
  reports IRB oversight, but this screen does not treat that alone as a
  blanket product-use or privacy determination.
- The activity annotations can plausibly be converted into a binary
  video-conditioned DecisionExample (activity present / absent in a fixed
  interval). This is a narrow activity-detection task, not general video
  reasoning and not an audio/video task.

Primary sources:

- MEVA dataset site: <https://mevadata.org/index.html>
- MEVA data README: <https://mevadata.org/resources/README-meva-kf1-data.html>
- MEVA license: <https://mevadata.org/resources/MEVA-data-license.txt>
- NIST ActEV SRL data and split instructions: <https://actev.nist.gov/srl>
- MEVA paper: <https://openaccess.thecvf.com/content/WACV2021/papers/Corona_MEVA_A_Large-Scale_Multiview_Multimodal_Video_Dataset_for_Activity_Detection_WACV_2021_paper.pdf>

## Metadata-only split audit

At source revisions `421841a75577b697c314e952e585aecbb1b99e17`
(`meva-data-repo`) and `3d1a6cade7d825d65eb0d28bd79bd4e78d01b320`
(`actev-data-repo`), the MEVA tree contains 1,443 `*.activities.yml` paths
under `kitware-meva-training` and 769 under `kitware`, matching the official
README's counts. Comparing filename and directory metadata only:

- the two groups share all seven dates: 2018-03-05, 03-07, 03-09, 03-11,
  03-12, 03-13, and 03-15;
- 142 of 146 evaluation-level date-camera groups also occur in
  training-level annotations;
- exact clip-basename overlap is zero, but adjacent clips and the shared
  date-camera capture groups mean clip-level non-overlap is insufficient.

The official MEVA GitLab web pages returned 403 in this environment. The
metadata checkout is shallow and blob-filtered; only path names were
enumerated. A contributor directory with `test` in its path appeared in the
tree listing, but no file from it was read. No NIST ActEV test partition,
labels, media, or sealed audit was opened.

## Admission gate and next action

MEVA remains **REVIEW / NOT ADMITTED**. Reconsider only after a bounded
train/development subset is constructed from allowed sources and passes all
of the following without relying on the official overlapping split:

1. group isolation at least by capture date/session and underlying scenario;
2. no cross-split camera-time, asset, normalized-content, or annotation-ID
   overlap;
3. written task conversion and a matched video-conditioned versus
   prompt-only baseline;
4. a specific use and privacy review for actor footage and sensitive activity
   labels; and
5. a bounded download/feature plan that fits the target local device and
   storage budget.

Until then, no HotUGPU reservation is justified for MEVA. The live RTX 3080
was occupied by Orbitune at the time of this screen; its process was not
stopped, signaled, or modified. No model, corpus, checkpoint, Teacher, or
existing run was changed. No cloud compute or paid resources were used.
