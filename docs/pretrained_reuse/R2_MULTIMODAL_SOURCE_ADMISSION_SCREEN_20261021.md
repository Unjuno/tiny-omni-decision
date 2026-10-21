# R2 multimodal source admission screen — 2026-10-21

Status: **screen only; neither source is admitted to a local quality run**.
No dataset content was downloaded, opened, or evaluated. No access request was
sent. No model was trained or run.

## Zenseact Open Dataset (ZOD)

Official sources: [project overview](https://zod.zenseact.com/),
[license](https://zod.zenseact.com/license/), [download/access](https://zod.zenseact.com/download/),
[sequence annotations](https://zod.zenseact.com/sequences/).

- The publisher describes camera, LiDAR, radar, GNSS/IMU, and vehicle data.
  Its Sequences subset has 1,473 twenty-second scenes; Drives has 29 longer
  recordings; Frames has 100,000 curated images with about two seconds of
  associated sensor data.
- The publisher states CC BY-SA 4.0 for the dataset and provides an attribution
  and privacy notice that must accompany public use.
- The official download procedure requires an emailed access request before
  downloading. No request was sent and no access URL is available in this
  workspace.
- Sequence annotations currently cover a central keyframe, not dense temporal
  labels. The inspected annotation inventory does not establish a task whose
  ground truth requires both camera and sensor streams. Scene synchronization
  and broad group support do not by themselves establish sensor complementarity.
- Pretraining overlap for any candidate frozen encoder is UNKNOWN. A future
  screen would also need split-by-drive/source integrity and a predeclared
  conditional task that defeats image-only and sensor-only shortcuts.

Disposition: potentially relevant to video-plus-sensor engineering, but access,
task necessity, and pretrained-source overlap are unresolved. Do not use it to
select a model or claim joint quality.

## Audio-Visually Indicated Actions (IIT)

Official source: [dataset page](https://pavis.iit.it/audio-visually-indicated-actions-dataset).

- The publisher describes 378 synchronized 30–60 second sequences, 9 people,
  14 sound-producing actions, three recording scenarios, RGB video, and
  128-channel audio. The source is deliberately constructed so the actions
  produce characteristic sounds; the existence of both streams does not show
  that both are needed for its action labels.
- Access is offered to researchers and educators for non-commercial research
  or education after submitting a request from an address linked to a
  research institution or university. The page does not give an open direct
  download or a reusable data license for the underlying recordings.
- Nine participants are inadequate for the project's provisional 200-group
  quality gate. No participant-independent task split or media hashes were
  verified.

Disposition: not admitted. It would require an authorized institutional access
path and a new, explicitly both-required task; even then, its participant count
would support only a low-power research diagnostic.

## R2 decision

Neither source currently provides an executable, independently grouped task
that can justify a Path A AV or sensor-fusion quality run. Keep the existing
synthetic CLEVRER timing result, PCH sensor readouts, Perception Test controls,
and native Lombard correspondence results in their original scopes. Do not
repeat their observed validation or infer general multimodal quality from them.
Continue source/task admission; revisit either candidate only if access,
source-level grouping, pretrained-overlap policy, and both-input necessity can
be established before any model selection.
