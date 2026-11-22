# Fresh audiovisual source gate — 2026-11-22

## Decision

Do not launch another audiovisual fit from the currently measured data. The
two admitted paired audio/video development tasks have not shown a reliable
fusion advantage, and their held-out cohorts have already influenced model
selection. The latest audio+generated-screen pilot is a different, narrow
mechanism test; it does not supply a fresh real-world AV cohort. R2-M remains
open pending a rights-cleared, adequately grouped, task-matched source.

## Evidence already in the repository

- **Lombard GRID:** On the same previously exposed source validation, audio-only
  achieved 97.94% macro slot Accuracy, video-only 30.15%, and joint 97.09%.
  The paired joint-minus-audio interval was wholly negative. The video bank is
  spatial/time-pooled and the selected task is closed-vocabulary speech slots.
  Do not repeat this comparison or infer a universal video limit. See the
  [audited result](R2_LOMBARD_GRID_RESULTS_20261014.md).
- **365 Sound Actions v2:** Five-fold recording-grouped TRAIN-only CV measured
  68.29% audio-only, 40.65% video-only, and 67.07% gated A/V Accuracy; the
  paired change interval includes zero and the fusion calibration metrics were
  worse. A separate fixed temporal pilot on already used development rows
  found 64.58% audio, 43.75% video, and 52.08% A/V. Stop tuning this small,
  single-recordist task. The 46-row sealed audit remains closed. See the
  [TRAIN-only CV](R2_SOUND_ACTIONS_TRAIN_ONLY_GROUP_CV_20261121.md) and
  [development result](R2_SOUND_ACTIONS_TEMPORAL_RESULTS_20261121.md).
- **HubEgo-100K:** Promising synchronized video+sensor volume, but payload
  access requires the user to approve access on their own account. The labels
  are machine generated and not human audited. No form or payload was used;
  retain the existing access gate in
  [the source screen](R2_HUBEGO100K_VIDEO_IMU_SOURCE_SCREEN_20261121.md).

## New official-source screens; no payload acquired

- **VGGSound:** The official repository publishes YouTube IDs, timestamps,
  labels, and a train/test split and states CC BY 4.0 for the dataset. It also
  says copyright remains with original video owners. Its released labels are
  sound classes and do not by themselves establish that both modalities are
  necessary for a supplied-option Decision. Keep unadmitted until the exact
  media-use scope, group independence, and a both-required task are verified.
  [Official repository](https://github.com/hche11/VGGSound).
- **AVSpeech:** The publisher download page states CC BY 4.0, but the video
  media is sourced from YouTube and downstream media rights / platform terms
  are not resolved by that page alone. The 1.49-TB mirror is not a bounded
  local candidate. No rows or media were fetched.
  [Official project page](https://looking-to-listen.github.io/avspeech/),
  [media-bearing mirror rights notes](https://huggingface.co/datasets/ProgramComputer/avspeech-visual-audio).
- **Audio-Visually Indicated Actions:** The publisher describes 378 aligned AV
  sequences, but access requires a signed agreement limiting use to the
  requesting user's research project. That does not fit the product-directed
  rights gate. Do not submit the form or download data.
  [Official task and access page](https://pavis.iit.it/audio-visually-indicated-actions-dataset),
  [official request form](https://www.iit.it/documents/1452990/1488719/Request-Form-AV-Actions-Dataset.pdf/e593a105-5823-cf39-7f5b-a06e41f7cbba).
- **VoxDub:** The publisher labels the dataset CC BY-NC-SA 4.0 and states the
  underlying media rights stay with original creators. Exclude it from this
  product-directed path.
  [Official project page](https://voxdub-dataset.github.io/VoxDub/).

## Next admission rule

Continue source screening without opening any sealed audit. A candidate may
advance to payload preflight only when all of the following are demonstrated:

1. Explicit rights apply to the actual audio and video assets for the intended
   product research; a code or annotation license is insufficient.
2. Immutable source revision, actual source/group identities, and enough
   non-overlapping TRAIN / selection / calibration / development groups are
   observable before labels or scores are used.
3. A predeclared supplied-option Decision is solvable from both streams and
   has plausible matched unimodal baselines; metadata-only predictability is
   measured before fitting.
4. Input mismatch, missing-stream, synchronization-offset, and label-quality
   controls are feasible without reusing an already exposed evaluation cohort.
5. Local media/cache/runtime requirements fit the existing C-drive reserve;
   no cloud compute, form submission, or access-term acceptance is implied.

If no source passes, continue work on independently admitted single-modality
tasks and the pretrained runtime/parameter audits. Do not retry audio/video
fusion on the observed cohorts or treat high accuracy on the generated-color
screen task as evidence that the fresh-AV gate passed.
