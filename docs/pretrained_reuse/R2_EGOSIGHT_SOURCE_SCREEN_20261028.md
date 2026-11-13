# EgoSight-Sample-500H source screen (2026-10-28)

Status: **REVIEW — no files downloaded and no fit**.

## Potential task fit

The official Hugging Face dataset card describes synchronized egocentric
multi-view video, hand/head poses, IMU, and episode/subtask descriptions. It
reports a 500-hour release and labels it CC BY 4.0. This could support a
narrow video+sensor temporal-decision task, but the card states there is no
audio stream. Free-form task descriptions must not be treated as labels or
inputs without leakage controls and a prospectively defined task.

## Unresolved admission checks

- Direct unauthenticated access to the official Hub dataset API returned HTTP
  401 during this screen. The exact current repository revision, file list,
  per-file sizes, access terms, and split/group metadata could not be
  independently verified. No account login or license acceptance was
  attempted.
- Participant consent, downstream model-training scope, and any restrictions
  beyond the card's CC BY label remain unverified.
- The reported 500-hour corpus is too large to acquire wholesale on the
  current machine. A smaller, rights-cleared train/validation subset would
  need to be selected by scene/operator/episode identity and checked for
  duplicate source recordings before any transfer.

## Decision

Keep this as a possible video+sensor source, not an admitted dataset. Revisit
only after public revision/access metadata, payload rights/consent scope, and
group-disjoint split feasibility are verified. Do not infer audio-video-sensor
fusion capability from this video+sensor-only corpus.

## Primary source

- [Official Hugging Face dataset card](https://huggingface.co/datasets/BodenAI/EgoSight-Sample-500H)
